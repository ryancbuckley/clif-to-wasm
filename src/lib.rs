//! A WebAssembly backend for Cranelift.
//!
//! Translates a Cranelift [`ir::Function`] into a WebAssembly function body.
//! Rather than implementing `TargetIsa` (which centers on register allocation),
//! this crate operates directly on CLIF in five stages:
//!
//! 0. **Preparation & Legalization** (`legalize_i128`, `legalize_abi`):
//!    resolves value aliases, splits `i128`/`f128` values into `(lo, hi)` `i64`
//!    pairs, and optionally demotes multi-result returns to memory.
//! 1. **Analysis** (`analysis`): computes reachability, RPO, CFG edges,
//!    dominators, loop headers, merge nodes, and single-pass backward liveness
//!    and use counts.
//! 2. **Expression trees** (`expr_trees`): folds single-use definitions into
//!    their consumer within each block when side-effect order is preserved
//!    (checked in O(1) via effect colors). Constants are rematerialized at each
//!    use.
//! 3. **Local allocation** (`locals`): assigns wasm locals to remaining
//!    values by coloring per-block live-range segments, preferring to coalesce
//!    block arguments with their target parameters.
//! 4. **Structuring and emission** (`structure`, `emit`, `lower`):
//!    reconstructs structured control flow from the dominator tree via Ramsey's
//!    "Beyond Relooper" (ICFP 2022) and lowers instructions with a `local.tee`
//!    peephole. Irreducible CFGs fall back to a dispatch loop.
//!
//! More expensive size passes (cross-block sinking, value-carrying blocks,
//! local frequency sorting) are left to `wasm-opt`.
//!
//! # Example
//!
//! Compile a function and assemble it into a module with [`ModuleBuilder`]:
//!
//! ```
//! use clif_to_wasm::{Config, Env, Error, FuncIndex, ModuleBuilder, Name, Result};
//!
//! let funcs = cranelift_reader::parse_functions(
//!     "function %add(i32, i32) -> i32 {
//!      block0(v0: i32, v1: i32):
//!          v2 = iadd v0, v1
//!          return v2
//!      }",
//! )
//! .unwrap();
//!
//! // `%add` calls nothing, so its environment resolves no names.
//! struct NoCalls;
//! impl Env for NoCalls {
//!     fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex> {
//!         Err(Error::Env(format!("unknown function {name}")))
//!     }
//! }
//!
//! let config = Config::default();
//! let mut builder = ModuleBuilder::new();
//! let (params, results) = clif_to_wasm::wasm_signature(&funcs[0].signature, &config.module)?;
//! let add = builder.declare_function(&params, &results);
//! let compiled = clif_to_wasm::compile_function(&funcs[0], &config, &mut NoCalls)?;
//! builder.define_function(add, compiled);
//! builder.export("add", add);
//! let wasm = builder.finish();
//! # wasmparser::validate(&wasm).unwrap();
//! # Ok::<(), Error>(())
//! ```

#![warn(
    missing_docs,
    missing_debug_implementations,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use cranelift_codegen::ir::{self, ExternalName, LibCall, Signature, UserExternalName};
use wasm_encoder::ValType;

// Re-exported because their types appear in this crate's API, so users can
// name them without depending on matching versions.
pub use cranelift_codegen;
#[cfg(feature = "module")]
pub use cranelift_module;
pub use wasm_encoder;

mod analysis;
mod emit;
mod expr_trees;
mod legalize_abi;
mod legalize_i128;
mod locals;
mod lower;
mod lower_atomic;
mod lower_float;
mod lower_simd;
pub mod module;
pub mod object;
#[cfg(feature = "module")]
pub mod object_module;
mod structure;

pub use module::{
    Export, FuncIndex, GlobalIndex, MemoryIndex, ModuleBuilder, TableIndex, TagIndex, TypeIndex,
};
pub use object::SymbolId;

/// Errors that can occur during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    /// The function uses a feature this backend does not (yet) support.
    Unsupported(String),
    /// The environment could not resolve a reference.
    Env(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Env(s) => write!(f, "environment error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// A `Result` whose error type is this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn unsupported<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(msg.into()))
}

/// Code generation settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Config {
    /// Target module layout; must match the surrounding wasm module.
    pub module: ModuleLayout,
    /// Wasm features supported by the target engine.
    pub engine: EngineFeatures,
    /// Output-quality switches (all enabled by default).
    pub opts: Optimizations,
}

/// Layout of the wasm module that will contain compiled functions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ModuleLayout {
    /// The address type of linear memory, which is also the wasm type of the
    /// shadow stack pointer.
    pub address_type: AddressType,
    /// Linear memory index accessed by loads and stores (`MemoryIndex(0)` in
    /// single-memory modules).
    pub memory_index: MemoryIndex,
    /// Function table used by `call_indirect` (`Some(TableIndex(0))` in
    /// single-table modules). When `None`, `call_indirect` is rejected.
    pub table_index: Option<TableIndex>,
    /// Mutable global holding the 16-byte-aligned, downward-growing shadow
    /// stack pointer. When `None`, functions requiring stack slots are rejected.
    pub stack_pointer_global: Option<GlobalIndex>,
    /// Mutable `i64` global backing `get_pinned_reg`/`set_pinned_reg`. When
    /// `None`, functions using the pinned register are rejected.
    pub pinned_reg_global: Option<GlobalIndex>,
    /// 4-byte-aligned byte address in linear memory of a 256-byte (64 × `i32`)
    /// striped spinlock table for `i128` atomics, disjoint from program data
    /// and the shadow stack. When `None`, `i128` atomics are rejected.
    pub atomic_lock_table_addr: Option<u64>,
    /// Mutable global holding the current thread's TLS base address
    /// (`__tls_base`). When `None`, `tls_value` is rejected.
    pub tls_base_global: Option<GlobalIndex>,
    /// Emit code for a relocatable object file ([WebAssembly linking
    /// conventions]) instead of a finished module.
    ///
    /// Indices returned by [`Env`] and globals/tables in this struct then refer
    /// to symbol table entries (type indices still refer to the type section),
    /// encoded as padded LEB128s and recorded in [`CompiledFunction::relocs`].
    /// See [`crate::object`].
    ///
    /// [WebAssembly linking conventions]: https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md
    pub relocatable: bool,
    /// Demote multi-value returns to a hidden first pointer parameter (LLVM's
    /// Basic C ABI for WebAssembly). When enabled, functions returning more
    /// than one wasm value (after splitting `i128`/`f128`) write results to a
    /// caller-provided C-layout return area and return nothing. When disabled,
    /// wasm multi-value returns are used.
    pub demote_multi_results: bool,
}

impl ModuleLayout {
    /// Wasm value type for memory addresses.
    pub(crate) fn ptr_ty(&self) -> ValType {
        self.address_type.val_type()
    }
}

/// The address type of a linear memory: `i32` for 32-bit memories, `i64` for
/// 64-bit ones (`memory64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressType {
    /// 32-bit addresses (wasm32).
    #[default]
    I32,
    /// 64-bit addresses (wasm64, the `memory64` proposal).
    I64,
}

impl AddressType {
    /// The wasm value type of an address.
    pub fn val_type(self) -> ValType {
        match self {
            AddressType::I32 => ValType::I32,
            AddressType::I64 => ValType::I64,
        }
    }
}

/// Wasm features available in the target engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineFeatures {
    /// Emit wasm tail calls (`return_call` / `return_call_indirect`). When
    /// false, tail calls lower to a normal call followed by `return`.
    pub tail_calls: bool,
}

impl Default for EngineFeatures {
    fn default() -> Self {
        EngineFeatures { tail_calls: true }
    }
}

/// Codegen optimization switches (all enabled by default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Optimizations {
    /// Fold single-use values into expression trees on the operand stack
    /// (`expr_trees.rs`). When false, every non-constant value uses a local.
    pub stack_values: bool,
    /// Coalesce block arguments with target block parameters into a shared
    /// local to eliminate moves.
    pub coalesce: bool,
    /// Fuse `local.set x; local.get x` into `local.tee x`.
    pub use_tee: bool,
}

impl Optimizations {
    /// Disable all optional optimizations.
    pub fn none() -> Self {
        Optimizations {
            stack_values: false,
            coalesce: false,
            use_tee: false,
        }
    }
}

impl Default for Optimizations {
    fn default() -> Self {
        Optimizations {
            stack_values: true,
            coalesce: true,
            use_tee: true,
        }
    }
}

/// A resolved external name, passed to the [`Env`] callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Name<'a> {
    /// A user-defined name, from the function's table of named functions.
    User(&'a UserExternalName),
    /// A test-case name (`%name` in CLIF text).
    TestCase(&'a str),
    /// A Cranelift runtime library call.
    LibCall(LibCall),
    /// A well-known symbol, such as the ELF global offset table.
    KnownSymbol(ir::KnownSymbol),
}

impl<'a> Name<'a> {
    pub(crate) fn resolve(func: &'a ir::Function, name: &'a ExternalName) -> Name<'a> {
        match name {
            ExternalName::User(r) => Name::User(&func.params.user_named_funcs()[*r]),
            ExternalName::TestCase(t) => {
                Name::TestCase(std::str::from_utf8(t.raw()).unwrap_or("<invalid>"))
            }
            ExternalName::LibCall(l) => Name::LibCall(*l),
            ExternalName::KnownSymbol(k) => Name::KnownSymbol(*k),
        }
    }
}

impl std::fmt::Display for Name<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Name::User(u) => write!(f, "u{}:{}", u.namespace, u.index),
            Name::TestCase(s) => write!(f, "%{s}"),
            Name::LibCall(l) => write!(f, "{l:?}"),
            Name::KnownSymbol(k) => write!(f, "{k:?}"),
        }
    }
}

/// Resolved value of a `symbol_value` or TLS offset for `tls_value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolValue {
    /// Link-time constant address or offset.
    Const(u64),
    /// Address held in a wasm global.
    Global(GlobalIndex),
    /// Data symbol in the object's symbol table (for
    /// [`ModuleLayout::relocatable`]).
    Symbol(SymbolId),
}

/// Resolves external references from a function to entities in the surrounding
/// wasm module.
///
/// An error returned by any method fails [`compile_function`] with that error.
/// Only [`Env::func_index`] is required: the other methods default to
/// returning [`Error::Env`], for environments whose functions never need them.
pub trait Env {
    /// Wasm function index (or symbol index when relocatable) of a direct callee.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not name a known function.
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex>;

    /// Wasm type index of a function or block type (`params -> results`).
    ///
    /// # Errors
    ///
    /// Returns an error if the type cannot be added to the module. The default
    /// implementation always returns [`Error::Env`].
    fn type_index(&mut self, params: &[ValType], results: &[ValType]) -> Result<TypeIndex> {
        let _ = (params, results);
        Err(Error::Env("call_indirect requires Env::type_index".into()))
    }

    /// Function table slot for `func_addr`, or function symbol index when
    /// [`ModuleLayout::relocatable`] is set.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not name a function with a table slot.
    /// The default implementation always returns [`Error::Env`].
    fn func_table_slot(&mut self, name: Name<'_>) -> Result<u64> {
        Err(Error::Env(format!(
            "func_addr {name} requires Env::func_table_slot"
        )))
    }

    /// Value of `symbol_value` (or TLS offset for `tls_value`).
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not name a known symbol. The default
    /// implementation always returns [`Error::Env`].
    fn symbol_value(&mut self, name: Name<'_>) -> Result<SymbolValue> {
        Err(Error::Env(format!(
            "symbol_value {name} requires Env::symbol_value"
        )))
    }

    /// Wasm tag index for a `try_call` exception handler. The tag carries two
    /// pointer-sized integers matching the calling convention's payload.
    ///
    /// # Errors
    ///
    /// Returns an error if `tag` has no wasm tag. The default implementation
    /// always returns [`Error::Env`].
    fn exception_tag(&mut self, tag: ir::ExceptionTag) -> Result<TagIndex> {
        Err(Error::Env(format!(
            "exception tag {tag} requires Env::exception_tag"
        )))
    }
}

/// A compiled wasm function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompiledFunction {
    /// Parameter types, as computed by [`wasm_signature`].
    pub params: Vec<ValType>,
    /// Result types, as computed by [`wasm_signature`].
    pub results: Vec<ValType>,
    /// Non-parameter locals, run-length encoded.
    pub locals: Vec<(u32, ValType)>,
    /// Encoded instruction bytes, including the trailing `end`.
    pub body: Vec<u8>,
    /// Code relocations within `body` when [`ModuleLayout::relocatable`] is set.
    pub relocs: Vec<Reloc>,
}

/// A relocation in [`CompiledFunction::body`] (5-byte or 10-byte padded LEB128,
/// or fixed-width immediate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reloc {
    /// Byte offset of the relocatee within [`CompiledFunction::body`].
    pub offset: u32,
    /// How the field is encoded and what it refers to.
    pub kind: RelocKind,
    /// Symbol table index (or type section index for [`RelocKind::TypeIndexLeb`]).
    pub index: u32,
    /// Addend added to the symbol's address (memory-address kinds only).
    pub addend: i64,
}

/// Relocation type codes from the WebAssembly linking conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RelocKind {
    /// A function index, as a 5-byte uleb128 (`R_WASM_FUNCTION_INDEX_LEB`).
    FunctionIndexLeb = 0,
    /// A function's table slot, as a 5-byte sleb128 in `i32.const`
    /// (`R_WASM_TABLE_INDEX_SLEB`).
    TableIndexSleb = 1,
    /// A function's table slot, as a `u32` in data (`R_WASM_TABLE_INDEX_I32`).
    TableIndexI32 = 2,
    /// A memory address, as a 5-byte sleb128 in `i32.const`
    /// (`R_WASM_MEMORY_ADDR_SLEB`).
    MemoryAddrSleb = 4,
    /// A memory address, as a `u32` in data (`R_WASM_MEMORY_ADDR_I32`).
    MemoryAddrI32 = 5,
    /// A type index, as a 5-byte uleb128 (`R_WASM_TYPE_INDEX_LEB`).
    TypeIndexLeb = 6,
    /// A global index, as a 5-byte uleb128 (`R_WASM_GLOBAL_INDEX_LEB`).
    GlobalIndexLeb = 7,
    /// A tag index, as a 5-byte uleb128 (`R_WASM_TAG_INDEX_LEB`).
    TagIndexLeb = 10,
    /// A memory address, as a 10-byte sleb128 in `i64.const`
    /// (`R_WASM_MEMORY_ADDR_SLEB64`).
    MemoryAddrSleb64 = 15,
    /// A memory address, as a `u64` in data (`R_WASM_MEMORY_ADDR_I64`).
    MemoryAddrI64 = 16,
    /// A function's table slot, as a 10-byte sleb128 in `i64.const`
    /// (`R_WASM_TABLE_INDEX_SLEB64`).
    TableIndexSleb64 = 18,
    /// A function's table slot, as a `u64` in data (`R_WASM_TABLE_INDEX_I64`).
    TableIndexI64 = 19,
    /// A table index, as a 5-byte uleb128 (`R_WASM_TABLE_NUMBER_LEB`).
    TableNumberLeb = 20,
    /// An offset from `__tls_base`, as a 5-byte sleb128 in `i32.const`
    /// (`R_WASM_MEMORY_ADDR_TLS_SLEB`).
    MemoryAddrTlsSleb = 21,
    /// An offset from `__tls_base`, as a 10-byte sleb128 in `i64.const`
    /// (`R_WASM_MEMORY_ADDR_TLS_SLEB64`).
    MemoryAddrTlsSleb64 = 25,
}

impl RelocKind {
    /// Whether this relocation kind carries an addend in the binary format.
    pub fn has_addend(self) -> bool {
        use RelocKind::*;
        matches!(
            self,
            MemoryAddrSleb
                | MemoryAddrI32
                | MemoryAddrSleb64
                | MemoryAddrI64
                | MemoryAddrTlsSleb
                | MemoryAddrTlsSleb64
        )
    }
}

impl CompiledFunction {
    /// Convert into a `wasm_encoder::Function` for a code section.
    pub fn to_wasm_function(&self) -> wasm_encoder::Function {
        let mut f = wasm_encoder::Function::new(self.locals.iter().copied());
        f.raw(self.body.iter().copied());
        f
    }
}

/// Map a Cranelift type to its representing wasm value type.
///
/// `i8`, `i16`, and `f16` live zero-extended in `i32`. Vectors up to 128 bits
/// map to `v128` (narrower vectors occupy the low bytes).
///
/// # Errors
///
/// Returns [`Error::Unsupported`] for types with no single wasm value type,
/// such as `i128` (which [`wasm_signature`] splits in two) or vectors wider
/// than 128 bits.
pub fn wasm_type(ty: ir::Type) -> Result<ValType> {
    use cranelift_codegen::ir::types;
    Ok(match fixed_type(ty) {
        types::I8 | types::I16 | types::I32 | types::F16 => ValType::I32,
        types::I64 => ValType::I64,
        types::F32 => ValType::F32,
        types::F64 => ValType::F64,
        ty if ty.is_vector()
            && (16..=128).contains(&ty.bits())
            && matches!(
                ty.lane_type(),
                types::I8
                    | types::I16
                    | types::I32
                    | types::I64
                    | types::F16
                    | types::F32
                    | types::F64
            ) =>
        {
            ValType::V128
        }
        _ => return unsupported(format!("type {ty}")),
    })
}

/// Map a dynamic vector type (e.g. `i32x4*gv0`) to its fixed-size base vector
/// (scale 1), or return `ty` unchanged.
pub(crate) fn fixed_type(ty: ir::Type) -> ir::Type {
    if ty.is_dynamic_vector() {
        ty.dynamic_to_vector().unwrap_or(ty)
    } else {
        ty
    }
}

/// Map a Cranelift signature to wasm parameter and result types, splitting each
/// `i128` or `f128` into two `i64`s (low half first) and applying
/// [`ModuleLayout::demote_multi_results`] when enabled in `layout`.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] if a parameter or return type has no wasm
/// representation (see [`wasm_type`]).
pub fn wasm_signature(
    sig: &Signature,
    layout: &ModuleLayout,
) -> Result<(Vec<ValType>, Vec<ValType>)> {
    let lower = |ps: &[ir::AbiParam]| -> Result<Vec<ValType>> {
        let mut out = Vec::with_capacity(ps.len());
        for p in ps {
            if matches!(p.value_type, ir::types::I128 | ir::types::F128) {
                out.extend([ValType::I64, ValType::I64]);
            } else {
                out.push(wasm_type(p.value_type)?);
            }
        }
        Ok(out)
    };
    let (mut params, mut results) = (lower(&sig.params)?, lower(&sig.returns)?);
    if layout.demote_multi_results && results.len() > 1 {
        params.insert(0, layout.ptr_ty());
        results.clear();
    }
    Ok((params, results))
}

/// Compile one function.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] if the function uses a feature this backend
/// does not support, or one that `config` does not provide for (such as a
/// stack slot without [`ModuleLayout::stack_pointer_global`]). Returns any
/// error from `env` unchanged.
pub fn compile_function(
    func: &ir::Function,
    config: &Config,
    env: &mut dyn Env,
) -> Result<CompiledFunction> {
    let func = legalize_i128::prepare(func, &config.module)?;
    let info = analysis::FuncInfo::compute(&func)?;
    let trees = expr_trees::build(&func, &info, config.opts.stack_values);
    let locals = locals::allocate(&func, &info, &trees, config.opts.coalesce)?;
    structure::emit_function(&func, config, env, &info, &trees, locals)
}
