//! A WebAssembly backend for Cranelift.
//!
//! This crate translates a Cranelift [`ir::Function`] into the body of a
//! WebAssembly function. It is not a `TargetIsa`: Cranelift's machine backends
//! are built around register allocation, which a stack machine with unlimited
//! locals does not need. Instead the pipeline is:
//!
//! 0. **Preparation & Legalization** ([`legalize_i128`]): value aliases are
//!    resolved and `i128` values become pairs of `i64`s, so the later passes
//!    only see canonical SSA values of types wasm has.
//! 1. **Analysis** ([`analysis`]): reachability, reverse postorder, CFG edges,
//!    dominators, loop headers and merge nodes, value-level dead code
//!    elimination and use counts in a single pass.
//! 2. **Expression trees** ([`expr_trees`]): greedy, backwards, per-block
//!    tree building, so that values pass on the operand stack instead of
//!    through locals. A single-use def is folded into its user when doing so
//!    does not reorder side effects; this is checked in O(1) with
//!    Cranelift-style side-effect *colors*. Constants are rematerialized at
//!    every use.
//! 3. **Local allocation** ([`locals`]): values that are not consumed directly
//!    off the operand stack get a wasm local; locals are shared between values
//!    whose live ranges do not overlap, preferring to give block arguments and
//!    parameters the same local.
//! 4. **Structuring and emission** ([`structure`], [`emit`], [`lower`]):
//!    control flow is rebuilt from the dominator tree following Ramsey's
//!    "Beyond Relooper" (ICFP 2022), while instructions are lowered and
//!    peephole-encoded into the function body. Irreducible control flow falls
//!    back to a dispatch loop.
//!
//! Anything more aggressive (cross-block local sinking, value-carrying blocks,
//! local reordering for LEB size) is left to `wasm-opt`.

use cranelift_codegen::ir::{self, ExternalName, LibCall, Signature, UserExternalName};
use wasm_encoder::ValType;

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
mod structure;

pub use module::{
    Export, FuncIndex, GlobalIndex, MemoryIndex, ModuleBuilder, TableIndex, TagIndex, TypeIndex,
};

/// Errors that can occur during compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
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

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn unsupported<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(msg.into()))
}

/// Code generation settings, in three groups.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Facts about the wasm module the code will be placed in. They must
    /// match how the embedder builds that module.
    pub module: ModuleLayout,
    /// Wasm features the engine that runs the code supports.
    pub engine: EngineFeatures,
    /// Optimizations. They change how good the output is, never what it
    /// does; turning one off is for benchmarking and debugging.
    pub opts: Optimizations,
}

/// Facts about the wasm module that will contain the compiled functions.
#[derive(Debug, Clone, Default)]
pub struct ModuleLayout {
    /// Whether the linear memory uses 64-bit addresses (memory64). This
    /// decides the wasm type of load/store addresses and of the shadow stack
    /// pointer. IR addresses of the other width are converted.
    pub memory64: bool,
    /// Which linear memory loads and stores access. A module can have
    /// several (the multi-memory feature); with one memory this is
    /// `MemoryIndex(0)`.
    pub memory_index: MemoryIndex,
    /// Which table `call_indirect` looks functions up in. Wasm has no raw
    /// function pointers: a function "address" is an index into a table of
    /// function references. With one table this is `Some(TableIndex(0))`. With
    /// `None`, functions that use `call_indirect` are rejected.
    pub table_index: Option<TableIndex>,
    /// A mutable global holding the shadow stack pointer. Stack slots must
    /// live in linear memory, since wasm has no addressable stack; functions
    /// that use them move this pointer down on entry and back on exit. The
    /// stack grows downwards and the pointer is assumed to be 16-byte
    /// aligned. With `None`, functions that need stack slots are rejected.
    pub stack_pointer_global: Option<GlobalIndex>,
    /// A mutable `i64` global standing in for Cranelift's pinned register
    /// (`get_pinned_reg`/`set_pinned_reg`). With `None`, functions that use
    /// it are rejected.
    pub pinned_reg_global: Option<GlobalIndex>,
    /// Base byte address in linear memory of a 256-byte (64 × `i32`) striped
    /// spinlock table used to synchronize `i128` atomic operations. Must be
    /// 4-byte aligned and disjoint from the program's data and shadow stack.
    /// With `None`, functions that use `i128` atomics are rejected.
    pub atomic_lock_table_addr: Option<u64>,
    /// A mutable global holding the base address of the current thread's
    /// thread-local storage block (`__tls_base` in the wasm tool
    /// conventions). `tls_value` adds a symbol's offset in that block to it.
    /// With `None`, functions that use `tls_value` are rejected.
    pub tls_base_global: Option<GlobalIndex>,
    /// Produce code for a relocatable object file (the wasm [tool
    /// conventions] for linking) instead of a finished module.
    ///
    /// The indices that [`Env`] returns and the global and table indices in
    /// this struct then name entries of the object's symbol table (type
    /// indices still name the object's type section), and every use of one is
    /// encoded with a padded LEB and recorded in [`CompiledFunction::relocs`].
    /// [`crate::object`] builds such objects.
    ///
    /// [tool conventions]: https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md
    pub relocatable: bool,
    /// Return more than one wasm value through memory, the way LLVM's wasm
    /// backend does: a function whose results (after splitting `i128` and
    /// `f128` into two `i64`s) are more than one gets a hidden first
    /// parameter pointing to space for them, laid out like a C struct (each
    /// result at the next multiple of its size), and returns nothing. This
    /// is needed to call, or be called by, code compiled by LLVM. When off,
    /// multiple results use multi-value returns.
    pub demote_multi_results: bool,
}

impl ModuleLayout {
    /// The wasm type of addresses.
    pub(crate) fn ptr_ty(&self) -> ValType {
        if self.memory64 {
            ValType::I64
        } else {
            ValType::I32
        }
    }
}

/// Wasm features the generated code may use.
#[derive(Debug, Clone)]
pub struct EngineFeatures {
    /// Tail calls (`return_call`). Without them, a Cranelift tail call
    /// becomes an ordinary call followed by a return, which is correct but
    /// grows the call stack.
    pub tail_calls: bool,
}

impl Default for EngineFeatures {
    fn default() -> Self {
        EngineFeatures { tail_calls: true }
    }
}

/// Optimization switches; all are on by default.
#[derive(Debug, Clone)]
pub struct Optimizations {
    /// Pass single-use values on the operand stack, by building expression
    /// trees (see `expr_trees.rs`). When off, every value goes through a
    /// local (constants are still rematerialized).
    pub stack_values: bool,
    /// Try to give a block argument and the parameter it is passed to the
    /// same local, so that the move disappears.
    pub coalesce: bool,
    /// Fuse `local.set x; local.get x` into `local.tee x`.
    pub use_tee: bool,
}

impl Optimizations {
    /// Every optimization switched off.
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
#[derive(Debug, Clone, Copy)]
pub enum Name<'a> {
    User(&'a UserExternalName),
    TestCase(&'a str),
    LibCall(LibCall),
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

/// The value of a `symbol_value`, or for `tls_value` the symbol's offset
/// from the thread-local storage base.
#[derive(Debug, Clone, Copy)]
pub enum SymbolValue {
    /// A link-time constant address.
    Const(u64),
    /// An address held in a wasm global.
    Global(GlobalIndex),
    /// With [`ModuleLayout::relocatable`]: the address of the data symbol
    /// with this index in the object's symbol table, filled in by the linker.
    Symbol(u32),
}

/// The module environment: resolves references from a function to entities
/// in the surrounding wasm module.
pub trait Env {
    /// The wasm function index of a direct callee.
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex>;

    /// The wasm type index of a signature, for `call_indirect`.
    fn type_index(&mut self, sig: &Signature) -> Result<TypeIndex> {
        let _ = sig;
        Err(Error::Env("call_indirect requires Env::type_index".into()))
    }

    /// The value of `func_addr`: the function's slot in the indirect call
    /// table. With [`ModuleLayout::relocatable`], the index of the
    /// function's symbol instead.
    fn func_table_slot(&mut self, name: Name<'_>) -> Result<u64> {
        Err(Error::Env(format!(
            "func_addr {name} requires Env::func_table_slot"
        )))
    }

    /// The value of `symbol_value`.
    fn symbol_value(&mut self, name: Name<'_>) -> Result<SymbolValue> {
        Err(Error::Env(format!(
            "symbol_value {name} requires Env::symbol_value"
        )))
    }

    /// The wasm tag index of a Cranelift exception tag, for the handlers of
    /// `try_call`. The tag's parameters are the exception payload of the
    /// callee's calling convention: two pointer-sized integers (`i64`, or
    /// `i32` if the handlers' payload values are `i32`). Exceptions are
    /// thrown by the embedder's runtime with wasm `throw`.
    fn exception_tag(&mut self, tag: ir::ExceptionTag) -> Result<TagIndex> {
        Err(Error::Env(format!(
            "exception tag {tag} requires Env::exception_tag"
        )))
    }
}

/// A compiled function: its wasm type, locals and encoded body.
#[derive(Debug, Clone)]
pub struct CompiledFunction {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
    /// Declared locals (excluding parameters), run-length encoded.
    pub locals: Vec<(u32, ValType)>,
    /// The encoded instruction sequence, including the final `end`.
    pub body: Vec<u8>,
    /// With [`ModuleLayout::relocatable`], the relocations in `body`.
    pub relocs: Vec<Reloc>,
}

/// A relocation in a function body, for [`ModuleLayout::relocatable`].
///
/// The field is a padded LEB (5 bytes, or 10 for the 64-bit kinds) holding a
/// placeholder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reloc {
    /// Byte offset of the field in [`CompiledFunction::body`].
    pub offset: u32,
    pub kind: RelocKind,
    /// The symbol index, or for [`RelocKind::TypeIndexLeb`] the type index.
    pub index: u32,
    /// Added to the symbol's address (memory address kinds only).
    pub addend: i64,
}

/// Relocation types of the wasm tool conventions, with their numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RelocKind {
    FunctionIndexLeb = 0,
    TableIndexSleb = 1,
    TableIndexI32 = 2,
    MemoryAddrSleb = 4,
    MemoryAddrI32 = 5,
    TypeIndexLeb = 6,
    GlobalIndexLeb = 7,
    TagIndexLeb = 10,
    MemoryAddrSleb64 = 15,
    MemoryAddrI64 = 16,
    TableIndexSleb64 = 18,
    TableIndexI64 = 19,
    TableNumberLeb = 20,
    MemoryAddrTlsSleb = 21,
    MemoryAddrTlsSleb64 = 25,
}

impl RelocKind {
    /// Whether the relocation has an addend.
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
    /// Build a `wasm_encoder::Function` for the code section.
    pub fn to_wasm_function(&self) -> wasm_encoder::Function {
        let mut f = wasm_encoder::Function::new(self.locals.iter().copied());
        f.raw(self.body.iter().copied());
        f
    }
}

/// Map a Cranelift scalar type to the wasm value type that represents it.
///
/// `i8` and `i16` values live in `i32`s and are kept zero-extended. Vectors
/// are `v128`s; one narrower than 128 bits occupies the low bytes, and the
/// other bytes are unspecified.
pub fn wasm_type(ty: ir::Type) -> Result<ValType> {
    use cranelift_codegen::ir::types;
    Ok(match fixed_type(ty) {
        // An f16 is its bits, like an i16.
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

/// The fixed-size type of a dynamic vector type such as `i32x4*gv0`, or
/// `ty` itself. Dynamic vectors have a target-chosen multiple of a fixed
/// size; here that multiple is 1, so they are plain vectors.
pub(crate) fn fixed_type(ty: ir::Type) -> ir::Type {
    if ty.is_dynamic_vector() {
        ty.dynamic_to_vector().unwrap_or(ty)
    } else {
        ty
    }
}

/// The wasm signature of a Cranelift signature.
///
/// An `i128` or `f128` parameter or return value becomes two `i64`s, low half
/// first.
pub fn wasm_signature(sig: &Signature) -> Result<(Vec<ValType>, Vec<ValType>)> {
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
    Ok((lower(&sig.params)?, lower(&sig.returns)?))
}

/// The wasm signature of a function with IR signature `sig` in a module laid
/// out as `layout`: [`wasm_signature`], after moving multiple results into
/// memory if [`ModuleLayout::demote_multi_results`] says so.
pub fn wasm_signature_in(
    sig: &Signature,
    layout: &ModuleLayout,
) -> Result<(Vec<ValType>, Vec<ValType>)> {
    let (mut params, mut results) = wasm_signature(sig)?;
    if layout.demote_multi_results && results.len() > 1 {
        params.insert(0, layout.ptr_ty());
        results.clear();
    }
    Ok((params, results))
}

/// Compile one function.
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
