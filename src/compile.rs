//! The compilation entry point and its output.

use crate::config::Config;
use crate::env::Env;
use crate::error::Result;
use crate::{analysis, expr_trees, legalize_i128, locals, structure};
use alloc::vec::Vec;
use cranelift_codegen::ir::Function;
use wasm_encoder::ValType;

/// A compiled wasm function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CompiledFunction {
    /// Parameter types, as computed by [`wasm_signature`](crate::wasm_signature).
    pub params: Vec<ValType>,
    /// Result types, as computed by [`wasm_signature`](crate::wasm_signature).
    pub results: Vec<ValType>,
    /// Non-parameter locals, run-length encoded.
    pub locals: Vec<(u32, ValType)>,
    /// Encoded instruction bytes, including the trailing `end`.
    pub body: Vec<u8>,
    /// Code relocations within `body` when
    /// [`ModuleLayout::relocatable`](crate::ModuleLayout::relocatable) is set.
    pub relocs: Vec<Reloc>,
}

impl CompiledFunction {
    /// A function with the given signature, locals, and encoded `body`
    /// (including the trailing `end`), and no relocations.
    pub fn new(
        params: Vec<ValType>,
        results: Vec<ValType>,
        locals: Vec<(u32, ValType)>,
        body: Vec<u8>,
    ) -> Self {
        CompiledFunction {
            params,
            results,
            locals,
            body,
            relocs: Vec::new(),
        }
    }

    /// Convert into a `wasm_encoder::Function` for a code section.
    pub fn to_wasm_function(&self) -> wasm_encoder::Function {
        let mut f = wasm_encoder::Function::new(self.locals.iter().copied());
        f.raw(self.body.iter().copied());
        f
    }
}

/// A relocation in [`CompiledFunction::body`] (5-byte or 10-byte padded LEB128,
/// or fixed-width immediate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

/// Relocation type codes from the [WebAssembly linking conventions](https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md#relocation-sections).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
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

/// Compile one function.
///
/// # Errors
///
/// Returns [`Error::Unsupported`](crate::Error::Unsupported) if the function
/// uses a feature this backend does not support, or one that `config` does not
/// provide for (such as a stack slot without
/// [`ModuleLayout::stack_pointer_global`](crate::ModuleLayout::stack_pointer_global)).
/// Returns any error from `env` unchanged.
///
/// # Panics
///
/// `func` must pass Cranelift's IR verifier
/// ([`cranelift_codegen::verify_function`]), which this function does not
/// run. Invalid IR may cause a panic, either in this crate or in the
/// `cranelift-codegen` code it calls. On valid IR, a panic is a bug in this
/// crate: features it lacks return
/// [`Error::Unsupported`](crate::Error::Unsupported) instead.
pub fn compile_function(
    func: &Function,
    config: &Config,
    env: &mut dyn Env,
) -> Result<CompiledFunction> {
    let func = legalize_i128::prepare(func, &config.module)?;
    let info = analysis::FuncInfo::compute(&func)?;
    let trees = expr_trees::build(&func, &info, config.opts.stack_values);
    let locals = locals::allocate(&func, &info, &trees, config.opts.coalesce)?;
    structure::emit_function(&func, config, env, &info, &trees, locals)
}
