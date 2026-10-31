//! Code generation settings.

use crate::module::{GlobalIndex, MemoryIndex, TableIndex};
use wasm_encoder::ValType;

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
    /// The address type of linear memory, 32-bit or 64-bit. This is also the wasm type of the
    /// shadow stack pointer.
    pub address_type: AddressType,
    /// The index of the linear memory accessed by loads and stores. Most modules have one memory, in which case this will be 0.
    pub memory_index: MemoryIndex,
    /// The index of the function table used by `call_indirect` (`Some(TableIndex(0))` in
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
    /// Indices returned by [`Env`](crate::Env) and globals/tables in this
    /// struct then refer to symbol table entries (type indices still refer to
    /// the type section), encoded as padded LEB128s and recorded in
    /// [`CompiledFunction::relocs`](crate::CompiledFunction::relocs). See
    /// [`crate::object`].
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

    /// The size of an address in bytes.
    pub(crate) fn bytes(self) -> u32 {
        match self {
            AddressType::I32 => 4,
            AddressType::I64 => 8,
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
