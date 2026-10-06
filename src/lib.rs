//! A WebAssembly backend for Cranelift.
//!
//! This crate translates a Cranelift [`ir::Function`] into the body of a
//! WebAssembly function. It is not a `TargetIsa`: Cranelift's machine backends
//! are built around register allocation, which a stack machine with unlimited
//! locals does not need. Instead the pipeline is:
//!
//! 0. **Legalization** ([`legalize_i128`]): `i128` values become pairs of
//!    `i64`s, so the later passes only see types wasm has.
//! 1. **Analysis** ([`analysis`]): reachability, reverse postorder, CFG edges,
//!    dominators, loop headers and merge nodes, value-level dead code
//!    elimination and use counts.
//! 2. **Stackification** ([`stackify`]): greedy, backwards, per-block expression
//!    tree building. A single-use def is folded into its user when doing so
//!    does not reorder side effects; this is checked in O(1) with Cranelift-style
//!    side-effect *colors*. Constants are rematerialized at every use.
//! 3. **Local allocation** ([`liveness`], [`locals`]): values that are not
//!    consumed directly off the operand stack get a wasm local; locals are
//!    shared between values whose live ranges do not overlap, preferring to
//!    give block arguments and parameters the same local.
//! 4. **Structuring and emission** ([`structure`], [`lower`]): control flow is
//!    rebuilt from the dominator tree following Ramsey's "Beyond Relooper"
//!    (ICFP 2022). Irreducible control flow falls back to a dispatch loop.
//!
//! Anything more aggressive (cross-block local sinking, value-carrying blocks,
//! local reordering for LEB size) is left to `wasm-opt`.

use cranelift_codegen::ir::{self, ExternalName, LibCall, Signature, UserExternalName};
use wasm_encoder::ValType;

mod analysis;
mod emit;
mod encode;
mod legalize_i128;
mod liveness;
mod locals;
mod lower;
mod lower_simd;
pub mod module;
mod stackify;
mod structure;

pub use module::ModuleBuilder;

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

/// Code generation settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Whether the target memory is a 64-bit (memory64) memory. This decides
    /// the wasm type of addresses and of the shadow stack pointer.
    pub memory64: bool,
    /// The memory that loads and stores access.
    pub memory_index: u32,
    /// The table used by `call_indirect`.
    pub table_index: u32,
    /// A mutable global holding the shadow stack pointer. Required by
    /// functions that use stack slots. The stack grows downwards and the
    /// pointer is assumed to be 16-byte aligned.
    pub stack_pointer_global: Option<u32>,
    /// Use the tail-call proposal for `return_call`. Without it, tail calls
    /// are lowered to a call followed by a return.
    pub tail_calls: bool,
    /// Use `local.tee` to feed the first use of a multi-use value directly
    /// from the stack.
    pub use_tee: bool,
    /// Build expression trees. When off, every value goes through a local
    /// (constants are still rematerialized); this is mainly useful as a
    /// baseline for comparisons.
    pub stackify: bool,
    /// Try to give a block argument and the parameter it is passed to the
    /// same local, so that the move disappears.
    pub coalesce: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            memory64: false,
            memory_index: 0,
            table_index: 0,
            stack_pointer_global: None,
            tail_calls: true,
            use_tee: true,
            stackify: true,
            coalesce: true,
        }
    }
}

impl Config {
    pub(crate) fn ptr_ty(&self) -> ValType {
        if self.memory64 {
            ValType::I64
        } else {
            ValType::I32
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

/// The value of a `symbol_value`.
#[derive(Debug, Clone, Copy)]
pub enum SymbolValue {
    /// A link-time constant address.
    Const(u64),
    /// An address held in a wasm global.
    Global(u32),
}

/// The module environment: resolves references from a function to entities
/// in the surrounding wasm module.
pub trait Env {
    /// The wasm function index of a direct callee.
    fn func_index(&mut self, name: Name<'_>) -> Result<u32>;

    /// The wasm type index of a signature, for `call_indirect`.
    fn type_index(&mut self, sig: &Signature) -> Result<u32> {
        let _ = sig;
        Err(Error::Env("call_indirect requires Env::type_index".into()))
    }

    /// The value of `func_addr`: the function's slot in the indirect call
    /// table.
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
/// `i8` and `i16` values live in `i32`s and are kept zero-extended. 128-bit
/// vectors are `v128`s.
pub fn wasm_type(ty: ir::Type) -> Result<ValType> {
    use cranelift_codegen::ir::types;
    Ok(match ty {
        types::I8 | types::I16 | types::I32 => ValType::I32,
        types::I64 => ValType::I64,
        types::F32 => ValType::F32,
        types::F64 => ValType::F64,
        ty if ty.is_vector()
            && ty.bits() == 128
            && matches!(
                ty.lane_type(),
                types::I8 | types::I16 | types::I32 | types::I64 | types::F32 | types::F64
            ) =>
        {
            ValType::V128
        }
        _ => return unsupported(format!("type {ty}")),
    })
}

/// The wasm signature of a Cranelift signature.
///
/// An `i128` parameter or return value becomes two `i64`s, low half first.
pub fn wasm_signature(sig: &Signature) -> Result<(Vec<ValType>, Vec<ValType>)> {
    let lower = |ps: &[ir::AbiParam]| -> Result<Vec<ValType>> {
        let mut out = Vec::with_capacity(ps.len());
        for p in ps {
            if p.value_type == ir::types::I128 {
                out.extend([ValType::I64, ValType::I64]);
            } else {
                out.push(wasm_type(p.value_type)?);
            }
        }
        Ok(out)
    };
    Ok((lower(&sig.params)?, lower(&sig.returns)?))
}

/// Compile one function.
pub fn compile_function(
    func: &ir::Function,
    config: &Config,
    env: &mut dyn Env,
) -> Result<CompiledFunction> {
    // The passes assume alias-free IR without `i128`. Both fixes need a
    // mutable copy, so only make one when needed.
    let rewritten;
    let (aliases, wide) = scan(func);
    let func = if aliases || wide {
        let mut f = func.clone();
        f.dfg.resolve_all_aliases();
        if wide {
            legalize_i128::legalize(&mut f)?;
        }
        rewritten = f;
        &rewritten
    } else {
        func
    };
    let info = analysis::FuncInfo::compute(func)?;
    let stack = stackify::stackify(func, &info, config.stackify);
    let ranges = liveness::compute(func, &info, &stack);
    let locals = locals::allocate(func, &info, &ranges, config.coalesce)?;
    structure::emit_function(func, config, env, &info, &stack, locals)
}

/// One pass over the function: does it use value aliases, and does it
/// involve `i128` anywhere (signatures, block parameters or results)?
fn scan(func: &ir::Function) -> (bool, bool) {
    let dfg = &func.dfg;
    let mut aliases = false;
    let mut wide = legalize_i128::signatures_need(func);
    for b in func.layout.blocks() {
        wide |= dfg
            .block_params(b)
            .iter()
            .any(|&p| dfg.value_type(p) == ir::types::I128);
        for inst in func.layout.block_insts(b) {
            aliases |= dfg.inst_values(inst).any(|v| dfg.value_is_alias(v));
            wide |= dfg
                .inst_results(inst)
                .iter()
                .any(|&r| dfg.value_type(r) == ir::types::I128);
        }
    }
    (aliases, wide)
}
