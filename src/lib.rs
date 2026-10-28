//! A WebAssembly backend for Cranelift.
//!
//! Translates a Cranelift [`ir::Function`](cranelift_codegen::ir::Function)
//! into a WebAssembly function body.
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
#![no_std]

extern crate alloc;

// Re-exported because their types appear in this crate's API, so users can
// name them without depending on matching versions.
pub use cranelift_codegen;
#[cfg(feature = "module")]
pub use cranelift_module;
pub use wasm_encoder;

mod analysis;
mod compile;
mod config;
mod emit;
mod env;
mod error;
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
mod wasm_types;

pub use compile::{CompiledFunction, Reloc, RelocKind, compile_function};
pub use config::{AddressType, Config, EngineFeatures, ModuleLayout, Optimizations};
pub use env::{Env, Name, SymbolValue};
pub use error::{Error, Result};
pub use module::{
    Export, FuncIndex, GlobalIndex, MemoryIndex, ModuleBuilder, TableIndex, TagIndex, TypeIndex,
};
pub use object::SymbolId;
pub use wasm_types::{wasm_signature, wasm_type};
