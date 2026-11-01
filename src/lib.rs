//! A WebAssembly backend for Cranelift.
//!
//! Compiles a Cranelift [`ir::Function`](cranelift_codegen::ir::Function)
//! into a WebAssembly function body, so anything that produces Cranelift IR
//! (CLIF) can target wasm. This may be useful for:
//!
//! - **JITs running inside wasm**, which emit wasm for the host engine to
//!   compile and link into the running module, or
//! - **Ahead-of-time compilers built on Cranelift**, such as
//!   `rustc_codegen_cranelift`, which emit relocatable wasm objects for a
//!   linker.
//!
//! Like Cranelift itself, this crate prioritizes compile speed over the size or
//! performance of generated code. Users may consider incorporating `wasm-opt`
//! into their toolchain if they need faster or smaller generated code.
//!
//! # Usage
//!
//! [`compile_function`] compiles one function. Its output depends on two
//! inputs:
//!
//!   1. a [`Config`] describing:
//!      - the target module's layout,
//!      - the engine features available, and
//!      - which optimizations to run; and
//!   2. an [`Env`] mapping names in the function (callees, symbols, exception tags) to
//!      wasm indices.
//!
//! The result is a [`CompiledFunction`] that can be placed in a module in several ways:
//!
//! - [`ModuleBuilder`] assembles a module that is either self-contained or
//!   imports its memory and function table from a host program.
//! - [`CompiledFunction::to_wasm_function`] produces a
//!   [`wasm_encoder::Function`] for a module you assemble yourself.
//! - [`object::ObjectBuilder`] writes a relocatable object for a linker, such as `wasm-ld`
//!   or `wild` (with [`ModuleLayout::relocatable`] enabled).
//! - With the `module` feature, `object_module::WasmObjectModule` implements
//!   `cranelift_module::Module` on top of `ObjectBuilder`.
//!
//! Unsupported operations return [`Error::Unsupported`].
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
//! )?;
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
//! # wasmparser::validate(&wasm)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
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

// Compile-test the README's examples.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
