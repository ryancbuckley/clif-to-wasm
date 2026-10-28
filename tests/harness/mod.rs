#![allow(dead_code)]
//! Test harness for compiling CLIF functions and `.clif` filetests (`; run:`
//! directives) to WebAssembly and executing them with Wasmtime or
//! `cranelift-interpreter`.

pub mod clifgen;
pub mod kernels;
mod runner;

pub use runner::*;
