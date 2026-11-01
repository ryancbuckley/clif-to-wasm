# clif-to-wasm

A WebAssembly backend for [Cranelift](https://cranelift.dev).

This crate compiles a Cranelift IR (CLIF)
[`Function`](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/ir/function/struct.Function.html)
into a WebAssembly (wasm) function body, so any language that can be compiled to
CLIF can run on wasm.

One use case is ahead-of-time compilation to emit relocatable `.o` wasm object
files and link them with `wasm-ld` or `wild`. For example, the
[in-browser Rust playground](https://ryancbuckley.github.io/rust-playground-inbrowser-site)
uses `rustc_codegen_cranelift` with this crate to compile and run Rust code in
the browser.

Another use case is JIT compilation inside WebAssembly: a program running in a
wasm engine can compile CLIF to a wasm module at runtime and ask the host engine
to instantiate it.

Like Cranelift itself, this crate favors fast compilation over optimal output
performance. If you need smaller or faster wasm, you might consider running
`wasm-opt` on the output.

## Usage

Compile a CLIF function and package it into a runnable wasm module with
`ModuleBuilder`:

```rust
use clif_to_wasm::{
    Config, Env, Error, FuncIndex, ModuleBuilder, Name, compile_function, wasm_signature,
};

// Our CLIF code doesn't call any other functions, so its environment resolves no names.
struct NoCalls;
impl Env for NoCalls {
    fn func_index(&mut self, name: Name<'_>) -> clif_to_wasm::Result<FuncIndex> {
        Err(Error::Env(format!("unknown function {name}")))
    }
}

fn build_module() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Compilers usually build IR with `cranelift-frontend`; here we parse text.
    let funcs = cranelift_reader::parse_functions(
        "function %add(i32, i32) -> i32 {
         block0(v0: i32, v1: i32):
             v2 = iadd v0, v1
             return v2
         }",
    )?;

    let config = Config::default();
    let mut builder = ModuleBuilder::new();
    let (params, results) = wasm_signature(&funcs[0].signature, &config.module)?;
    let add = builder.declare_function(&params, &results);
    let compiled = compile_function(&funcs[0], &config, &mut NoCalls)?;
    builder.define_function(add, compiled);
    builder.export("add", add);
    Ok(builder.finish())
}
```

`compile_function` takes two things alongside the CLIF `Function`:

- [`Config`](https://docs.rs/clif-to-wasm/latest/clif_to_wasm/struct.Config.html):
  describes the target module layout (`ModuleLayout`, such as which global holds
  the stack pointer and which table holds function pointers), engine features
  (`EngineFeatures`), and compiler optimizations (`Optimizations`).
- [`Env`](https://docs.rs/clif-to-wasm/latest/clif_to_wasm/trait.Env.html):
  a trait you implement to resolve external references in the function (callee
  functions, global symbols, indirect call signatures, and exception tags) to
  wasm indices. Only `Env::func_index` is required if the function does not use
  indirect calls, global symbols, or exceptions.

Once compiled, a `CompiledFunction` can be turned into wasm in a few ways:

- `ModuleBuilder` builds a standalone wasm module (defining its own memory,
  globals, and function table) or a JIT module that imports them from a host
  instance (see `tests/module.rs`).
- `CompiledFunction::to_wasm_function` returns a `wasm_encoder::Function` if you
  are assembling a module yourself with `wasm-encoder`.
- `object::ObjectBuilder` writes a relocatable `.o` wasm file for a linker.

## Relocatable object files and `cranelift-module`

When `config.module.relocatable` is enabled, `compile_function` emits
relocation records (`CompiledFunction::relocs`) for function calls, globals,
memory addresses, and table indices, following the
[WebAssembly tool conventions](https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md)
expected by `wasm-ld` and `wild`.

Enabling the `module` Cargo feature provides `object_module::WasmObjectModule`,
an implementation of `cranelift_module::Module` built on `ObjectBuilder`.
Existing `cranelift-module` frontends (such as `rustc_codegen_cranelift`) can
plug it in directly to emit wasm object files.

Function calls follow LLVM's Basic C ABI for WebAssembly so output can be linked
with C or Rust code compiled by LLVM. When linking with LLVM-compiled code, set
`config.module.demote_multi_results = true` so functions with multiple return
values pass them through a return pointer in linear memory, matching LLVM's C
ABI.

## Unsupported CLIF features

These return `Error::Unsupported`:

- `f128` arithmetic and conversions (Cranelift frontends lower these to libcalls).
- `umulhi` and `smulhi` on `i128`.
- Big-endian vector loads, stores, and bitcasts; `f16` vector arithmetic.
- `try_call` with `i128`/`f128` arguments, results, or exception payloads, or
  with by-value struct arguments; `catch_all` blocks that read a payload.
- Tail calls (`return_call`, `return_call_indirect`) with by-value struct
  arguments.
- Dynamic stack slots, stack slots aligned to more than 16 bytes, and
  `get_return_address`.
- `try_call` in relocatable object files (`wasm-ld` does not yet support the
  required tag relocations).

## Testing

```sh
cargo test --release                     # CLIF runtests + differential fuzzing
cargo test --release --features module   # also links object files with rust-lld

# Run Cranelift's upstream test suite (wasmtime v50.0.1 matches Cranelift 0.137.1):
git clone --depth 1 --branch v50.0.1 --filter=blob:none --sparse \
    https://github.com/bytecodealliance/wasmtime
(cd wasmtime && git sparse-checkout set cranelift/filetests/filetests/runtests)
cargo run --release --example upstream -- wasmtime/cranelift/filetests/filetests/runtests

# Print the WAT a CLIF file compiles to:
cargo run --example clif2wat -- tests/clif/control.clif fib
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a walkthrough of the compiler passes,
an annotated CLIF-to-WAT example, and details on the test suite.

## Disclaimer

This project is not affiliated with the Bytecode Alliance, Wasmtime, Cranelift,
my current employer, or any of my previous employers.
