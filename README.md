# clif-to-wasm

A WebAssembly backend for [Cranelift](https://cranelift.dev).

It compiles a Cranelift IR (CLIF) [`Function`](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/ir/function/struct.Function.html)
into a WebAssembly (wasm) function body, so anything that produces CLIF can
target wasm. That could be a JIT running inside wasm that emits code for the
host engine to compile and link into the running module, or an ahead-of-time
compiler that emits relocatable wasm objects for a linker. The
[in-browser Rust playground](https://ryancbuckley.github.io/rust-playground-inbrowser-site)
is an example of the second: it compiles and runs Rust entirely in the browser
using `rustc_codegen_cranelift` and this crate.

Like Cranelift itself, this crate favors compile speed over the size and speed
of the code it generates. Run `wasm-opt` on the output if you need faster or
smaller code.

## Usage

Compile a function and assemble it into a module with `ModuleBuilder`:

```rust
use clif_to_wasm::{
    Config, Env, Error, FuncIndex, ModuleBuilder, Name, compile_function, wasm_signature,
};

// Our CLIF code doesn't call anything, so the environment resolves no names.
struct NoCalls;
impl Env for NoCalls {
    fn func_index(&mut self, name: Name<'_>) -> clif_to_wasm::Result<FuncIndex> {
        Err(Error::Env(format!("unknown function {name}")))
    }
}

fn build_module() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Frontends usually build functions with `cranelift-frontend`; this one is
    // parsed from text with `cranelift-reader`.
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

`Env` maps names in the function to wasm indices. Only `func_index` is
required; functions that use `call_indirect`, `func_addr`, `symbol_value`, or
`try_call` handlers also need `type_index`, `func_table_slot`, `symbol_value`,
or `exception_tag` respectively.

`ModuleBuilder` assembles a module whose indices match your `Config` and `Env`.
It can build a self-contained module with its own memory, globals and function
table, or a module that imports the host's memory, table and stack pointer, as
a JIT inside wasm would (see `tests/module.rs`). To assemble the module
yourself, `CompiledFunction::to_wasm_function` returns a
`wasm_encoder::Function`.

`Config` describes the module the code will live in (`ModuleLayout`: which
memory, table and globals to use, and so on), which wasm features the engine
supports (`EngineFeatures`), and which optimizations to run (`Optimizations`,
all on by default). For example, `config.module.stack_pointer_global = Some(sp)`
enables stack slots and `config.module.table_index = Some(t)` enables
`call_indirect`. The [API docs](https://docs.rs/clif-to-wasm) describe each
setting.

## Relocatable objects

With `ModuleLayout::relocatable` set, the output follows the
[WebAssembly linking conventions] used by `wasm-ld` and `wild`, and
`object::ObjectBuilder` writes it out as a relocatable object file. The `module`
Cargo feature adds `object_module::WasmObjectModule`, a
`cranelift_module::Module` that compiles functions into such an object; this
is what `rustc_codegen_cranelift` uses to compile Rust to wasm.

Calls follow LLVM's Basic C ABI for WebAssembly, so the output can link with
LLVM-compiled code such as Rust's standard library. Set
`ModuleLayout::demote_multi_results` when linking with LLVM code, which returns
multiple values through memory rather than with wasm multi-value returns.

[WebAssembly linking conventions]: https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md

## Unsupported features

These return `Error::Unsupported`:

- `f128` arithmetic and conversions (handled via libcalls in Cranelift).
- `umulhi`/`smulhi` on `i128`.
- Big-endian vector loads, stores, and bitcasts; `f16` vector arithmetic.
- `try_call` with `i128`/`f128` arguments, results, or payloads, or with
  by-value struct arguments; `catch_all` handlers that use the payload.
- Tail calls with by-value struct arguments.
- Dynamic stack slots, stack slots aligned to more than 16 bytes, and
  `get_return_address`.
- In relocatable objects: `try_call`. `WasmObjectModule` also rejects `i128`
  atomics, as it sets no `atomic_lock_table_addr`.

## Testing

```sh
cargo test --release                     # CLIF runtests + differential fuzzing
cargo test --release --features module   # also tests relocatable objects with rust-lld

# Cranelift's upstream runtests (wasmtime v50.0.1 matches Cranelift 0.137.1):
git clone --depth 1 --branch v50.0.1 --filter=blob:none --sparse \
    https://github.com/bytecodealliance/wasmtime
(cd wasmtime && git sparse-checkout set cranelift/filetests/filetests/runtests)
cargo run --release --example upstream -- wasmtime/cranelift/filetests/filetests/runtests

# See what a function compiles to:
cargo run --example clif2wat -- tests/clif/control.clif fib
```

[ARCHITECTURE.md](ARCHITECTURE.md) describes how the backend works, with an
annotated example, and what each test covers.

## Disclaimer

This project is not affiliated with the Bytecode Alliance, wasmtime, Cranelift, my current employer, or any of my previous employers.
