# clif-to-wasm

A WebAssembly backend for [Cranelift](https://cranelift.dev).

Compiles a Cranelift IR (CLIF) [`Function`] into a WebAssembly (wasm) function body,
so anything that produces CLIF can target wasm. This may be useful for building:

- a JIT running inside wasm that emits wasm for the host engine to compile
  and link into the running module, or
- an ahead-of-time compiler built on Cranelift that emits relocatable wasm objects for a linker.
   - The [in-browser Rust playground](https://ryancbuckley.github.io/rust-playground-inbrowser-site) is a demo of such a compiler, compiling and running Rust code fully in the browser.

Like Cranelift itself, this crate prioritizes compile speed over the size or
performance of generated code. Users may consider incorporating `wasm-opt` into
their toolchain if they need faster or smaller generated code.

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
or `exception_tag` respectively. To assemble the module yourself,
`CompiledFunction::to_wasm_function` returns a `wasm_encoder::Function`.

`ModuleBuilder` (`src/module.rs`) assembles a complete wasm module whose
indices match your `Config` and `Env`:
- **Self-contained modules** define their own memory, globals, and a `funcref`
  table where slot `i` holds function `i` (used by the test suite).
- **Dynamically linked modules** (e.g., for an in-wasm JIT) import the host's
  memory, table, and stack-pointer global, and use `place_in_table` to populate
  free table slots. See `tests/module.rs`.

`Config` has three groups of settings:

| Group | Fields | Meaning |
|---|---|---|
| `module: ModuleLayout` | `address_type`, `memory_index`, `table_index`, `stack_pointer_global`, `tls_base_global`, `pinned_reg_global`, `atomic_lock_table_addr`, `relocatable`, `demote_multi_results` | Target module layout; must match the embedder's module definition. |
| `engine: EngineFeatures` | `tail_calls` | Engine feature support. When false, `return_call` lowers to `call` + `return`. |
| `opts: Optimizations` | `stack_values`, `coalesce`, `use_tee` | Codegen optimizations (all enabled by default; `Optimizations::none()` disables all). |

For example, `config.module.stack_pointer_global = Some(sp)` enables stack
slots, `config.module.table_index = Some(t)` enables `call_indirect`,
`config.module.atomic_lock_table_addr = Some(addr)` enables `i128` atomics,
and `config.opts.coalesce = false` disables move coalescing.

### Relocatable objects and `cranelift-module`

When `ModuleLayout::relocatable` is enabled, the backend emits code for a
relocatable wasm object following the [WebAssembly linking conventions] used by
`wasm-ld` and `wild`: `Env` returns symbol indices, and function, global, table,
type, data-address, table-slot, and TLS-offset immediates are encoded as
5-byte padded LEB128s recorded in `CompiledFunction::relocs`.
`object::ObjectBuilder` (`src/object.rs`) collects symbols and definitions and
writes the object file—including `linking`, `reloc.CODE`, `reloc.DATA`, and
`target_features` sections—while importing `__linear_memory`,
`__stack_pointer`, `__indirect_function_table`, and `__tls_base` from `env` to
match LLVM.

The `module` Cargo feature adds `object_module::WasmObjectModule`, a
`cranelift_module::Module` implementation that compiles functions into a
relocatable wasm object (used by `rustc_codegen_cranelift` to compile Rust to
wasm). Because Cranelift has no wasm `TargetIsa` and `TargetIsa` is sealed to
`cranelift-codegen`, `object_module::frontend_isa` supplies a Pulley ISA of the
matching pointer width for frontend queries.

[WebAssembly linking conventions]: https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md

### Calling convention

Calls follow LLVM's Basic C ABI for WebAssembly so output can link with
LLVM-compiled code (such as Rust's standard library):

- Narrow (`i8`/`i16`) arguments and returns marked `sext` are sign-extended by
  the caller (arguments) or callee (returns); `uext` values are already
  zero-extended. Narrow call results without `uext` are masked because callees
  may leave garbage in high bits.
- By-value struct arguments (`ArgumentPurpose::StructArgument`) are copied into
  the caller's stack frame and passed by pointer.
- With `ModuleLayout::demote_multi_results`, functions returning multiple wasm
  values (after splitting `i128`/`f128`) instead take a hidden first pointer to
  a C-layout return area and return nothing (`src/legalize_abi.rs`). Otherwise,
  wasm multi-value returns are used.

## Unsupported features (return `Error::Unsupported`):
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

## Implementation details

This is not a Cranelift `TargetIsa`. Cranelift's machine backends center on
register allocation, which a stack machine with unlimited locals does not need.
This backend operates directly on Cranelift IR.

### Pipeline

| Pass | File | What it does | Switch |
|------|------|--------------|--------|
| Prepare & Legalize | `src/legalize_i128.rs`, `src/legalize_abi.rs` | Scans once for value aliases, `i128`/`f128`, and multi-result ABI demotion; when needed, clones the function, resolves aliases, splits 128-bit scalars into `(lo, hi)` `i64` pairs, and rewrites multi-result signatures to use a return-area pointer | `demote_multi_results` |
| Analysis | `src/analysis.rs` | Computes RPO, deduplicated CFG edges, dominators, loop headers, merge nodes, reducibility, and single-pass backward liveness and use counts | |
| Expression trees | `src/expr_trees.rs` | Folds single-use defs into their user within a block when side-effect order is preserved (checked in O(1) via effect colors). Rematerializes constants at each use. `operands()` defines the evaluation order for all later passes | `stack_values` |
| Locals | `src/locals.rs` | Builds per-block live-range segments for non-inlined values, collects block-argument moves, and greedily colors locals per wasm type. Move partners (argument ↔ parameter) try each other's local first to eliminate most moves | `coalesce` |
| Structure | `src/structure.rs` | Runs Ramsey's "Beyond Relooper" (ICFP 2022) over the dominator tree using an explicit task stack. Irreducible CFGs fall back to a `loop` + `br_table` dispatcher | |
| Emit | `src/emit.rs` | Emits the prologue/epilogue (shadow stack and narrow parameter masking), values, scoped LIFO scratch locals, and instructions with a `local.set x; local.get x` → `local.tee x` peephole | `use_tee` |
| Lower | `src/lower.rs`, `src/lower_simd.rs`, `src/lower_atomic.rs`, `src/lower_float.rs` | Lowers each opcode to wasm. Vector ops, atomics, and emulated floats (`f16`, `fma`, and `FloatCC`) live in dedicated modules | |

Cross-module contracts are each defined in one place:

- **Evaluation order:** `expr_trees::operands` defines it; lowering pulls
  operands in order via `Emitter::operand` (branches are the one documented
  exception).
- **Skipping block-argument moves:** `Locals::move_needed` relies on the
  live-range conventions documented in `locals.rs`.

Tree-folding safety: `color[i]` counts effectful instructions before `i` in its
block. An effectful def `d` may fold into the tree rooted at `r` iff
`color[r] - color[d] - 1` equals the number of effectful defs already folded
into that tree. Multi-target branches only fold pure trees into block arguments
so unconditional effects never become conditional.

### CLIF -> Wasm Example

Given this CLIF function (`cargo run --example clif2wat -- file.clif`):

```text
function %dot_step(i32, i32, i32) -> i32 {
block0(v0: i32, v1: i32, v2: i32):
    v3 = iconst.i32 0
    jump block1(v3, v2)                    ; acc = 0, n = v2

block1(v4: i32, v5: i32):                  ; loop header: acc, n
    brif v5, block2, block4(v4)

block2:                                    ; loop body
    v6 = load.i32 aligned v0
    v7 = load.i32 aligned v1
    v8 = imul v6, v7
    v9 = iconst.i32 0
    v10 = icmp slt v8, v9
    brif v10, block3, block5(v8)

block3:
    v11 = ineg v8
    jump block5(v11)

block5(v12: i32):                          ; |v6 * v7|
    v13 = iadd v4, v12
    v14 = iconst.i32 1
    v15 = isub v5, v14
    jump block1(v13, v15)

block4(v16: i32):
    return v16
}
```

the backend emits:

```wasm
(func (param i32 i32 i32) (result i32)
  (local i32 i32 i32)
  i32.const 0              ;; jump block1(v3, v2): push all args,
  local.get 2              ;;   then set parameters in reverse.
  local.set 3              ;;   n  := v2      (local 3)
  local.set 2              ;;   acc := 0      (local 2, reusing dead param v2)
  loop                     ;; block1 is a loop header
    local.get 3
    if                     ;; brif v5
      block                ;; block5 is a merge node: a `block` ending before it
        local.get 0
        i32.load           ;; v6, v7, v8 fold into one expression tree
        local.get 1
        i32.load
        i32.mul
        local.tee 4        ;; v8 has multiple uses, so it gets a local
        i32.const 0        ;; constant v9 is rematerialized inline
        i32.lt_s
        if                 ;; brif v10
          local.get 4      ;; block3 has one predecessor: emitted inline
          i32.const -1
          i32.mul          ;; ineg
          local.set 5      ;; v12 := v11
          br 1             ;; branch to end of `block` = block5
        else
          local.get 4
          local.set 5      ;; v12 := v8
          br 1
        end
      end
      local.get 2          ;; block5: v13 and v15 fold into the jump's
      local.get 5          ;; arguments, needing no locals of their own
      i32.add
      local.get 3
      i32.const 1
      i32.sub
      local.set 3          ;; n := v15 (coalesced in place)
      local.set 2          ;; acc := v13
      br 1                 ;; back edge to loop
    else
      local.get 2          ;; block4 inline
      local.tee 4
      return
    end
  end
  unreachable
)
```

## Testing

```sh
cargo test --release                     # CLIF runtests + differential fuzzing
cargo test --release --features module   # also tests relocatable objects with rust-lld
FUZZ_ITERS=30000 cargo test --release --test fuzz -- --nocapture

# Cranelift's upstream runtests (wasmtime v50.0.1 matches Cranelift 0.137.1):
git clone --depth 1 --branch v50.0.1 --filter=blob:none --sparse \
    https://github.com/bytecodealliance/wasmtime
(cd wasmtime && git sparse-checkout set cranelift/filetests/filetests/runtests)
cargo run --release --example upstream -- wasmtime/cranelift/filetests/filetests/runtests

# Inspect output:
cargo run --example clif2wat -- tests/clif/control.clif fib
cargo run --example runclif  -- tests/clif/control.clif
```

- `tests/clif/*.clif`: Cranelift filetests with `; run:` directives, compiled,
  validated with `wasmparser`, and executed in wasmtime across five configs
  (default, no-tee, no-coalesce, memory64, and `demote_multi_results`).
- `tests/fuzz.rs`: generates random functions with `cranelift-frontend`
  (scalars, SIMD, and `i128`) and compares execution in `cranelift-interpreter`
  against wasmtime. Reproduce or shrink a failure with `FUZZ_SEED=<n>`,
  `FUZZ_BUDGET=<n>`, and `FUZZ_ONLY=scalar|simd|i128`.
- `tests/cfg_fuzz.rs`: proptest suite that fuzzes `structure.rs` and `locals.rs`
  on arbitrary reducible and irreducible CFGs with multi-parameter block
  arguments, swaps, and unreachable blocks.
- `tests/simd_ops.rs` and `tests/narrow_vectors.rs`: differential tests of all
  vector operations across 128-bit and narrow vector types against the
  interpreter, including with non-zero junk in unused upper bytes.
- `tests/float.rs`: tests emulated `f16` conversions (exhaustively across all
  65,536 bit patterns and rounding midpoints), `f16` arithmetic, and `f16`/`f32`/`f64`
  `fma` against exact software references.
- `tests/module.rs`: links a JIT module into a host module, sharing memory,
  table, and stack pointer, and tests cross-module calls in both directions.
- `tests/object.rs` (`--features module`): emits relocatable objects via
  `WasmObjectModule`, links them with `rust-lld -flavor wasm`, and runs the
  linked binary in wasmtime.
- Upstream runtests: 14,043 run commands pass and 0 fail across 396 of 398
  files. The two unsupported files are `throw.clif` (uses `get_return_address`
  for native unwinding) and `i128-conversion.clif` (an interpreter-only test
  expecting saturating float-to-`i128` behaviour where Cranelift specifies trap
  on overflow).

### Differential testing notes

- `cranelift-interpreter` traps on `INT_MIN srem -1`, whereas Cranelift IR
  defines it as `0`; the fuzzer avoids that input.
- Wasm NaN payloads are nondeterministic and the interpreter preserves signaling
  NaNs, so test comparisons treat all NaNs as equal (per lane for vectors) and
  the fuzzer canonicalizes NaNs before bit-level operations.
- The interpreter panics on `avg_round` and some 16/32-bit narrow vector ops,
  and saturates trapping float-to-`i128` conversions, so differential tests skip
  those cases (`avg_round` is tested by upstream runtests).
- `iconst.i128` is rejected by Cranelift's verifier (tests use `iconcat`),
  though the backend still supports it by zero-extending the immediate.

## Possible next steps

- Remaining unsupported operations (`i128` widening multiplies, `f128` software
  arithmetic).
- Operand commutation and pre-passes to expose more single-use expression trees.
- Compile-speed tuning: local allocation, analysis, and emission each take ~1/3
  of compile time; caching `operands()`/`value_def` lookups across passes is the
  main remaining lever.

## Additional Disclaimer
This project is not affiliated with the ByteCode Alliance, wasmtime, Cranelift, my current employer, or any of my previous employers.
