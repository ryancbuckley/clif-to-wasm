# clif-to-wasm

A WebAssembly backend for [Cranelift](https://cranelift.dev) that compiles a
Cranelift IR `Function` into a wasm function body.

The primary use case is a JIT running inside wasm that emits wasm for the host
to compile. Compile speed matters more than code size: optimizing host tiers
rebuild SSA from locals anyway, and `wasm-opt` can shrink the output when size
matters.

This is not a Cranelift `TargetIsa`. Cranelift's machine backends center on
register allocation, which a stack machine with unlimited locals does not need.
This backend operates directly on Cranelift IR.

## Pipeline

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

### Example

Given this CLIF function (`cargo run --example clif2wat -- file.clif`):

```text
function %dot_step(i32, i32, i32) -> i32 {
block0(v0: i32, v1: i32, v2: i32):
    v3 = iconst.i32 0
    jump block1(v3, v2)                    ; acc = 0, n = v2

block1(v4: i32, v5: i32):                  ; loop header: acc, n
    brif v5, block2, block4(v4)

block2:                                    ; loop body
    v6 = load.i32 v0
    v7 = load.i32 v1
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

### Representation

- `i8` and `i16` live in `i32` locals, kept zero-extended. Operations that can
  dirty upper bits mask their result; signed operations sign-extend inputs.
- `f16` lives in an `i32` local as zero-extended raw bits.
- Vectors narrower than 128 bits occupy the low bytes of a `v128`; upper bytes
  are unspecified. Dynamic vectors are treated as fixed-size (scale 1).
- Big-endian loads and stores byte-swap, as wasm memory is little-endian.
- Values crossing block boundaries live in locals. Every `block`, `loop`, and
  `if` has block type `[] -> []`.
- Block-argument moves push all arguments before setting any parameter, so
  parallel swaps need no temporaries.
- Stack slots use a shadow stack in linear memory via
  `ModuleLayout::stack_pointer_global`.
- Traps emit `unreachable`; trap codes are dropped.

## Usage

```rust
use clif_to_wasm::{compile_function, Config, Env, FuncIndex, Name, Result};

struct MyEnv;
impl Env for MyEnv {
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex> {
        // Map a callee to its wasm function index.
        todo!()
    }
    // Optional: type_index (call_indirect), func_table_slot (func_addr),
    // symbol_value, exception_tag (try_call handlers).
}

let compiled = compile_function(&func, &Config::default(), &mut MyEnv)?;
// compiled.params / results / locals / body, or:
let wasm_fn: wasm_encoder::Function = compiled.to_wasm_function();
```

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
| `module: ModuleLayout` | `memory64`, `memory_index`, `table_index`, `stack_pointer_global`, `tls_base_global`, `pinned_reg_global`, `atomic_lock_table_addr`, `relocatable`, `demote_multi_results` | Target module layout; must match the embedder's module definition. |
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

## Supported features

**Scalars (`i8`–`i64`, `f32`, `f64`):** integer and float arithmetic, bitwise
ops (including on floats), shifts and rotates, comparisons, `select`/`bitselect`,
conversions (including saturating `fcvt_to_{s,u}int_sat` to `i8`/`i16`),
overflow-flag arithmetic (`*_overflow`, `*_overflow_cin`, `*_overflow_bin`,
`*_overflow_trap`), loads and stores (all widths, negative offsets, little- and
big-endian), stack slots, `call`, `call_indirect`, `return_call(_indirect)`,
`func_addr`, `symbol_value`, `tls_value`, `br_table`, traps,
`get_stack_pointer`/`get_frame_pointer`, and the pinned register (via
`ModuleLayout::pinned_reg_global`).

**Atomics (`src/lower_atomic.rs`):** uses wasm threads instructions for loads,
stores, `fence`, compare-exchange, and native RMW ops (`add`, `sub`, `and`,
`or`, `xor`, `xchg`). `nand`, `umin`/`umax`/`smin`/`smax`, and big-endian RMWs
lower to a compare-exchange loop. `i128` atomics (`atomic_load`, `atomic_store`,
`atomic_cas`, `atomic_rmw`) expand in `src/legalize_i128.rs` to two `i64`
accesses guarded by a 64-entry striped `i32` spinlock table at
`ModuleLayout::atomic_lock_table_addr`.

**`i128` and `f128` (`src/legalize_i128.rs`):** legalized into `(lo, hi)` pairs
of `i64`. For `i128`, arithmetic, bitwise ops, shifts, rotates, comparisons,
`select`, extensions/truncations, `clz`/`ctz`/`popcnt`/`cls`, `bswap`/`bitrev`,
loads/stores, atomics, calls, block parameters, overflow add/sub, and 128-bit
vector bitcasts expand to `i64` ops; division and remainder use a 128-step
shift-subtract loop; `i128`-to-float conversions round to odd in `i64` before
converting to avoid double rounding. For `f128`, `f128const`, loads/stores,
calls, block parameters, `select`, bitwise ops, `fneg`/`fabs`/`fcopysign`, and
bitcasts to/from `i128` and 128-bit vectors are supported—matching Cranelift's
`x64`, `aarch64`, and `riscv64` backends, which delegate `binary128` arithmetic
to `compiler-rt` libcalls.

**Exceptions:** `try_call` and `try_call_indirect` lower to `try_table` blocks
whose `catch` clauses target one block per handler. `Env::exception_tag` maps
each CLIF exception tag to a wasm tag carrying two pointer-sized payload values.
`default` (`catch_all`) handlers cannot access payloads because wasm's
`catch_all` carries none. Throwing is performed by the embedder's runtime via
wasm `throw` (so upstream `throw.clif`, which relies on Cranelift's native stack
unwinder, is unsupported).

**`f16` and `fma` (`src/lower_float.rs`):** emulated with exact IEEE 754
rounding:
- `f16` arithmetic promotes to `f32`, computes, and rounds back to `f16`
  (provably free of double rounding for `+`, `-`, `*`, `/`, and `sqrt`).
  `fneg`, `fabs`, and `fcopysign` manipulate bits directly.
- `fma` on `f32` and `f16` (scalar and `f32x4`/`f32x2`) computes `a * b + c` in
  `f64` using round-to-odd, then rounds once to the target type.
- `fma` on `f64` (scalar and `f64x2`) unpacks finite non-zero operands into
  53-bit significands, computes the exact 106-bit product as a 128-bit `(lo, hi)`
  pair, aligns and adds `c` with a sticky bit, and rounds once to nearest-even.

**SIMD (`src/lower_simd.rs`):** 128-bit vectors (`i8x16`, `i16x8`, `i32x4`,
`i64x2`, `f32x4`, `f64x2`), narrower vectors (`i32x2`, `i8x8`, `f32x2`, `i16x2`,
etc.), and dynamic vectors map to `v128` (`f16x8` supports lane movement only).
Operations without a direct 1-to-1 wasm instruction are emulated:
- `imul.i8x16`, `umulhi`/`smulhi`: widening multiplies plus shuffles (or scalar
  per lane on `i64x2`).
- `iadd_pairwise`: two shuffles and an add.
- `i64x2` unsigned comparisons and `umin`/`umax`: bias by flipping sign bits.
- Saturating add/sub on 32/64-bit lanes: wrapping arithmetic with overflow
  select. `avg_round`: `(a | b) - ((a ^ b) >> 1)`. `sqmul_round_sat.i32x4`:
  widening 64-bit products.
- `snarrow`/`unarrow`/`uunarrow`: clamp then narrow (or clamp and shuffle from
  `i64x2`).
- `popcnt` on 16/32/64-bit lanes: `i8x16.popcnt` summed pairwise.
- `rotl`/`rotr`: two shifts and an `or`.
- `i64x2` ↔ `f64x2` conversions: scalar conversion per lane.
- Narrow vectors: reductions and lane-combining ops (`vall_true`, `vany_true`,
  `vhigh_bits`, `iadd_pairwise`, high-half widening, narrowing) mask or shift to
  ignore unused upper lanes.

**Unsupported** (returns `Error::Unsupported`):
- `f128` arithmetic and conversions (handled via libcalls in Cranelift).
- `umul_overflow`/`smul_overflow`/`umulhi`/`smulhi` on `i128`.
- Big-endian vector loads/stores and `f16` vector arithmetic.
- `try_call` with `i128`/`f128` arguments, results, or payloads; dynamic stack
  slots; `get_return_address`.
- In relocatable objects: `try_call`, `i128` atomics, and tail calls with
  by-value struct arguments.

## Testing

```sh
cargo test --release                     # CLIF runtests + differential fuzzing
cargo test --release --features module   # also tests relocatable objects with rust-lld
FUZZ_ITERS=30000 cargo test --release --test fuzz -- --nocapture

# Cranelift's upstream runtests (wasmtime v49.0.2 matches Cranelift 0.136.2):
git clone --depth 1 --branch v49.0.2 --filter=blob:none --sparse \
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
- Upstream runtests: 13,729 run commands pass and 0 fail across 393 of 395
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

## Benchmarks

```sh
cargo bench --bench codegen          # BENCH_QUICK=1 for a faster run
UPSTREAM_RUNTESTS=wasmtime/cranelift/filetests/filetests/runtests cargo bench --bench codegen
cargo run --release --example compile_loop   # loop for profiling
cargo run --example wasm2wat -- target/bench/kernels-default.wasm
```

`benches/codegen.rs` compares `naive` (`stack_values` and `use_tee` off),
`no-coalesce`, and `default` configs across three corpora:
- 9 kernels in `tests/harness/kernels.rs` (`fib_iter`, `sieve`, `matmul`,
  `crc32`, `sort`, `mandel`, `collatz`, `dispatch`, `fib_rec`), verified against
  the interpreter in `tests/kernels.rs`.
- 1,000 random functions from `clifgen`.
- Upstream Cranelift runtests (when `UPSTREAM_RUNTESTS` is set).

It measures code size (bytes, locals, `local.*` counts), backend compile time,
V8 compile time, and kernel execution time in wasmtime, V8 Liftoff
(`--liftoff-only`), and V8 TurboFan (`--no-liftoff`).

### Results (Node 26, wasmtime 49)

Kernel runtime (median ms):

| kernel | Liftoff naive | Liftoff no-coalesce | Liftoff default | TurboFan default | wasmtime default |
|---|---:|---:|---:|---:|---:|
| fib_iter | 47.6 | 19.6 | 19.8 | 7.5 | 13.6 |
| sieve | 4.9 | 3.7 | 3.8 | 1.9 | 2.6 |
| matmul | 4.7 | 4.5 | 4.8 | 1.0 | 1.3 |
| crc32 | 24.2 | 24.1 | 24.8 | 15.6 | 15.9 |
| sort | 4.1 | 3.0 | 3.1 | 1.3 | 2.1 |
| mandel | 14.6 | 14.2 | 14.3 | 5.4 | 6.7 |
| collatz | 79.4 | 71.7 | 52.2 | 32.7 | 48.0 |
| dispatch | 67.8 | 51.9 | 47.9 | 34.4 | 38.7 |
| fib_rec | 5.2 | 5.3 | 5.3 | 2.8 | 2.8 |

Key takeaways:
- Optimizing tiers (TurboFan, wasmtime) run `naive` and `default` code at the
  same speed because they reconstruct SSA from locals.
- Baseline tiers (Liftoff) benefit heavily from output quality: expression trees
  speed up `fib_iter` by 2.4×, and move coalescing cuts `collatz` runtime by
  27% and `dispatch` by 8% (while eliminating 33% of `local.set`s across the
  random corpus).
- Compile throughput is ~4M CLIF instructions/s (~250 ns/instruction). Running
  `binaryen.js -O2` on the same modules takes 500–1000× longer.

## Possible next steps

- Remaining unsupported operations (`i128` widening multiplies, `f128` software
  arithmetic).
- Operand commutation and pre-passes to expose more single-use expression trees.
- Compile-speed tuning: local allocation, analysis, and emission each take ~1/3
  of compile time; caching `operands()`/`value_def` lookups across passes is the
  main remaining lever.

## License

Apache-2.0 WITH LLVM-exception, the same as Cranelift. See `LICENSE`.
