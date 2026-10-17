# cranelift-wasm-backend

A WebAssembly backend for [Cranelift](https://cranelift.dev): it compiles a
Cranelift IR `Function` into a wasm function body.

The motivating use case is a JIT that itself runs inside wasm and emits wasm
for the host to compile again. Compile speed therefore matters more than code
size. The host's optimizing tier undoes most local-heavy code anyway, and
`wasm-opt` can shrink the output when size matters.

This is not a Cranelift `TargetIsa`. Cranelift's machine backends are built
around register allocation, which a stack machine with unlimited locals does
not need. The backend works directly on the IR instead.

## Pipeline

| Pass | File | What it does | Switch |
|------|------|--------------|--------|
| Prepare & Legalize | `src/legalize_i128.rs` | Scans once for value aliases and `i128`/`f128`; when present, clones the function, resolves aliases, and rewrites 128-bit scalar values into `(lo, hi)` pairs of `i64` (signatures, block params, calls) before anything else runs | |
| Analysis | `src/analysis.rs` | RPO, deduplicated CFG edges, dominators, loop headers, merge nodes, reducibility, and single-pass backwards value/instruction liveness + use counts | |
| Expression trees | `src/expr_trees.rs` | Greedy backwards tree building: a single-use def in the same block is folded into its user unless that would reorder side effects. The check is O(1) using Cranelift-style side-effect colors. Constants are rematerialized at every use. `operands()` here defines evaluation order for every later pass | `stack_values` |
| Locals | `src/locals.rs` | Builds per-block live-range segments for values that need a local, collects block-argument moves, and performs greedy coloring per wasm type. A value first tries its move partner's local (argument ↔ parameter), so most block-argument moves disappear | `coalesce` |
| Structure | `src/structure.rs` | Ramsey's "Beyond Relooper" (ICFP 2022) over the dominator tree, run with an explicit task stack in `Structurer`. Irreducible CFGs fall back to a `loop` + `br_table` dispatcher | |
| Emit | `src/emit.rs` | Function prologue/epilogue (shadow stack and narrow parameter masking), value emission, operand cursor, scoped LIFO scratch locals, and byte encoding with a `local.set x; local.get x` → `local.tee x` peephole | `use_tee` |
| Lower | `src/lower.rs`, `src/lower_simd.rs`, `src/lower_atomic.rs`, `src/lower_float.rs` | One case per opcode. Vector operations, atomics, and emulated floating point (`f16`, `fma`, and shared scalar/vector `FloatCC` lowering) have their own modules | |

Cross-module contracts are each stated in one place:

- **Evaluation order:** `expr_trees::operands` defines it, and lowering pulls
  operands from it through `Emitter::operand`. Branches are the one
  documented exception.
- **When a block-argument move can be skipped:** `Locals::move_needed`
  depends on the live-range conventions documented in `locals.rs`.

The safety condition for folding into a tree: `color[i]` counts the effectful
instructions before `i` in its block. An effectful def `d` may join the tree
rooted at `r` iff `color[r] - color[d] - 1` equals the number of effectful
defs already moved into that tree. Block arguments of multi-way branches
accept only pure trees, so an unconditional effect never becomes conditional.

### Representation choices

- `i8`/`i16` live in `i32` locals and are always kept zero-extended. Ops that
  can dirty the high bits mask their result; signed ops sign-extend inputs.
- `f16` lives in an `i32` as its bits, zero-extended like an `i16`.
- Vectors narrower than 128 bits live in the low bytes of a `v128`; the other
  bytes are unspecified. Dynamic vectors are treated as their fixed-size
  type (a scale of 1).
- Big-endian loads and stores reverse bytes, since wasm memory is
  little-endian.
- Every value that crosses a block boundary lives in a local. All
  `block`/`loop`/`if` are `[] -> []`.
- Block-argument moves push every argument before setting any parameter, so
  parallel moves (swaps) need no temporaries.
- Stack slots live on a shadow stack in linear memory, through a
  stack-pointer global (`ModuleLayout::stack_pointer_global`).
- Traps become `unreachable`; trap codes are not preserved.

## Usage

```rust
use cranelift_wasm_backend::{compile_function, Config, Env, FuncIndex, Name, Result};

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

`ModuleBuilder` (`src/module.rs`) assembles a complete module whose indices
agree with your `Config` and `Env` answers. It handles both kinds of module:
- **Self-contained** modules define their own memory and globals, plus a
  funcref table where slot `i` holds function `i`. The tests use this.
- **Modules linked into a running program**, as a JIT running inside wasm
  needs. Such a module imports the program's memory, table and stack pointer
  global, and `place_in_table` puts its functions in free slots of that table.
  `tests/module.rs` walks through this case.

`Config` has three groups of settings:

| Group | Fields | Meaning |
|---|---|---|
| `module: ModuleLayout` | `memory64`, `memory_index`, `table_index`, `stack_pointer_global`, `pinned_reg_global`, `atomic_lock_table_addr` | Facts about the module the code goes in; they must match how the embedder builds it. |
| `engine: EngineFeatures` | `tail_calls` | Wasm features the engine supports. Without tail calls, `return_call` falls back to call + return. |
| `opts: Optimizations` | `stack_values`, `coalesce`, `use_tee` | Output-quality switches, all on by default; `Optimizations::none()` turns them all off. |

For example, `config.module.stack_pointer_global = Some(sp)` enables stack
slots, `config.module.table_index = Some(t)` enables `call_indirect`,
`config.module.atomic_lock_table_addr = Some(addr)` enables `i128` atomics,
and `config.opts.coalesce = false` disables move coalescing.

## Supported

Scalar `i8`–`i64`, `f32` and `f64`: integer and float arithmetic, bit ops
(on floats too), shifts and rotates, comparisons, `select`/`bitselect`,
conversions (including saturating ones to `i8`/`i16`), arithmetic with an
overflow flag (`*_overflow`, with carry or borrow in), loads and stores (all
widths, negative offsets, either byte order), stack slots, `call`,
`call_indirect`, `return_call(_indirect)`, `func_addr`, `symbol_value`,
`br_table`, traps, `get_stack_pointer`/`get_frame_pointer`, and the pinned
register (as a global, `ModuleLayout::pinned_reg_global`).

Atomics (`src/lower_atomic.rs`) use the wasm threads proposal's
instructions: loads, stores, `fence`, compare-exchange, and the
read-modify-writes that wasm has (`add`, `sub`, `and`, `or`, `xor`,
`xchg`). `nand`, `umin`/`umax`/`smin`/`smax` and big-endian
read-modify-writes become a compare-exchange loop. `i128` atomics
(`atomic_load`, `atomic_store`, `atomic_cas`, `atomic_rmw`) expand in
`src/legalize_i128.rs` into a spinlock-protected pair of `i64` atomic
accesses using a 64-entry striped `i32` lock table at
`ModuleLayout::atomic_lock_table_addr`.

`i128` and `f128` are supported by legalizing them into pairs of `i64`
(`src/legalize_i128.rs`). At the wasm level an `i128` or `f128` parameter or
result is two `i64`s, low half first. For `i128`, arithmetic, bitwise ops,
shifts and rotates, comparisons, `select`, extensions and truncations,
`clz`/`ctz`/`popcnt`/`cls`, `bswap`/`bitrev`, loads and stores, atomics,
calls, block parameters, add and subtract with an overflow flag, and bitcasts
to and from 128-bit vectors expand into `i64` code; division and remainder
become a 128-iteration shift-subtract loop, and conversions from `i128` to
floats are correctly rounded (round to odd, then convert). For `f128`, constants
(`f128const`), loads, stores, calls, block parameters, `select`, bitwise ops,
`fneg`/`fabs`/`fcopysign`, and bitcasts to and from `i128` and 128-bit vectors
are supported — the same surface that Cranelift's own `x64`, `aarch64`, and
`riscv64` backends implement (relying on compiler-rt libcalls for IEEE 754
`binary128` arithmetic).

Exceptions use wasm's exception handling. A `try_call` becomes a call
inside a `try_table` whose `catch` clauses branch to one block per handler;
`Env::exception_tag` maps each Cranelift exception tag to a wasm tag, whose
payload is the calling convention's two pointer-sized values. A catch-all
handler (`default`) cannot use the payload, since wasm's `catch_all` provides
none. Throwing is up to the embedder's runtime (a wasm `throw`), as it is in
Cranelift, where only a native unwinder can throw; so the upstream
`throw.clif`, which uses that unwinder, is not supported.

`f16` and `fma` (`src/lower_float.rs`) are emulated, with correctly rounded
results:

- `f16` arithmetic converts to `f32`, computes there and rounds back; for
  `+`, `-`, `*`, `/` and `sqrt` that double rounding is provably harmless.
  `fneg`, `fabs` and `fcopysign` are bit operations.
- `fma` on `f32` and `f16` (scalar and `f32` vector lanes) computes the sum
  in `f64` rounded to odd, then rounds once to the result type.
- `fma` on `f64` (scalar and `f64x2` vector lanes) unpacks finite non-zero
  operands into 53-bit integer significands, computes the exact 106-bit
  product in a 128-bit `(lo, hi)` pair, aligns and adds/subtracts the third
  operand with a sticky bit, and rounds once to nearest-even.

Vectors of 128 bits (`i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2`)
map to `v128` (`src/lower_simd.rs`); so do narrower ones (`i32x2`, `i8x8`,
`f32x2`, `i16x2`, …) and dynamic ones. `f16x8` supports moving lanes
around, not arithmetic. Most operations map to a single wasm SIMD
instruction. These have no direct wasm counterpart and are emulated:

- `imul.i8x16`, `umulhi`/`smulhi`: widening multiplies plus a shuffle; on
  `i64x2`, the scalar algorithm per lane.
- `iadd_pairwise`: two shuffles and an add.
- `i64x2` unsigned comparisons and min/max: flip the sign bits.
- Saturating add/subtract on 32/64-bit lanes: the wrapping result, with
  overflowed lanes replaced. `avg_round` on them: `(a | b) - ((a ^ b) >> 1)`.
  `sqmul_round_sat` on `i32x4`: 64-bit products.
- `snarrow`/`unarrow`/`uunarrow`: clamp, then narrow (from `i64x2`, clamp
  and shuffle).
- `popcnt` of 16/32/64-bit lanes: byte popcounts summed pairwise.
- `rotl`/`rotr`: two shifts and an `or`.
- `i64x2` ↔ `f64x2` conversions: lane by lane through scalars.
- Narrower vectors: the operations that see the unused bytes
  (`vall_true`, `vany_true`, `vhigh_bits`, `iadd_pairwise`, widening the
  high half, narrowing) account for the lane count.

Not yet supported (functions using these return `Error::Unsupported`):

- `f128` arithmetic and conversions (matching Cranelift's x64/aarch64/riscv64
  backends).
- `umul_overflow`/`smul_overflow`/`umulhi`/`smulhi` on `i128`.
- Big-endian vector loads and stores, and arithmetic on `f16` lanes.
- `try_call` with `i128`/`f128` arguments, results or payloads; dynamic stack
  slots; `get_return_address`.

## Testing

```sh
cargo test --release                     # local CLIF runtests + differential fuzzing
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

- `tests/clif/*.clif` are Cranelift filetests with `; run:` lines. Each one is
  compiled, validated with `wasmparser` and executed in wasmtime under four
  configs (default, no-tee, no-coalesce, memory64). As in Cranelift's own
  runner, a run command calls the function it is attached to.
- `tests/fuzz.rs` builds random functions with `cranelift-frontend`
  (arithmetic, conversions, loads/stores, diamonds, loops, `br_table`,
  traps, overflow flags) and runs them in `cranelift-interpreter` and as wasm, comparing
  results or traps. Each seed gives three functions: scalar only, with
  vector code, and with `i128` code.
- `tests/simd_ops.rs` checks every vector operation on every vector type,
  128-bit and narrower, with random inputs, against the interpreter.
- `tests/float.rs` checks the emulated floating point against independent
  references: `f16` conversions exhaustively (all 65,536 values, and every
  rounding midpoint with its neighbors), `f16` arithmetic and `fma` against
  exact arithmetic, and `f32` `fma` against `f32::mul_add`, including inputs
  built to expose double rounding.
- `tests/module.rs` links a JIT-style module into a host module, sharing the
  host's memory, table and stack pointer, and calls across in both
  directions.
- The harness provides what upstream tests expect from an embedder:
  libcalls (`CeilF32` and friends, compiled from CLIF), a pinned-register
  global, atomics (wasmtime's `threads` feature), and exception tags. A
  test throws by calling `%throw_tag<N>(i64, i64)`, which the harness
  defines as a wasm `throw` (`tests/clif/exceptions.clif`).
- Upstream runtests: 13,497 run commands pass and 0 fail; 383 of 395 files
  are fully supported. The rest use `f128` (4 files), `f64` `fma` (4),
  `i128` atomics (2) or Cranelift's native unwinder (`throw.clif`). Files
  that only run under
  Cranelift's interpreter
  (`test interpret` without `test run`) are reported separately:
  `i128-conversion.clif` expects the interpreter's saturating
  float-to-`i128` conversion where Cranelift's semantics (and this backend)
  give the exact value or trap.

### Notes

- The interpreter traps on `INT_MIN srem -1`, which Cranelift defines as 0.
  The fuzzer avoids that case.
- NaN bit patterns are nondeterministic in wasm, and the interpreter keeps
  signaling NaNs that hardware quiets. Results compare all NaNs as equal,
  per lane for float vectors. Wherever the fuzzer reinterprets float bits
  as integers (stores, bitcasts, shuffles, bitwise ops), it canonicalizes
  NaNs first.
- The interpreter panics on `avg_round`, and on some operations on 16- and
  32-bit vectors, so `simd_ops` skips those; `avg_round` is covered by the
  upstream runtests. It also saturates float-to-`i128` conversions instead of
  trapping, so the fuzzer does not generate those.
- `iconst.i128` is not valid IR in this Cranelift version (the verifier
  rejects it); generated `i128` constants are built with `iconcat`. The
  backend still accepts `iconst.i128`, zero-extending the immediate like
  Cranelift's x64 backend.
- Tests originally ran on wasmi 2.0.0, which miscomputed one fuzz case that
  V8 and the interpreter agree on. That is why execution now uses wasmtime.

## Benchmarks

```sh
cargo bench --bench codegen          # BENCH_QUICK=1 for a faster run
UPSTREAM_RUNTESTS=wasmtime/cranelift/filetests/filetests/runtests cargo bench --bench codegen
cargo run --release --example compile_loop   # compile the random corpus in a loop, for profiling
cargo run --example wasm2wat -- target/bench/kernels-default.wasm
```

`benches/codegen.rs` compares a `naive` configuration (`opts.stack_values`
and `opts.use_tee` off), a `no-coalesce` one and the default one on three
corpora:

- nine kernels in `tests/harness/kernels.rs` (fib, sieve, matmul, crc32,
  insertion sort, mandelbrot, collatz, a `br_table` state machine, recursive
  fib). They are built with `cranelift-frontend` the way a JIT would build
  them. `tests/kernels.rs` checks them against the interpreter.
- 1000 random functions from the fuzzer's generator.
- Cranelift's upstream runtests, if `UPSTREAM_RUNTESTS` is set.

It reports:

- output size: bytes, locals and `local.*` ops;
- the backend's own compile time;
- kernel runtime in wasmtime, V8 Liftoff (`--liftoff-only`) and V8
  TurboFan (`--no-liftoff`), the last two when `node` is on the `PATH`;
- V8's compile time for the module.

### Results (node 26, wasmtime 49)

Effect of precise live ranges plus move hints (default configuration,
before → after):

| | random corpus | kernels |
|---|---|---|
| code bytes | 1,240,324 → 1,080,592 (−13%) | 1678 → 1636 |
| locals | 26,968 → 12,282 (−54%) | 63 → 49 |
| `local.get` | 128,546 → 92,772 (−28%) | 210 → 201 |
| `local.set` | 70,171 → 34,397 (−51%) | 116 → 107 |
| compile time | 57 → 64 µs/function (+12%) | |

Of these gains, the move hints alone (`opts.coalesce` off vs. default, on 200 random
functions) account for 33% fewer `local.set`s and 6% fewer bytes. The rest
comes from the precise live ranges.

Kernel runtime, median ms:

| kernel | Liftoff naive | Liftoff default (before) | Liftoff default (now) | TurboFan default | wasmtime default |
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

Findings:

- Optimizing tiers (TurboFan, wasmtime/Cranelift) run naive and default
  output at the same speed (within noise): they rebuild SSA from locals
  anyway.
- The baseline tier is where output quality matters. Expression trees make
  Liftoff up to 2.4× faster than naive output (fib_iter), and coalescing
  block-argument moves gives another 27% on collatz and 8% on dispatch.
- The backend compiles about 4M CLIF instructions per second (≈250 ns per
  instruction). Running binaryen.js `-O2` on the same small modules took
  roughly 500–1000× longer than generating them.
- Tried and dropped: sinking pure single-use values into later blocks so
  they can join an expression tree. On these corpora it changed nothing
  measurable.

## Possible next steps

- The unsupported operations listed above: `f64` `fma` (double-double
  arithmetic), `f128` (software floating point).
- Operand commutation and pre-passes that increase the number of
  single-use values (as LLVM does).
- Compile speed: the allocator, analysis and emission each take roughly a
  third of the time. The main lever left is avoiding repeated
  `operands()`/`value_def` work across passes.
