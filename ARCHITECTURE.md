# Architecture

`clif-to-wasm` translates a Cranelift IR (`CLIF`) function into a WebAssembly
function body.

## Why compiling CLIF to WebAssembly is nontrivial

CLIF and WebAssembly have different execution models:

1. **Control flow**: CLIF is a graph of basic blocks, and you can jump from
   block to block with instructions like `jump`, `brif`, and `br_table`.
   WebAssembly, by contrast, has no arbitrary jumps (`goto`). Branches can jump
   only to enclosing scopes when in a `block`, `loop`, or `if`.
2. **Data flow**: CLIF is in Static Single Assignment (SSA) form, where each
   instruction defines values that are passed to other blocks as block
   parameters (`jump block1(v3, v2)`). WebAssembly is a stack machine with
   mutable local variables.
3. **Types**: CLIF has `i8`, `i16`, `i32`, `i64`, `i128`, `f16`, `f32`, `f64`,
   `f128`, and vectors of various widths. WebAssembly only has `i32`, `i64`,
   `f32`, `f64`, and `v128`.

Because WebAssembly is a stack machine with an unlimited number of local
variables, it does not need Cranelift's machine-code register allocator (`regalloc2`)
or `TargetIsa` instruction-selection framework. Instead, `clif-to-wasm` works
directly on `cranelift_codegen::ir::Function` in a sequence of small passes
orchestrated by `src/compile.rs`.

## Compiler passes

### 1. Legalization (`src/legalize_i128.rs`, `src/legalize_abi.rs`)

Before analysis begins, `legalize_i128::prepare` scans the function once to see
if it uses value aliases, `i128`/`f128` types, or multi-value returns that need
ABI demotion. In the common case where none of these are present, the function
is compiled directly without cloning.

When legalization is needed, the function is cloned and rewritten in place:
- **128-bit scalars (`i128`, `f128`)** are split into `(lo, hi)` pairs of `i64`
  values across block parameters, function signatures, and instructions.
  128-bit division and remainder expand into a shift-subtract loop, and `i128`
  atomics expand into a spinlock loop around two `i64` accesses.
- **Multi-result demotion (`demote_multi_results`)**: LLVM's Basic C ABI for
  WebAssembly returns at most one scalar value directly; functions with multiple
  results write them to a caller-provided pointer in linear memory. When this
  option is enabled, `legalize_abi` rewrites multi-result function signatures
  and calls to pass a hidden first return-area pointer.

### 2. CFG and liveness analysis (`src/analysis.rs`)

`FuncInfo::compute` walks the control-flow graph once to compute:
- Reverse postorder (RPO) of reachable blocks, predecessor/successor lists, and
  immediate dominators.
- Loop headers (targets of back-edges) and merge blocks (blocks reached from two
  or more forward edges), plus whether the CFG is reducible.
- Backward liveness and use counts for every SSA value. Dead instructions and
  unused block parameters are ignored by later passes.

### 3. Expression tree folding (`src/expr_trees.rs`)

A naive translator would store the result of every CLIF instruction in a wasm
local variable and call `local.get` for every use. But since most values in a
basic block are used only once, right after they are computed, we can instead
put them on the WebAssembly operand stack.

`expr_trees::build` folds a single-use instruction into its user within the same
basic block when doing so preserves program behavior:
- Constants (`iconst`, `f32const`, `f64const`, `vconst`) are cheap and side-effect
  free, so they are re-emitted inline at every use rather than stored in locals.
- Instructions with side effects (loads, calls, divisions that can trap) can
  still be folded into a later instruction as long as no other effectful
  instruction runs between the definition and the use. Each instruction gets an
  "effect color" (the count of effectful instructions before it in the block),
  making this check a constant-time subtraction.

`expr_trees::operands` defines the operand evaluation order that local
allocation and code emission both follow.

### 4. Local variable allocation (`src/locals.rs`)

Values that are not folded onto the operand stack (multi-use values, values live
across basic blocks, and block parameters) need WebAssembly local variables.

`locals::allocate` computes a live range for each such value as a sorted list of
half-open position intervals `[start, end)`, then assigns locals greedily by
start position:
1. **Move coalescing**: In CLIF, `jump block1(v3)` copies `v3` into `block1`'s
   parameter `v4`. If `v3` is not used after the jump, its live range ends right
   where `v4`'s begins. The allocator tries to give `v3` and `v4` the same wasm
   local first; when they share a local, the branch does not need to emit any
   copy instructions (`Locals::move_needed`).
2. **Expired locals**: A min-heap tracks locals whose previous value's live
   range has already ended before the current value starts.
3. **Live-range holes**: If no local has fully expired, the allocator checks up
   to 32 recently created locals of the same wasm type to see if this value fits
   into a gap between their active segments.
4. **Fresh local**: Otherwise, it allocates a new local of that `ValType`.

### 5. Control-flow structuring (`src/structure.rs`)

`structure::structure` converts CLIF's basic-block graph into nested WebAssembly
control flow using Norman Ramsey's *Beyond Relooper* algorithm (ICFP 2022),
walking the dominator tree with an explicit task stack:

- A block with a single predecessor is emitted inline at the branch site.
- A **merge block** (multiple forward predecessors) gets a wasm `block ... end`
  scope placed around the code that precedes it; branching to that scope jumps
  forward to the merge block.
- A **loop header** (target of a back-edge) gets a wasm `loop ... end` scope;
  branching to that scope jumps backward to the top of the loop.
- If the function has **irreducible control flow** (a loop with multiple entry
  points that cannot be expressed with nested `loop`s alone), the pass falls
  back to a `loop` + `br_table` state-machine dispatcher.

### 6. Emission and instruction lowering (`src/emit.rs`, `src/lower*.rs`)

`Emitter` walks the structured control-flow tree and emits WebAssembly
instructions:
- **Shadow stack frame**: If the function uses `stack_addr` or passes structs by
  value, it subtracts its frame size from the stack-pointer global
  (`__stack_pointer`) on entry, keeping 16-byte alignment, and restores it
  before every return or tail call. Because WebAssembly linear memory has no
  guard pages, `ModuleLayout::stack_limit_global` (`__stack_limit`) can
  optionally be set to the lower bound of the stack; the function then traps on
  entry if `sp < limit + size` instead of overflowing into memory below the
  stack.
- **Narrow integers (`i8`, `i16`)**: Stored in `i32` locals with their upper 24
  or 16 bits zeroed. Operations that can set high bits (`iadd`, `isub`, `imul`,
  `ineg`, `bnot`, `ishl`, `sshr`) mask the result back to 8 or 16 bits; signed
  operations (`sdiv`, `srem`, `sshr`, signed `icmp`) sign-extend their inputs
  first.
- **Emulated floating-point (`src/lower_float.rs`)**: `f16` values are stored as
  16-bit bit patterns in `i32`. Arithmetic promotes `f16` to `f32` (which has
  enough precision that double rounding is exact for `+`, `-`, `*`, `/`, and
  `sqrt`) and rounds back to `f16`. `fma` on `f16`/`f32` computes in `f64` using
  round-to-odd; `fma` on `f64` uses an exact 128-bit integer significand
  calculation.
- **SIMD (`src/lower_simd.rs`) and atomics (`src/lower_atomic.rs`)**: Vectors
  narrower than 128 bits occupy the low lanes of a `v128`. Operations that wasm
  SIMD or threads proposals lack natively (such as unsigned 64-bit lane
  comparisons, `imul.i8x16`, and atomic `nand`/min/max) are lowered to short
  instruction sequences or compare-and-swap loops.
- **Peephole (`use_tee`)**: When `local.set x` is immediately followed by
  `local.get x`, the emitter replaces the pair with `local.tee x`.

## Annotated example

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

`clif-to-wasm` emits:

```wasm
(func (param i32 i32 i32) (result i32)
  (local i32 i32 i32)
  i32.const 0              ;; jump block1(v3, v2): push args onto stack,
  local.get 2              ;;   then write target block params in reverse order.
  local.set 3              ;;   n   := v2 (local 3)
  local.set 2              ;;   acc := 0  (local 2, reusing dead param v2)
  loop                     ;; block1 is a loop header
    local.get 3
    if                     ;; brif v5
      block                ;; block5 has two predecessors, so a `block` wraps
        local.get 0        ;;   the code before it; `br 1` jumps to block5.
        i32.load           ;; v6, v7, v8 fold into a single stack expression
        local.get 1
        i32.load
        i32.mul
        local.tee 4        ;; v8 is used more than once, so it gets local 4
        i32.const 0        ;; constant v9 is emitted inline
        i32.lt_s
        if                 ;; brif v10
          local.get 4      ;; block3 has one predecessor: emitted inline
          i32.const -1
          i32.mul          ;; ineg
          local.set 5      ;; v12 := v11
          br 1             ;; jump to end of enclosing `block` (block5)
        else
          local.get 4
          local.set 5      ;; v12 := v8
          br 1
        end
      end
      local.get 2          ;; block5: v13 and v15 fold directly into the
      local.get 5          ;;   back-edge jump arguments
      i32.add
      local.get 3
      i32.const 1
      i32.sub
      local.set 3          ;; n   := v15
      local.set 2          ;; acc := v13
      br 1                 ;; jump back to start of `loop` (block1)
    else
      local.get 2          ;; block4 has one predecessor: emitted inline
      local.tee 4
      return
    end
  end
  unreachable
)
```

## Test suite

- `tests/clif/*.clif`: Cranelift `.clif` test files with `; run:` directives.
  Each test is compiled, validated with `wasmparser`, and executed in Wasmtime
  under five configurations (default, `use_tee = false`, `coalesce = false`,
  64-bit memory `AddressType::I64`, and `demote_multi_results = true`).
- `tests/fuzz.rs`: Differential fuzzer that generates random CLIF functions
  (scalars, SIMD, and `i128`) and checks that Wasmtime produces the same results
  as `cranelift-interpreter`. Run longer with:
  ```sh
  FUZZ_ITERS=30000 cargo test --release --test fuzz -- --nocapture
  ```
  Failures can be replayed or minimized with `FUZZ_SEED=<n>`, `FUZZ_BUDGET=<n>`,
  and `FUZZ_ONLY=scalar|simd|i128`.
- `tests/cfg_fuzz.rs`: Property tests that feed random reducible and irreducible
  control-flow graphs into `structure.rs` and `locals.rs`.
- `tests/simd_ops.rs` and `tests/narrow_vectors.rs`: Differential tests for
  128-bit and narrower vector types against `cranelift-interpreter`, including
  tests with non-zero bits in unused upper lanes.
- `tests/float.rs`: Exhaustive tests of `f16` conversions across all 65,536 bit
  patterns and rounding midpoints, plus `f16`/`f32`/`f64` `fma` tests against
  exact software reference implementations.
- `tests/module.rs`: Tests dynamic linking between a host module and a JIT
  module sharing memory, function table, and stack pointer.
- `tests/object.rs` (`--features module`): Compiles functions and data into
  relocatable `.o` files via `WasmObjectModule`, links them with
  `rust-lld -flavor wasm`, and runs the linked module in Wasmtime.
- `examples/upstream.rs`: Runs Cranelift's upstream filetest suite (14,043
  `; run:` checks pass across 396 of 398 files; the two skipped files test
  native stack unwinding via `get_return_address` and an interpreter-only
  saturating float-to-`i128` behavior).
