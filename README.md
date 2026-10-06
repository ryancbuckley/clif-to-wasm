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

| Pass | File | What it does |
|------|------|--------------|
| Analysis | `src/analysis.rs` | RPO, deduplicated CFG edges, dominators, loop headers, merge nodes, reducibility, value-level DCE, use counts |
| Stackify | `src/stackify.rs` | Greedy backwards tree building: a single-use def in the same block is folded into its user unless that would reorder side effects. The check is O(1) using Cranelift-style side-effect colors. Constants are rematerialized at every use |
| Locals | `src/locals.rs` | Values not passed on the stack get locals. Live intervals are hulls computed from SSA liveness; linear-scan coloring per wasm type |
| Structure | `src/structure.rs` | Ramsey's "Beyond Relooper" (ICFP 2022) over the dominator tree, run with an explicit task stack. Irreducible CFGs fall back to a `loop` + `br_table` dispatcher |
| Lower | `src/lower.rs` | Per-opcode emission, plus a `local.set x; local.get x` → `local.tee x` peephole |

The stackifier's safety condition: `color[i]` counts the effectful
instructions before `i` in its block. An effectful def `d` may join the tree
rooted at `r` iff `color[r] - color[d] - 1` equals the number of effectful
defs already moved into that tree. Block arguments of multi-way branches
accept only pure trees, so an unconditional effect never becomes conditional.

### Representation choices

- `i8`/`i16` live in `i32` locals and are always kept zero-extended. Ops that
  can dirty the high bits mask their result; signed ops sign-extend inputs.
- Every value that crosses a block boundary lives in a local. All
  `block`/`loop`/`if` are `[] -> []`.
- Block-argument moves push every argument before setting any parameter, so
  parallel moves (swaps) need no temporaries.
- Stack slots live on a shadow stack in linear memory, through a
  stack-pointer global (`Config::stack_pointer_global`).
- Traps become `unreachable`; trap codes are not preserved.

## Usage

```rust
use cranelift_wasm_backend::{compile_function, Config, Env, Name, Result};

struct MyEnv;
impl Env for MyEnv {
    fn func_index(&mut self, name: Name<'_>) -> Result<u32> {
        // Map a callee to its wasm function index.
        todo!()
    }
    // Optional: type_index (call_indirect), func_table_slot (func_addr),
    // symbol_value.
}

let compiled = compile_function(&func, &Config::default(), &mut MyEnv)?;
// compiled.params / results / locals / body, or:
let wasm_fn: wasm_encoder::Function = compiled.to_wasm_function();
```

`ModuleBuilder` (`src/module.rs`) assembles a complete module (imports,
memory, a funcref table, globals, exports) for simple cases and tests.

`Config` options: `memory64`, `memory_index`, `table_index`,
`stack_pointer_global`, `tail_calls` (use `return_call`, or fall back to
call + return), and `use_tee`.

## Supported

Scalar `i8`–`i64`, `f32` and `f64`: integer and float arithmetic, bit ops,
shifts and rotates, comparisons, `select`/`bitselect`, conversions,
loads/stores (all widths, negative offsets), stack slots, `call`,
`call_indirect`, `return_call(_indirect)`, `func_addr`, `symbol_value`,
`br_table`, traps, `get_stack_pointer`/`get_frame_pointer`.

Not yet supported (functions using these return `Error::Unsupported`):
SIMD types, `i128`, `f16`/`f128`, big-endian memory access,
`bswap`/`bitrev`, multi-result overflow ops (`*_overflow`), 64-bit
`umulhi`/`smulhi`, narrow `fcvt_*_sat`, bitwise ops on floats, atomics,
`fma`, exceptions (`try_call`), and dynamic stack slots.

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
  compiled, validated with `wasmparser` and executed in wasmtime under three
  configs (default, no-tee, memory64).
- `tests/fuzz.rs` builds random functions with `cranelift-frontend`
  (arithmetic, conversions, loads/stores, diamonds, loops, `br_table`,
  traps). It runs them in `cranelift-interpreter` and as wasm, and compares
  results or traps.
- Upstream runtests: 8603 run commands pass and 0 fail. The rest are skipped
  as unsupported, mostly SIMD and `i128`.

### Notes

- The interpreter traps on `INT_MIN srem -1`, which Cranelift defines as 0.
  The fuzzer avoids that case.
- Tests originally ran on wasmi 2.0.0, which miscomputed one fuzz case that
  V8 and the interpreter agree on. That is why execution now uses wasmtime.

## Possible next steps

- Better coalescing of block-parameter moves (intervals with holes, or
  move hints).
- `i128` (as i64 pairs), SIMD (`v128`), atomics, and the remaining scalar
  ops.
- Operand commutation and pre-passes that increase the number of
  single-use values (as LLVM does).
