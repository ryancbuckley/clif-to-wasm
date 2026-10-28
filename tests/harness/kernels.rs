//! Benchmark kernels, built with `cranelift-frontend` the way a JIT would
//! build them (variables, structured loops, stack-allocated arrays).
//!
//! Every kernel is a function `%k_<name>(i32 n) -> i64` that sets up its own
//! data and returns a checksum. Addresses are `i32`, as for a wasm32 host.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F64, I8, I32, I64};
use cranelift_codegen::ir::{
    AbiParam, BlockCall, Function, InstBuilder, JumpTableData, MemFlagsData, Signature, StackSlot,
    StackSlotData, StackSlotKind, Type, UserFuncName, Value,
};
use cranelift_codegen::isa::{CallConv, TargetFrontendConfig};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

pub struct Kernel {
    /// Export name, without the `k_` prefix.
    pub name: &'static str,
    /// An `n` small enough for the interpreter.
    pub small: i32,
    /// An `n` for benchmarking.
    pub large: i32,
}

pub const KERNELS: &[Kernel] = &[
    Kernel {
        name: "fib_iter",
        small: 50,
        large: 30_000_000,
    },
    Kernel {
        name: "sieve",
        small: 100,
        large: 1_000_000,
    },
    Kernel {
        name: "matmul",
        small: 4,
        large: 128,
    },
    Kernel {
        name: "crc32",
        small: 16,
        large: 2_000_000,
    },
    Kernel {
        name: "sort",
        small: 20,
        large: 4096,
    },
    Kernel {
        name: "mandel",
        small: 4,
        large: 400,
    },
    Kernel {
        name: "collatz",
        small: 20,
        large: 300_000,
    },
    Kernel {
        name: "dispatch",
        small: 30,
        large: 20_000_000,
    },
    Kernel {
        name: "fib_rec",
        small: 10,
        large: 30,
    },
];

/// All kernel functions, plus their helpers.
pub fn functions() -> Vec<Function> {
    vec![
        fib_iter(),
        sieve(),
        matmul(),
        crc32(),
        sort(),
        mandel(),
        collatz(),
        dispatch(),
        fib_rec_helper(),
        fib_rec(),
    ]
}

fn flags() -> MemFlagsData {
    MemFlagsData::trusted()
}

fn build(
    name: &str,
    params: &[Type],
    rets: &[Type],
    body: impl FnOnce(&mut FunctionBuilder, &[Value]),
) -> Function {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.extend(params.iter().map(|&t| AbiParam::new(t)));
    sig.returns.extend(rets.iter().map(|&t| AbiParam::new(t)));
    let mut func = Function::with_name_signature(UserFuncName::testcase(name), sig);
    let mut ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut func, &mut ctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let ps = b.block_params(entry).to_vec();
    body(&mut b, &ps);
    b.seal_all_blocks();
    b.finalize(TargetFrontendConfig {
        default_call_conv: CallConv::SystemV,
        pointer_width: target_lexicon::PointerWidth::U32,
        page_size_align_log2: 16,
    });
    func
}

fn var(b: &mut FunctionBuilder, ty: Type, init: Value) -> Variable {
    let v = b.declare_var(ty);
    b.def_var(v, init);
    v
}

fn slot(b: &mut FunctionBuilder, size: u32) -> StackSlot {
    b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, 3))
}

/// `for i in start..end` (signed `i32`).
fn for_range(
    b: &mut FunctionBuilder,
    start: Value,
    end: Value,
    body: impl FnOnce(&mut FunctionBuilder, Value),
) {
    let iv = var(b, I32, start);
    let (header, body_blk, exit) = (b.create_block(), b.create_block(), b.create_block());
    b.ins().jump(header, &[]);
    b.switch_to_block(header);
    let i = b.use_var(iv);
    let c = b.ins().icmp(IntCC::SignedLessThan, i, end);
    b.ins().brif(c, body_blk, &[], exit, &[]);
    b.switch_to_block(body_blk);
    body(b, i);
    let next = b.ins().iadd_imm_s(i, 1);
    b.def_var(iv, next);
    b.ins().jump(header, &[]);
    b.switch_to_block(exit);
}

/// `while cond() { body() }`
fn while_loop(
    b: &mut FunctionBuilder,
    cond: impl FnOnce(&mut FunctionBuilder) -> Value,
    body: impl FnOnce(&mut FunctionBuilder),
) {
    let (header, body_blk, exit) = (b.create_block(), b.create_block(), b.create_block());
    b.ins().jump(header, &[]);
    b.switch_to_block(header);
    let c = cond(b);
    b.ins().brif(c, body_blk, &[], exit, &[]);
    b.switch_to_block(body_blk);
    body(b);
    b.ins().jump(header, &[]);
    b.switch_to_block(exit);
}

fn if_else(
    b: &mut FunctionBuilder,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder),
    els: impl FnOnce(&mut FunctionBuilder),
) {
    let (t, e, m) = (b.create_block(), b.create_block(), b.create_block());
    b.ins().brif(c, t, &[], e, &[]);
    b.switch_to_block(t);
    then(b);
    b.ins().jump(m, &[]);
    b.switch_to_block(e);
    els(b);
    b.ins().jump(m, &[]);
    b.switch_to_block(m);
}

/// `base + (i << shift)`
fn elem(b: &mut FunctionBuilder, base: Value, i: Value, shift: i64) -> Value {
    let off = if shift == 0 {
        i
    } else {
        b.ins().ishl_imm_u(i, shift)
    };
    b.ins().iadd(base, off)
}

fn fib_iter() -> Function {
    build("k_fib_iter", &[I32], &[I64], |b, p| {
        let zero = b.ins().iconst(I64, 0);
        let one = b.ins().iconst(I64, 1);
        let a = var(b, I64, zero);
        let c = var(b, I64, one);
        let start = b.ins().iconst(I32, 0);
        for_range(b, start, p[0], |b, _| {
            let (x, y) = (b.use_var(a), b.use_var(c));
            let t = b.ins().iadd(x, y);
            b.def_var(a, y);
            b.def_var(c, t);
        });
        let r = b.use_var(a);
        b.ins().return_(&[r]);
    })
}

fn sieve() -> Function {
    build("k_sieve", &[I32], &[I64], |b, p| {
        let n = p[0];
        let ss = slot(b, 1 << 20);
        let base = b.ins().stack_addr(I32, ss, 0);
        let zero = b.ins().iconst(I32, 0);
        for_range(b, zero, n, |b, i| {
            let a = elem(b, base, i, 0);
            let one = b.ins().iconst(I8, 1);
            b.ins().store(flags(), one, a, 0);
        });
        let z64 = b.ins().iconst(I64, 0);
        let count = var(b, I64, z64);
        let two = b.ins().iconst(I32, 2);
        let n64 = b.ins().uextend(I64, n);
        for_range(b, two, n, |b, i| {
            let a = elem(b, base, i, 0);
            let flag = b.ins().load(I8, flags(), a, 0);
            let (t, m) = (b.create_block(), b.create_block());
            b.ins().brif(flag, t, &[], m, &[]);
            b.switch_to_block(t);
            let c = b.use_var(count);
            let c = b.ins().iadd_imm_s(c, 1);
            b.def_var(count, c);
            let i64v = b.ins().uextend(I64, i);
            let sq = b.ins().imul(i64v, i64v);
            let j = var(b, I64, sq);
            while_loop(
                b,
                |b| {
                    let jv = b.use_var(j);
                    b.ins().icmp(IntCC::SignedLessThan, jv, n64)
                },
                |b| {
                    let jv = b.use_var(j);
                    let j32 = b.ins().ireduce(I32, jv);
                    let a = elem(b, base, j32, 0);
                    let z = b.ins().iconst(I8, 0);
                    b.ins().store(flags(), z, a, 0);
                    let next = b.ins().iadd(jv, i64v);
                    b.def_var(j, next);
                },
            );
            b.ins().jump(m, &[]);
            b.switch_to_block(m);
        });
        let r = b.use_var(count);
        b.ins().return_(&[r]);
    })
}

fn matmul() -> Function {
    const MAX: u32 = 128;
    build("k_matmul", &[I32], &[I64], |b, p| {
        let n = p[0];
        let ss = slot(b, 3 * MAX * MAX * 8);
        let a = b.ins().stack_addr(I32, ss, 0);
        let bm = b.ins().stack_addr(I32, ss, (MAX * MAX * 8) as i32);
        let c = b.ins().stack_addr(I32, ss, (2 * MAX * MAX * 8) as i32);
        let zero = b.ins().iconst(I32, 0);
        for_range(b, zero, n, |b, i| {
            for_range(b, zero, n, |b, j| {
                let row = b.ins().imul(i, n);
                let idx = b.ins().iadd(row, j);
                let s = b.ins().iadd(i, j);
                let d = b.ins().isub(i, j);
                let s = b.ins().fcvt_from_sint(F64, s);
                let d = b.ins().fcvt_from_sint(F64, d);
                let pa = elem(b, a, idx, 3);
                b.ins().store(flags(), s, pa, 0);
                let pb = elem(b, bm, idx, 3);
                b.ins().store(flags(), d, pb, 0);
            });
        });
        for_range(b, zero, n, |b, i| {
            for_range(b, zero, n, |b, j| {
                let fz = b.ins().f64const(0.0);
                let sum = var(b, F64, fz);
                for_range(b, zero, n, |b, k| {
                    let ri = b.ins().imul(i, n);
                    let ia = b.ins().iadd(ri, k);
                    let rk = b.ins().imul(k, n);
                    let ib = b.ins().iadd(rk, j);
                    let pa = elem(b, a, ia, 3);
                    let pb = elem(b, bm, ib, 3);
                    let x = b.ins().load(F64, flags(), pa, 0);
                    let y = b.ins().load(F64, flags(), pb, 0);
                    let m = b.ins().fmul(x, y);
                    let s = b.use_var(sum);
                    let s = b.ins().fadd(s, m);
                    b.def_var(sum, s);
                });
                let row = b.ins().imul(i, n);
                let idx = b.ins().iadd(row, j);
                let pc = elem(b, c, idx, 3);
                let s = b.use_var(sum);
                b.ins().store(flags(), s, pc, 0);
            });
        });
        let fz = b.ins().f64const(0.0);
        let total = var(b, F64, fz);
        let nn = b.ins().imul(n, n);
        for_range(b, zero, nn, |b, i| {
            let pc = elem(b, c, i, 3);
            let x = b.ins().load(F64, flags(), pc, 0);
            let t = b.use_var(total);
            let t = b.ins().fadd(t, x);
            b.def_var(total, t);
        });
        let t = b.use_var(total);
        let r = b.ins().fcvt_to_sint_sat(I64, t);
        b.ins().return_(&[r]);
    })
}

fn lcg(b: &mut FunctionBuilder, x: Variable) -> Value {
    let v = b.use_var(x);
    let m = b.ins().imul_imm_s(v, 1103515245);
    let n = b.ins().iadd_imm_s(m, 12345);
    b.def_var(x, n);
    b.ins().ushr_imm_u(n, 16)
}

fn crc32() -> Function {
    build("k_crc32", &[I32], &[I64], |b, p| {
        let n = p[0];
        let ss = slot(b, 65536);
        let base = b.ins().stack_addr(I32, ss, 0);
        let seed = b.ins().iconst(I32, 42);
        let x = var(b, I32, seed);
        let zero = b.ins().iconst(I32, 0);
        let len = b.ins().iconst(I32, 65536);
        for_range(b, zero, len, |b, i| {
            let r = lcg(b, x);
            let a = elem(b, base, i, 0);
            b.ins().istore8(flags(), r, a, 0);
        });
        let init = b.ins().iconst(I32, -1);
        let crc = var(b, I32, init);
        for_range(b, zero, n, |b, i| {
            let wrapped = b.ins().band_imm_u(i, 0xffff);
            let a = elem(b, base, wrapped, 0);
            let byte = b.ins().uload8(I32, flags(), a, 0);
            let c = b.use_var(crc);
            let c = b.ins().bxor(c, byte);
            b.def_var(crc, c);
            let eight = b.ins().iconst(I32, 8);
            for_range(b, zero, eight, |b, _| {
                let c = b.use_var(crc);
                let low = b.ins().band_imm_u(c, 1);
                let mask = b.ins().ineg(low);
                let poly = b.ins().band_imm_u(mask, 0xedb8_8320);
                let sh = b.ins().ushr_imm_u(c, 1);
                let c = b.ins().bxor(sh, poly);
                b.def_var(crc, c);
            });
        });
        let c = b.use_var(crc);
        let c = b.ins().bnot(c);
        let r = b.ins().uextend(I64, c);
        b.ins().return_(&[r]);
    })
}

fn sort() -> Function {
    build("k_sort", &[I32], &[I64], |b, p| {
        let n = p[0];
        let ss = slot(b, 4096 * 4);
        let base = b.ins().stack_addr(I32, ss, 0);
        let seed = b.ins().iconst(I32, 7);
        let x = var(b, I32, seed);
        let zero = b.ins().iconst(I32, 0);
        let one = b.ins().iconst(I32, 1);
        for_range(b, zero, n, |b, i| {
            let r = lcg(b, x);
            let a = elem(b, base, i, 2);
            b.ins().store(flags(), r, a, 0);
        });
        for_range(b, one, n, |b, i| {
            let pi = elem(b, base, i, 2);
            let key = b.ins().load(I32, flags(), pi, 0);
            let jm = b.ins().iadd_imm_s(i, -1);
            let j = var(b, I32, jm);
            // while j >= 0 && a[j] > key
            let (h, check, body, exit) = (
                b.create_block(),
                b.create_block(),
                b.create_block(),
                b.create_block(),
            );
            b.ins().jump(h, &[]);
            b.switch_to_block(h);
            let jv = b.use_var(j);
            let nonneg = b.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, jv, 0);
            b.ins().brif(nonneg, check, &[], exit, &[]);
            b.switch_to_block(check);
            let pj = elem(b, base, jv, 2);
            let aj = b.ins().load(I32, flags(), pj, 0);
            let gt = b.ins().icmp(IntCC::SignedGreaterThan, aj, key);
            b.ins().brif(gt, body, &[], exit, &[]);
            b.switch_to_block(body);
            b.ins().store(flags(), aj, pj, 4);
            let jn = b.ins().iadd_imm_s(jv, -1);
            b.def_var(j, jn);
            b.ins().jump(h, &[]);
            b.switch_to_block(exit);
            let jv = b.use_var(j);
            let j1 = b.ins().iadd_imm_s(jv, 1);
            let pj = elem(b, base, j1, 2);
            b.ins().store(flags(), key, pj, 0);
        });
        let z64 = b.ins().iconst(I64, 0);
        let sum = var(b, I64, z64);
        for_range(b, zero, n, |b, i| {
            let pi = elem(b, base, i, 2);
            let v = b.ins().load(I32, flags(), pi, 0);
            let v = b.ins().sextend(I64, v);
            let w = b.ins().iadd_imm_s(i, 1);
            let w = b.ins().uextend(I64, w);
            let t = b.ins().imul(v, w);
            let s = b.use_var(sum);
            let s = b.ins().iadd(s, t);
            b.def_var(sum, s);
        });
        let r = b.use_var(sum);
        b.ins().return_(&[r]);
    })
}

fn mandel() -> Function {
    build("k_mandel", &[I32], &[I64], |b, p| {
        let n = p[0];
        let z64 = b.ins().iconst(I64, 0);
        let count = var(b, I64, z64);
        let zero = b.ins().iconst(I32, 0);
        let nf = b.ins().fcvt_from_sint(F64, n);
        for_range(b, zero, n, |b, py| {
            for_range(b, zero, n, |b, px| {
                let pxf = b.ins().fcvt_from_sint(F64, px);
                let pyf = b.ins().fcvt_from_sint(F64, py);
                let three = b.ins().f64const(3.0);
                let two = b.ins().f64const(2.0);
                let onef = b.ins().f64const(1.0);
                let cr = b.ins().fmul(pxf, three);
                let cr = b.ins().fdiv(cr, nf);
                let cr = b.ins().fsub(cr, two);
                let ci = b.ins().fmul(pyf, two);
                let ci = b.ins().fdiv(ci, nf);
                let ci = b.ins().fsub(ci, onef);
                let f0 = b.ins().f64const(0.0);
                let zr = var(b, F64, f0);
                let zi = var(b, F64, f0);
                let it = var(b, I64, z64);
                let (h, check, body, exit) = (
                    b.create_block(),
                    b.create_block(),
                    b.create_block(),
                    b.create_block(),
                );
                b.ins().jump(h, &[]);
                b.switch_to_block(h);
                let itv = b.use_var(it);
                let more = b.ins().icmp_imm_s(IntCC::SignedLessThan, itv, 64);
                b.ins().brif(more, check, &[], exit, &[]);
                b.switch_to_block(check);
                let (r, i) = (b.use_var(zr), b.use_var(zi));
                let rr = b.ins().fmul(r, r);
                let ii = b.ins().fmul(i, i);
                let mag = b.ins().fadd(rr, ii);
                let four = b.ins().f64const(4.0);
                let inside = b.ins().fcmp(FloatCC::LessThanOrEqual, mag, four);
                b.ins().brif(inside, body, &[], exit, &[]);
                b.switch_to_block(body);
                let t = b.ins().fsub(rr, ii);
                let t = b.ins().fadd(t, cr);
                let ri = b.ins().fmul(r, i);
                let ri2 = b.ins().fadd(ri, ri);
                let ni = b.ins().fadd(ri2, ci);
                b.def_var(zr, t);
                b.def_var(zi, ni);
                let itn = b.ins().iadd_imm_s(itv, 1);
                b.def_var(it, itn);
                b.ins().jump(h, &[]);
                b.switch_to_block(exit);
                let c = b.use_var(count);
                let itv = b.use_var(it);
                let c = b.ins().iadd(c, itv);
                b.def_var(count, c);
            });
        });
        let r = b.use_var(count);
        b.ins().return_(&[r]);
    })
}

fn collatz() -> Function {
    build("k_collatz", &[I32], &[I64], |b, p| {
        let n = p[0];
        let z64 = b.ins().iconst(I64, 0);
        let best = var(b, I64, z64);
        let one = b.ins().iconst(I32, 1);
        for_range(b, one, n, |b, i| {
            let x0 = b.ins().uextend(I64, i);
            let x = var(b, I64, x0);
            let steps = var(b, I64, z64);
            while_loop(
                b,
                |b| {
                    let xv = b.use_var(x);
                    b.ins().icmp_imm_s(IntCC::NotEqual, xv, 1)
                },
                |b| {
                    let xv = b.use_var(x);
                    let odd = b.ins().band_imm_u(xv, 1);
                    if_else(
                        b,
                        odd,
                        |b| {
                            let t = b.ins().imul_imm_s(xv, 3);
                            let t = b.ins().iadd_imm_s(t, 1);
                            b.def_var(x, t);
                        },
                        |b| {
                            let t = b.ins().ushr_imm_u(xv, 1);
                            b.def_var(x, t);
                        },
                    );
                    let s = b.use_var(steps);
                    let s = b.ins().iadd_imm_s(s, 1);
                    b.def_var(steps, s);
                },
            );
            let s = b.use_var(steps);
            let bv = b.use_var(best);
            let better = b.ins().icmp(IntCC::SignedGreaterThan, s, bv);
            let nb = b.ins().select(better, s, bv);
            b.def_var(best, nb);
        });
        let r = b.use_var(best);
        b.ins().return_(&[r]);
    })
}

fn dispatch() -> Function {
    build("k_dispatch", &[I32], &[I64], |b, p| {
        let n = p[0];
        let one = b.ins().iconst(I64, 1);
        let acc = var(b, I64, one);
        let z32 = b.ins().iconst(I32, 0);
        let state = var(b, I32, z32);
        for_range(b, z32, n, |b, i| {
            let arms: Vec<_> = (0..5).map(|_| b.create_block()).collect();
            let merge = b.create_block();
            let s = b.use_var(state);
            let pool = &mut b.func.dfg.value_lists;
            let calls: Vec<BlockCall> = arms.iter().map(|&a| BlockCall::new(a, [], pool)).collect();
            let def = BlockCall::new(arms[0], [], pool);
            let jt = b.create_jump_table(JumpTableData::new(def, &calls));
            b.ins().br_table(s, jt);
            // Each arm updates acc and picks the next state.
            for (k, &arm) in arms.iter().enumerate() {
                b.switch_to_block(arm);
                let a = b.use_var(acc);
                let (na, ns) = match k {
                    0 => {
                        let t = b.ins().imul_imm_s(a, 3);
                        let i64v = b.ins().uextend(I64, i);
                        (b.ins().iadd(t, i64v), b.ins().iconst(I32, 1))
                    }
                    1 => {
                        let t = b.ins().ushr_imm_u(a, 7);
                        (b.ins().bxor(a, t), b.ins().iconst(I32, 2))
                    }
                    2 => {
                        let t = b.ins().iadd_imm_s(a, 0x9e37_79b9);
                        let s = b.ins().ireduce(I32, t);
                        (t, b.ins().band_imm_u(s, 3))
                    }
                    3 => (b.ins().rotl_imm_u(a, 5), b.ins().iconst(I32, 4)),
                    _ => {
                        let i64v = b.ins().uextend(I64, i);
                        (b.ins().isub(a, i64v), b.ins().iconst(I32, 0))
                    }
                };
                b.def_var(acc, na);
                b.def_var(state, ns);
                b.ins().jump(merge, &[]);
            }
            b.switch_to_block(merge);
        });
        let r = b.use_var(acc);
        b.ins().return_(&[r]);
    })
}

fn fib_rec_helper() -> Function {
    build("fib_rec", &[I32], &[I64], |b, p| {
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(I32));
        sig.returns.push(AbiParam::new(I64));
        let sig = b.import_signature(sig);
        let callee = b.import_function(cranelift_codegen::ir::ExtFuncData {
            name: cranelift_codegen::ir::ExternalName::testcase("fib_rec"),
            signature: sig,
            colocated: true,
            patchable: false,
        });
        let n = p[0];
        let small = b.ins().icmp_imm_s(IntCC::SignedLessThan, n, 2);
        let (base, rec) = (b.create_block(), b.create_block());
        b.ins().brif(small, base, &[], rec, &[]);
        b.switch_to_block(base);
        let r = b.ins().uextend(I64, n);
        b.ins().return_(&[r]);
        b.switch_to_block(rec);
        let n1 = b.ins().iadd_imm_s(n, -1);
        let n2 = b.ins().iadd_imm_s(n, -2);
        let c1 = b.ins().call(callee, &[n1]);
        let r1 = b.inst_results(c1)[0];
        let c2 = b.ins().call(callee, &[n2]);
        let r2 = b.inst_results(c2)[0];
        let r = b.ins().iadd(r1, r2);
        b.ins().return_(&[r]);
    })
}

fn fib_rec() -> Function {
    build("k_fib_rec", &[I32], &[I64], |b, p| {
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(I32));
        sig.returns.push(AbiParam::new(I64));
        let sig = b.import_signature(sig);
        let callee = b.import_function(cranelift_codegen::ir::ExtFuncData {
            name: cranelift_codegen::ir::ExternalName::testcase("fib_rec"),
            signature: sig,
            colocated: true,
            patchable: false,
        });
        let c = b.ins().call(callee, &[p[0]]);
        let r = b.inst_results(c)[0];
        b.ins().return_(&[r]);
    })
}
