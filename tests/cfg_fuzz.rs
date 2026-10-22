//! Differential testing on arbitrary control-flow graphs: unlike `fuzz.rs`,
//! whose functions come from structured statements, these are built block by
//! block, so they contain irreducible loops, branches whose two arms go to the
//! same block with different arguments, and block arguments that permute the
//! target's parameters (swaps, rotations). Results from Cranelift's
//! interpreter and from wasmtime running this backend's output must agree.
//!
//! Every block `H_i(fuel, p...)` checks a fuel counter before running its body
//! `B_i`, and returns its parameters once fuel runs out, so every execution
//! terminates.

mod harness;

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{AtomicRmwOp, Endianness};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee32, Ieee64};
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, BlockCall, ExtFuncData, ExternalName, FuncRef, Function, SigRef, InstBuilder, JumpTableData,
    MemFlagsData, Signature, StackSlot, StackSlotData, StackSlotKind, Type, UserFuncName, Value, types,
};
use cranelift_codegen::isa::CallConv;
use clif_to_wasm::{Config, Optimizations};
use proptest::prelude::*;

/// Scalar types first (the first five are the integers), then vectors,
/// including ones narrower than 128 bits, which live in the low bytes of a
/// `v128` whose other bytes are unspecified.
const TYS: [Type; 16] = [
    types::I8,
    types::I16,
    types::I32,
    types::I64,
    types::I128,
    types::F32,
    types::F64,
    types::I32X4,
    types::I8X16,
    types::I64X2,
    types::F32X4,
    types::I32X2,
    types::I8X8,
    types::I16X4,
    types::F32X2,
    types::I16X2,
];
const NUM_SCALARS: usize = 7;

const INT_CCS: [IntCC; 10] = [
    IntCC::Equal,
    IntCC::NotEqual,
    IntCC::SignedLessThan,
    IntCC::SignedGreaterThanOrEqual,
    IntCC::SignedGreaterThan,
    IntCC::SignedLessThanOrEqual,
    IntCC::UnsignedLessThan,
    IntCC::UnsignedGreaterThanOrEqual,
    IntCC::UnsignedGreaterThan,
    IntCC::UnsignedLessThanOrEqual,
];
const SLOT_SIZE: u32 = 48;

#[derive(Clone, Debug)]
enum Op {
    Const(usize, i64),
    Bin(u8, usize, usize),
    Icmp(u8, usize, usize),
    Select(usize, usize, usize),
    Convert(usize, usize),
    Div(bool, usize, usize),
    /// A store of a value at an offset in the slot, through `stack_store`
    /// or through `stack_addr` and a `notrap` store.
    Store(usize, u8, bool),
    /// Call `%g` (see `CALLEE`) with the slot's address.
    Call(usize, usize),
    /// An atomic operation on an integer type, at an aligned offset in the
    /// slot: read-modify-write (`AtomicRmwOp` index), compare and swap, load
    /// (of `TYS[..5]`) or store. The type is that of the first operand.
    AtomicRmw(u8, usize, u8),
    AtomicCas(usize, usize, u8),
    AtomicLoad(usize, u8),
    AtomicStore(usize, u8),
    /// Vector operations: a lane-wise comparison, a reduction to a scalar
    /// (`vall_true`, `vany_true`, `vhigh_bits`, `extractlane`), a `splat` to
    /// `TYS[t]`, widening, narrowing, `iadd_pairwise`, and a bitcast from a
    /// vector to an integer of its size.
    VCmp(u8, usize, usize),
    VReduce(u8, usize),
    VSplat(usize, usize),
    VWiden(u8, usize),
    VNarrow(u8, usize, usize),
    VPairwise(usize, usize),
    VBitcast(usize),
    /// `insertlane` of a scalar (third operand) into lane `k` of a vector.
    VInsert(u8, usize, usize),
    /// Call `%w` (see `WRITER`), which writes to the slot and returns one
    /// value, so that the call can fold into an operand tree.
    CallWrite(usize, u8),
    /// Call `%s` (see `BYVAL_CALLEE`), passing 16 bytes of the slot by
    /// value, directly or through `call_indirect`.
    CallByval(usize, u8, bool),
    /// Call one of `NARROW_CALLEES` (index, arguments), directly or through
    /// `call_indirect`.
    CallNarrow(usize, usize, usize, bool),
    Load(usize, u8, bool),
}

#[derive(Clone, Debug)]
struct Edge {
    target: usize,
    args: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Term {
    Jump(Edge),
    Brif(usize, Edge, Edge),
    BrTable(usize, Edge, Vec<Edge>),
    Return(Vec<usize>),
}

#[derive(Clone, Debug)]
struct BlockSpec {
    ops: Vec<Op>,
    term: Term,
}

#[derive(Clone, Debug)]
struct Spec {
    tys: Vec<Type>,
    blocks: Vec<BlockSpec>,
}

fn edge(nblocks: usize, nparams: usize) -> impl Strategy<Value = Edge> {
    // Small indices mostly pick the block's own parameters, so the arguments
    // are often a permutation of them.
    (
        0..nblocks,
        prop::collection::vec(prop_oneof![3 => 0usize..4, 1 => 0usize..32], nparams),
    )
        .prop_map(|(target, args)| Edge { target, args })
}

/// The kinds of `Op`, with their weights.
fn op_kinds() -> Vec<(u32, BoxedStrategy<Op>)> {
    let i = || 0usize..32;
    vec![
        (1, (0..TYS.len(), any::<i64>()).prop_map(|(t, c)| Op::Const(t, c)).boxed()),
        (4, (0u8..10, i(), i()).prop_map(|(k, a, b)| Op::Bin(k, a, b)).boxed()),
        (1, (0u8..10, i(), i()).prop_map(|(k, a, b)| Op::Icmp(k, a, b)).boxed()),
        (1, (i(), i(), i()).prop_map(|(c, a, b)| Op::Select(c, a, b)).boxed()),
        (1, (i(), 0..TYS.len()).prop_map(|(a, t)| Op::Convert(a, t)).boxed()),
        (1, (any::<bool>(), i(), i()).prop_map(|(s, a, b)| Op::Div(s, a, b)).boxed()),
        (2, (i(), 0u8..4, any::<bool>()).prop_map(|(a, o, d)| Op::Store(a, o, d)).boxed()),
        (2, (0..TYS.len(), 0u8..4, any::<bool>()).prop_map(|(t, o, d)| Op::Load(t, o, d)).boxed()),
        (1, (i(), i()).prop_map(|(a, b)| Op::Call(a, b)).boxed()),
        (2, (0u8..11, i(), any::<u8>()).prop_map(|(k, a, o)| Op::AtomicRmw(k, a, o)).boxed()),
        (1, (i(), i(), any::<u8>()).prop_map(|(a, b, o)| Op::AtomicCas(a, b, o)).boxed()),
        (1, (0usize..5, any::<u8>()).prop_map(|(t, o)| Op::AtomicLoad(t, o)).boxed()),
        (1, (i(), any::<u8>()).prop_map(|(a, o)| Op::AtomicStore(a, o)).boxed()),
        (1, (0u8..10, i(), i()).prop_map(|(k, a, b)| Op::VCmp(k, a, b)).boxed()),
        (2, (any::<u8>(), i()).prop_map(|(k, a)| Op::VReduce(k, a)).boxed()),
        (1, (NUM_SCALARS..TYS.len(), i()).prop_map(|(t, a)| Op::VSplat(t, a)).boxed()),
        (1, (0u8..4, i()).prop_map(|(k, a)| Op::VWiden(k, a)).boxed()),
        (1, (0u8..3, i(), i()).prop_map(|(k, a, b)| Op::VNarrow(k, a, b)).boxed()),
        (1, (i(), i()).prop_map(|(a, b)| Op::VPairwise(a, b)).boxed()),
        (1, i().prop_map(Op::VBitcast).boxed()),
        (2, (i(), any::<u8>(), any::<bool>()).prop_map(|(a, o, ind)| Op::CallByval(a, o, ind)).boxed()),
        (1, (i(), any::<u8>()).prop_map(|(a, o)| Op::CallWrite(a, o)).boxed()),
        (2, (any::<u8>(), i(), i()).prop_map(|(k, a, b)| Op::VInsert(k, a, b)).boxed()),
        (
            1,
            (0..NARROW_CALLEES.len(), i(), i(), any::<bool>())
                .prop_map(|(k, a, b, ind)| Op::CallNarrow(k, a, b, ind))
                .boxed(),
        ),
    ]
}

/// Ops drawn from the kinds that `enabled` selects (swarm testing: each case
/// leaves some kinds out, so the others are more frequent and interact more).
fn op(enabled: &[bool]) -> impl Strategy<Value = Op> + use<> {
    let kinds: Vec<(u32, BoxedStrategy<Op>)> = op_kinds()
        .into_iter()
        .zip(enabled)
        .filter(|&(_, &e)| e)
        .map(|(k, _)| k)
        .collect();
    prop::strategy::Union::new_weighted(kinds)
}

fn spec() -> impl Strategy<Value = Spec> {
    (
        prop::collection::vec(
            prop_oneof![
                3 => prop::sample::select(&TYS[..NUM_SCALARS]),
                2 => prop::sample::select(&TYS[NUM_SCALARS..]),
            ],
            1..=4,
        ),
        1usize..=6,
        // At least the first kind (constants) stays enabled.
        prop::collection::vec(any::<bool>(), op_kinds().len() - 1),
    )
        .prop_flat_map(|(tys, nblocks, enabled)| {
            let enabled: Vec<bool> = std::iter::once(true).chain(enabled).collect();
            let n = tys.len();
            let term = prop_oneof![
                3 => edge(nblocks, n).prop_map(Term::Jump),
                4 => (0usize..32, edge(nblocks, n), edge(nblocks, n))
                    .prop_map(|(c, a, b)| Term::Brif(c, a, b)),
                1 => (0usize..32, edge(nblocks, n), prop::collection::vec(edge(nblocks, n), 0..4))
                    .prop_map(|(c, d, t)| Term::BrTable(c, d, t)),
                1 => prop::collection::vec(0usize..32, n).prop_map(Term::Return),
            ];
            let block = (prop::collection::vec(op(&enabled), 0..6), term)
                .prop_map(|(ops, term)| BlockSpec { ops, term });
            (Just(tys), prop::collection::vec(block, nblocks))
                .prop_map(|(tys, blocks)| Spec { tys, blocks })
        })
}

/// Values available in a block body.
struct Pool(Vec<Value>);

impl Pool {
    fn pick(&self, func: &Function, ty: Option<Type>, i: usize) -> Option<Value> {
        self.pick_where(func, |t| ty.is_none_or(|ty| t == ty), i)
    }

    fn pick_where(&self, func: &Function, f: impl Fn(Type) -> bool, i: usize) -> Option<Value> {
        let c: Vec<Value> = self
            .0
            .iter()
            .copied()
            .filter(|&v| f(func.dfg.value_type(v)))
            .collect();
        // Half the indices pick one of the three most recent values, so that
        // values often feed the next operations and long chains form.
        (!c.is_empty()).then(|| {
            if i >= 16 {
                c[c.len() - 1 - ((i - 16) / 6).min(c.len() - 1)]
            } else {
                c[i % c.len()]
            }
        })
    }
}

fn is_int(t: Type) -> bool {
    t.is_int() && !t.is_vector()
}

fn is_narrow_int(t: Type) -> bool {
    is_int(t) && t.bits() <= 64
}

/// A digest of the slot's contents.
fn digest(pos: &mut FuncCursor, slot: StackSlot) -> Value {
    let mut d = pos.ins().iconst(types::I64, 0);
    for off in (0..SLOT_SIZE).step_by(8) {
        let w = pos.ins().stack_load(types::I64, types::I64, slot, off as i32);
        let r = pos.ins().rotl_imm_u(d, 7);
        d = pos.ins().bxor(r, w);
    }
    d
}

/// The address of a naturally aligned `ty` in the slot, chosen by `o`.
fn atomic_addr(pos: &mut FuncCursor, slot: StackSlot, ty: Type, o: u8) -> Value {
    let n = SLOT_SIZE / ty.bytes();
    let off = (o as u32 % n) * ty.bytes();
    pos.ins().stack_addr(types::I64, slot, off as i32)
}

/// Lower `op` at `pos`, returning its result if it has one. Ops whose
/// operands have no suitable value are skipped.
fn lower_op(
    pos: &mut FuncCursor,
    pool: &mut Pool,
    slot: StackSlot,
    g: FuncRef,
    narrow: &[(FuncRef, SigRef)],
    byval: (FuncRef, SigRef),
    writer: FuncRef,
    op: &Op,
) -> Option<Value> {
    let ty_of = |pos: &FuncCursor, v: Value| pos.func.dfg.value_type(v);
    Some(match *op {
        Op::Const(t, c) => {
            let ty = TYS[t];
            match ty {
                types::I128 => {
                    let v = pos.ins().iconst(types::I64, c);
                    if c & 1 == 0 {
                        pos.ins().uextend(ty, v)
                    } else {
                        pos.ins().sextend(ty, v)
                    }
                }
                types::F32 => pos.ins().f32const(Ieee32::with_bits(c as u32)),
                types::F64 => pos.ins().f64const(Ieee64::with_bits(c as u64)),
                _ if ty.is_vector() => {
                    let lane = ty.lane_type();
                    let v = match lane {
                        types::F32 => pos.ins().f32const(Ieee32::with_bits(c as u32)),
                        types::F64 => pos.ins().f64const(Ieee64::with_bits(c as u64)),
                        _ => pos.ins().iconst(lane, c & lane.bounds(false).1 as i64),
                    };
                    pos.ins().splat(ty, v)
                }
                _ => pos.ins().iconst(ty, c & ty.bounds(false).1 as i64),
            }
        }
        Op::Bin(k, a, b) => {
            let a = pool.pick(pos.func, None, a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            let ins = pos.ins();
            if ty.is_vector() && ty.lane_type().is_float() {
                match k % 6 {
                    0 => ins.fadd(a, b),
                    1 => ins.fsub(a, b),
                    2 => ins.fmul(a, b),
                    3 => ins.fdiv(a, b),
                    4 => ins.fmin(a, b),
                    _ => ins.fmax(a, b),
                }
            } else if ty.is_vector() {
                match k % 8 {
                    0 => ins.iadd(a, b),
                    1 => ins.isub(a, b),
                    2 => ins.imul(a, b),
                    3 => ins.band(a, b),
                    4 => ins.bor(a, b),
                    5 => ins.bxor(a, b),
                    6 => ins.umin(a, b),
                    _ => ins.smax(a, b),
                }
            } else if ty.is_float() {
                match k % 6 {
                    0 => ins.fadd(a, b),
                    1 => ins.fsub(a, b),
                    2 => ins.fmul(a, b),
                    3 => ins.fdiv(a, b),
                    4 => ins.fmin(a, b),
                    _ => ins.fmax(a, b),
                }
            } else {
                match k {
                    0 => ins.iadd(a, b),
                    1 => ins.isub(a, b),
                    2 => ins.imul(a, b),
                    3 => ins.band(a, b),
                    4 => ins.bor(a, b),
                    5 => ins.bxor(a, b),
                    6 => ins.rotl(a, b),
                    7 => ins.ishl(a, b),
                    8 => ins.sshr(a, b),
                    _ => ins.ushr(a, b),
                }
            }
        }
        Op::Icmp(k, a, b) => {
            let a = pool.pick_where(pos.func, |t| !t.is_vector(), a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            if ty.is_float() {
                let cc = [
                    FloatCC::Equal,
                    FloatCC::NotEqual,
                    FloatCC::LessThan,
                    FloatCC::LessThanOrEqual,
                    FloatCC::GreaterThan,
                    FloatCC::Unordered,
                ][k as usize % 6];
                pos.ins().fcmp(cc, a, b)
            } else {
                pos.ins().icmp(INT_CCS[k as usize], a, b)
            }
        }
        Op::Select(c, a, b) => {
            let c = pool.pick_where(pos.func, is_narrow_int, c)?;
            let a = pool.pick(pos.func, None, a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            pos.ins().select(c, a, b)
        }
        Op::Convert(a, t) => {
            let a = pool.pick(pos.func, None, a)?;
            let from = ty_of(pos, a);
            let to = TYS[t];
            let ins = pos.ins();
            match (from, to) {
                (f, t) if is_int(f) && is_int(t) && t.bits() > f.bits() => {
                    if a.as_u32() % 2 == 0 {
                        ins.uextend(t, a)
                    } else {
                        ins.sextend(t, a)
                    }
                }
                (f, t) if is_int(f) && is_int(t) && t.bits() < f.bits() => ins.ireduce(t, a),
                (types::I32, types::F32)
                | (types::F32, types::I32)
                | (types::I64, types::F64)
                | (types::F64, types::I64) => ins.bitcast(to, MemFlagsData::new(), a),
                (types::I32 | types::I64, types::F32 | types::F64) => {
                    if a.as_u32() % 2 == 0 {
                        ins.fcvt_from_uint(to, a)
                    } else {
                        ins.fcvt_from_sint(to, a)
                    }
                }
                (types::F32 | types::F64, types::I32 | types::I64) => match a.as_u32() % 3 {
                    0 => ins.fcvt_to_sint_sat(to, a),
                    1 => ins.fcvt_to_uint_sat(to, a),
                    _ => ins.fcvt_to_sint(to, a),
                },
                (types::F32, types::F64) => ins.fpromote(to, a),
                (types::F64, types::F32) => ins.fdemote(to, a),
                (types::I32, types::I32X4) => ins.splat(to, a),
                (types::I32X4, _) => ins.extractlane(a, (t % 4) as u8),
                _ => return None,
            }
        }
        Op::Div(signed, a, b) => {
            let a = pool.pick_where(pos.func, is_int, a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            if signed {
                pos.ins().sdiv(a, b)
            } else {
                pos.ins().urem(a, b)
            }
        }
        Op::Store(a, o, direct) => {
            let a = pool.pick(pos.func, None, a)?;
            if direct {
                let p = pos.ins().stack_addr(types::I64, slot, 0);
                pos.ins().store(MemFlagsData::new().with_notrap(), a, p, o as i32 * 8);
            } else {
                pos.ins().stack_store(types::I64, a, slot, o as i32 * 8);
            }
            return None;
        }
        Op::Load(t, o, direct) => {
            if direct {
                let p = pos.ins().stack_addr(types::I64, slot, 0);
                pos.ins().load(TYS[t], MemFlagsData::new().with_notrap(), p, o as i32 * 8)
            } else {
                pos.ins().stack_load(types::I64, TYS[t], slot, o as i32 * 8)
            }
        }
        Op::AtomicRmw(k, a, o) => {
            let a = pool.pick_where(pos.func, is_int, a)?;
            let ty = ty_of(pos, a);
            let p = atomic_addr(pos, slot, ty, o);
            let op = [
                AtomicRmwOp::Add,
                AtomicRmwOp::Sub,
                AtomicRmwOp::And,
                AtomicRmwOp::Nand,
                AtomicRmwOp::Or,
                AtomicRmwOp::Xor,
                AtomicRmwOp::Xchg,
                AtomicRmwOp::Umin,
                AtomicRmwOp::Umax,
                AtomicRmwOp::Smin,
                AtomicRmwOp::Smax,
            ][k as usize];
            pos.ins().atomic_rmw(ty, MemFlagsData::new(), op, p, a)
        }
        Op::AtomicCas(a, b, o) => {
            let a = pool.pick_where(pos.func, is_int, a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            let p = atomic_addr(pos, slot, ty, o);
            pos.ins().atomic_cas(MemFlagsData::new(), p, a, b)
        }
        Op::AtomicLoad(t, o) => {
            let ty = TYS[t];
            let p = atomic_addr(pos, slot, ty, o);
            pos.ins().atomic_load(ty, MemFlagsData::new(), p)
        }
        Op::AtomicStore(a, o) => {
            let a = pool.pick_where(pos.func, is_int, a)?;
            let ty = ty_of(pos, a);
            let p = atomic_addr(pos, slot, ty, o);
            pos.ins().atomic_store(MemFlagsData::new(), a, p);
            return None;
        }
        Op::VCmp(k, a, b) => {
            let a = pool.pick_where(pos.func, |t| t.is_vector(), a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            if ty.lane_type().is_float() {
                let cc = [
                    FloatCC::Equal,
                    FloatCC::NotEqual,
                    FloatCC::LessThan,
                    FloatCC::LessThanOrEqual,
                    FloatCC::GreaterThan,
                    FloatCC::Unordered,
                ][k as usize % 6];
                pos.ins().fcmp(cc, a, b)
            } else {
                pos.ins().icmp(INT_CCS[k as usize], a, b)
            }
        }
        Op::VReduce(k, a) => {
            let a = pool.pick_where(pos.func, |t| t.is_vector(), a)?;
            let ty = ty_of(pos, a);
            match k % 4 {
                0 => pos.ins().vall_true(a),
                1 => pos.ins().vany_true(a),
                2 if !ty.lane_type().is_float() => {
                    let r = if k & 4 == 0 { types::I32 } else { types::I64 };
                    pos.ins().vhigh_bits(r, a)
                }
                _ => pos.ins().extractlane(a, (k as u32 / 4 % ty.lane_count()) as u8),
            }
        }
        Op::VSplat(t, a) => {
            let ty = TYS[t];
            let a = pool.pick(pos.func, Some(ty.lane_type()), a)?;
            pos.ins().splat(ty, a)
        }
        Op::VWiden(k, a) => {
            let a = pool.pick_where(
                pos.func,
                |t| t.is_vector() && t.lane_type().is_int() && t.lane_bits() < 64 && t.lane_count() >= 4,
                a,
            )?;
            let ins = pos.ins();
            match k {
                0 => ins.swiden_low(a),
                1 => ins.swiden_high(a),
                2 => ins.uwiden_low(a),
                _ => ins.uwiden_high(a),
            }
        }
        Op::VNarrow(k, a, b) => {
            let a = pool.pick_where(
                pos.func,
                |t| t.is_vector() && t.lane_type().is_int() && t.lane_bits() >= 16,
                a,
            )?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            let ins = pos.ins();
            match k {
                0 => ins.snarrow(a, b),
                1 => ins.unarrow(a, b),
                _ => ins.uunarrow(a, b),
            }
        }
        Op::VPairwise(a, b) => {
            let a = pool.pick_where(
                pos.func,
                |t| t.is_vector() && t.lane_type().is_int() && t.lane_bits() <= 32,
                a,
            )?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty), b)?;
            pos.ins().iadd_pairwise(a, b)
        }
        Op::VInsert(k, a, b) => {
            let a = pool.pick_where(pos.func, |t| t.is_vector(), a)?;
            let ty = ty_of(pos, a);
            let b = pool.pick(pos.func, Some(ty.lane_type()), b)?;
            pos.ins().insertlane(a, b, (k as u32 % ty.lane_count()) as u8)
        }
        Op::VBitcast(a) => {
            // Only from vectors of at most 64 bits to integers:
            // cranelift-interpreter returns a lane-typed scalar for a bitcast
            // from an integer to a vector, and panics on 128-bit ones.
            let a = pool.pick_where(pos.func, |t| t.is_vector() && t.bits() <= 64, a)?;
            let to = Type::int(ty_of(pos, a).bits() as u16).unwrap();
            let flags = MemFlagsData::new().with_endianness(Endianness::Little);
            pos.ins().bitcast(to, flags, a)
        }
        Op::CallWrite(a, o) => {
            let a = pool.pick(pos.func, Some(types::I64), a)?;
            let off = (o as u32 % (SLOT_SIZE / 8)) * 8;
            let p = pos.ins().stack_addr(types::I64, slot, off as i32);
            let call = pos.ins().call(writer, &[p, a]);
            pos.func.dfg.first_result(call)
        }
        Op::CallByval(a, o, indirect) => {
            let a = pool.pick(pos.func, Some(types::I64), a)?;
            if let cranelift_codegen::ir::ValueDef::Result(d, _) = pos.func.dfg.value_def(a) && pos.func.dfg.insts[d].opcode().is_call() { stat("nested-call-arg"); }
            let off = (o as u32 % (SLOT_SIZE / 8 - 1)) * 8;
            let p = pos.ins().stack_addr(types::I64, slot, off as i32);
            let out_off = (o as u32 / 8 % (SLOT_SIZE / 8)) * 8;
            let out = pos.ins().stack_addr(types::I64, slot, out_off as i32);
            let (f, sig) = byval;
            let call = if indirect {
                let addr = pos.ins().func_addr(types::I64, f);
                pos.ins().call_indirect(sig, addr, &[p, out, a])
            } else {
                pos.ins().call(f, &[p, out, a])
            };
            pos.func.dfg.first_result(call)
        }
        Op::CallNarrow(k, a, b, indirect) => {
            let a = pool.pick(pos.func, Some(types::I8), a)?;
            let b = pool.pick(pos.func, Some(types::I16), b)?;
            let (f, sig) = narrow[k];
            let call = if indirect {
                let addr = pos.ins().func_addr(types::I64, f);
                pos.ins().call_indirect(sig, addr, &[a, b])
            } else {
                pos.ins().call(f, &[a, b])
            };
            let rs = pos.func.dfg.inst_results(call).to_vec();
            let (&last, rest) = rs.split_last().unwrap();
            pool.0.extend(rest);
            last
        }
        Op::Call(a, b) => {
            let a = pool.pick(pos.func, Some(types::I32), a)?;
            let b = pool.pick(pos.func, Some(types::I64), b)?;
            let p = pos.ins().stack_addr(types::I64, slot, 0);
            let call = pos.ins().call(g, &[p, a, b]);
            let rs = pos.func.dfg.inst_results(call);
            let (r0, r1) = (rs[0], rs[1]);
            pool.0.push(r0);
            r1
        }
    })
}

/// The successors of each block of `spec`.
fn succs(spec: &Spec) -> Vec<Vec<usize>> {
    spec.blocks
        .iter()
        .map(|b| match &b.term {
            Term::Jump(e) => vec![e.target],
            Term::Brif(_, x, y) => vec![x.target, y.target],
            Term::BrTable(_, d, ts) => std::iter::once(d).chain(ts).map(|e| e.target).collect(),
            Term::Return(_) => vec![],
        })
        .collect()
}

/// The dominators of each block of `spec` (`None` if unreachable), by the
/// iterative data-flow definition.
fn dominators(spec: &Spec) -> Vec<Option<Vec<usize>>> {
    use std::collections::BTreeSet;
    let n = spec.blocks.len();
    let succs = succs(spec);
    let mut reachable = vec![false; n];
    let mut stack = vec![0];
    while let Some(b) = stack.pop() {
        if !std::mem::replace(&mut reachable[b], true) {
            stack.extend(&succs[b]);
        }
    }
    let all: BTreeSet<usize> = (0..n).filter(|&b| reachable[b]).collect();
    let mut dom: Vec<BTreeSet<usize>> = (0..n)
        .map(|b| if b == 0 { [0].into() } else { all.clone() })
        .collect();
    loop {
        let mut changed = false;
        for b in 1..n {
            if !reachable[b] {
                continue;
            }
            let mut d: Option<BTreeSet<usize>> = None;
            for p in (0..n).filter(|&p| reachable[p] && succs[p].contains(&b)) {
                d = Some(match d {
                    None => dom[p].clone(),
                    Some(d) => d.intersection(&dom[p]).copied().collect(),
                });
            }
            let mut d = d.unwrap();
            d.insert(b);
            if d != dom[b] {
                dom[b] = d;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    (0..n)
        .map(|b| reachable[b].then(|| dom[b].iter().copied().collect()))
        .collect()
}

fn build(spec: &Spec) -> Function {
    let tys = &spec.tys;
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I32));
    sig.params.extend(tys.iter().map(|&t| AbiParam::new(t)));
    sig.returns.extend(tys.iter().map(|&t| AbiParam::new(t)));
    // A digest of the slot, so that every store is observable.
    sig.returns.push(AbiParam::new(types::I64));
    let mut func = Function::with_name_signature(UserFuncName::testcase("f"), sig);
    let slot = func.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        SLOT_SIZE,
        4,
    ));

    let mut gsig = Signature::new(CallConv::SystemV);
    gsig.params
        .extend([types::I64, types::I32, types::I64].map(AbiParam::new));
    gsig.returns.extend([types::I32, types::I64].map(AbiParam::new));
    let gsig = func.import_signature(gsig);
    let g = func.import_function(ExtFuncData {
        name: ExternalName::testcase("g"),
        signature: gsig,
        colocated: false,
        patchable: false,
    });

    let narrow: Vec<(FuncRef, SigRef)> = NARROW_CALLEES
        .iter()
        .map(|text| {
            let callee = cranelift_reader::parse_functions(text).unwrap().remove(0);
            let sig = func.import_signature(callee.signature.clone());
            let f = func.import_function(ExtFuncData {
                name: ExternalName::testcase(harness::func_name(&callee)),
                signature: sig,
                colocated: false,
                patchable: false,
            });
            (f, sig)
        })
        .collect();

    let byval = {
        let callee = cranelift_reader::parse_functions(BYVAL_CALLEE).unwrap().remove(0);
        let sig = func.import_signature(callee.signature.clone());
        let f = func.import_function(ExtFuncData {
            name: ExternalName::testcase("s"),
            signature: sig,
            colocated: false,
            patchable: false,
        });
        (f, sig)
    };

    let writer = {
        let callee = cranelift_reader::parse_functions(WRITER).unwrap().remove(0);
        let sig = func.import_signature(callee.signature.clone());
        func.import_function(ExtFuncData {
            name: ExternalName::testcase("w"),
            signature: sig,
            colocated: false,
            patchable: false,
        })
    };

    let entry = func.dfg.make_block();
    let heads: Vec<Block> = spec.blocks.iter().map(|_| func.dfg.make_block()).collect();
    let bodies: Vec<Block> = spec.blocks.iter().map(|_| func.dfg.make_block()).collect();
    let exits: Vec<Block> = spec.blocks.iter().map(|_| func.dfg.make_block()).collect();

    let mut pos = FuncCursor::new(&mut func);
    pos.insert_block(entry);
    let fuel = pos.func.dfg.append_block_param(entry, types::I32);
    let params: Vec<Value> = tys
        .iter()
        .map(|&t| pos.func.dfg.append_block_param(entry, t))
        .collect();
    // A different word at each offset, so that reading the wrong part of
    // the slot shows.
    for off in (0..SLOT_SIZE).step_by(8) {
        let w = pos.ins().iconst(types::I64, (off as i64 + 1) * 0x0101_0101_0101);
        pos.ins().stack_store(types::I64, w, slot, off as i32);
    }
    let fuel = pos.ins().band_imm_u(fuel, 63);
    let mut args = vec![BlockArg::Value(fuel)];
    args.extend(params.iter().map(|&v| BlockArg::Value(v)));
    pos.ins().jump(heads[0], &args);

    // Build dominators first, so that a body can use the values of the
    // bodies (and the entry block) that dominate it.
    let dom = dominators(spec);
    let mut order: Vec<usize> = (0..spec.blocks.len()).collect();
    order.sort_by_key(|&i| dom[i].as_ref().map_or(usize::MAX, |d| d.len()));
    let mut defs: Vec<Vec<Value>> = vec![Vec::new(); spec.blocks.len()];

    for i in order {
        let b = &spec.blocks[i];
        // H_i: the fuel check.
        pos.insert_block(heads[i]);
        let fuel = pos.func.dfg.append_block_param(heads[i], types::I32);
        let ps: Vec<Value> = tys
            .iter()
            .map(|&t| pos.func.dfg.append_block_param(heads[i], t))
            .collect();
        let more = pos.ins().icmp_imm_s(IntCC::SignedGreaterThan, fuel, 0);
        pos.ins().brif(more, bodies[i], &[], exits[i], &[]);
        pos.insert_block(exits[i]);
        let mut rs = ps.clone();
        rs.push(digest(&mut pos, slot));
        pos.ins().return_(&rs);

        // B_i: the body.
        pos.insert_block(bodies[i]);
        let fuel = pos.ins().iadd_imm_s(fuel, -1);
        let mut pool = Pool(ps.clone());
        if let Some(d) = &dom[i] {
            pool.0.extend(&params);
            for &d in d.iter().filter(|&&d| d != i) {
                pool.0.extend(&defs[d]);
            }
        }
        let first_def = pool.0.len();
        for op in &b.ops {
            if let Some(v) = lower_op(&mut pos, &mut pool, slot, g, &narrow, byval, writer, op) {
                pool.0.push(v);
            }
        }
        defs[i] = pool.0[first_def..].to_vec();

        let call = |pos: &mut FuncCursor, e: &Edge| -> BlockCall {
            let mut args = vec![BlockArg::Value(fuel)];
            for (j, &t) in tys.iter().enumerate() {
                args.push(BlockArg::Value(pool.pick(pos.func, Some(t), e.args[j]).unwrap()));
            }
            pos.func.dfg.block_call(heads[e.target], &args)
        };
        match &b.term {
            Term::Jump(e) => {
                let c = call(&mut pos, e);
                let args: Vec<BlockArg> = c.args(&pos.func.dfg.value_lists).collect();
                pos.ins().jump(heads[e.target], &args);
            }
            Term::Brif(c, x, y) => {
                let c = pool.pick_where(pos.func, is_int, *c).unwrap_or(fuel);
                let (cx, cy) = (call(&mut pos, x), call(&mut pos, y));
                let ax: Vec<BlockArg> = cx.args(&pos.func.dfg.value_lists).collect();
                let ay: Vec<BlockArg> = cy.args(&pos.func.dfg.value_lists).collect();
                pos.ins().brif(c, heads[x.target], &ax, heads[y.target], &ay);
            }
            Term::BrTable(c, d, ts) => {
                let mut c = pool.pick_where(pos.func, is_int, *c).unwrap_or(fuel);
                if pos.func.dfg.value_type(c).bits() > 32 {
                    c = pos.ins().ireduce(types::I32, c);
                } else if pos.func.dfg.value_type(c) != types::I32 {
                    c = pos.ins().uextend(types::I32, c);
                }
                let d = call(&mut pos, d);
                let ts: Vec<BlockCall> = ts.iter().map(|e| call(&mut pos, e)).collect();
                let jt = pos.func.dfg.jump_tables.push(JumpTableData::new(d, &ts));
                pos.ins().br_table(c, jt);
            }
            Term::Return(rs) => {
                let rs: Vec<Value> = tys
                    .iter()
                    .zip(rs)
                    .map(|(&t, &r)| pool.pick(pos.func, Some(t), r).unwrap())
                    .collect();
                let mut rs = rs;
                rs.push(digest(&mut pos, slot));
                pos.ins().return_(&rs);
            }
        }
    }
    func
}

/// The callee of `Op::Call`: it reads and writes the caller's slot, so calls
/// must stay ordered with the caller's loads and stores.
const CALLEE: &str = "
function %g(i64, i32, i64) -> i32, i64 system_v {
block0(v0: i64, v1: i32, v2: i64):
    v3 = load.i64 v0+8
    store v1, v0+4
    v4 = iadd v3, v2
    store v4, v0+16
    v5 = load.i32 v0
    return v5, v4
}
";

/// The callee of `Op::CallWrite`.
const WRITER: &str = "
function %w(i64, i64) -> i64 system_v {
block0(v0: i64, v1: i64):
    v2 = load.i64 v0
    v3 = bxor v2, v1
    store v3, v0
    return v3
}
";

/// The callee of `Op::CallByval`: it reads its copy of the struct, and
/// stores its result through the second argument, so that the call shows in
/// the slot's digest even when its result is unused. It does not write to
/// the copy, because cranelift-interpreter passes the caller's pointer
/// instead of a copy (`tests/clif/byval.clif` checks the copy); for the
/// same reason it reads the struct before storing.
const BYVAL_CALLEE: &str = "
function %s(i64 sarg(16), i64, i64) -> i64 system_v {
block0(v0: i64, v1: i64, v6: i64):
    v2 = load.i64 v0
    v3 = load.i64 v0+8
    v4 = rotl v2, v6
    v5 = bxor v4, v3
    store v5, v1
    return v5
}
";

/// Callees with narrow parameters and results, under each extension
/// convention; their arithmetic leaves the upper bits dirty unless masked.
const NARROW_CALLEES: [&str; 5] = [
    "function %n_plain(i8, i16) -> i8, i16 system_v {
block0(v0: i8, v1: i16):
    v2 = iadd v0, v0
    v3 = imul v1, v1
    v4 = ireduce.i8 v1
    v5 = isub v2, v4
    return v5, v3
}",
    "function %n_sext(i8 sext, i16 sext) -> i8 sext, i16 sext system_v {
block0(v0: i8, v1: i16):
    v2 = iadd v0, v0
    v3 = imul v1, v1
    v4 = ireduce.i8 v1
    v5 = isub v2, v4
    return v5, v3
}",
    "function %n_uext(i8 uext, i16 uext) -> i8 uext, i16 uext system_v {
block0(v0: i8, v1: i16):
    v2 = iadd v0, v0
    v3 = imul v1, v1
    v4 = ireduce.i8 v1
    v5 = isub v2, v4
    return v5, v3
}",
    "function %n_one_sext(i8 sext, i16 sext) -> i16 sext system_v {
block0(v0: i8, v1: i16):
    v2 = sextend.i16 v0
    v3 = imul v1, v2
    return v3
}",
    "function %n_mixed(i8 sext, i16) -> i64, i8 sext, i32, i16 uext system_v {
block0(v0: i8, v1: i16):
    v2 = sextend.i64 v0
    v3 = iadd v0, v0
    v4 = uextend.i32 v1
    v5 = ineg v1
    return v2, v3, v4, v5
}",
];

fn configs() -> Vec<(&'static str, Config)> {
    vec![
        ("default", Config::default()),
        ("none", harness::config(|c| c.opts = Optimizations::none())),
        ("no-tee", harness::config(|c| c.opts.use_tee = false)),
        ("no-coalesce", harness::config(|c| c.opts.coalesce = false)),
        ("no-stack-values", harness::config(|c| c.opts.stack_values = false)),
        ("memory64", harness::config(|c| c.module.memory64 = true)),
        (
            "demote",
            harness::config(|c| c.module.demote_multi_results = true),
        ),
    ]
}

fn arg_strategy(tys: Vec<Type>) -> impl Strategy<Value = Vec<DataValue>> {
    // Fuel: how many blocks run before every block returns.
    let fuel = prop_oneof![1i32..8, 8i32..64].prop_map(DataValue::I32).boxed();
    let vals: Vec<BoxedStrategy<DataValue>> = std::iter::once(fuel)
        .chain(tys.into_iter().map(|t| {
            let small = || prop_oneof![Just(0i64), Just(1), Just(-1), -4i64..64, any::<i64>()];
            match t {
                types::I8 => small().prop_map(|v| DataValue::I8(v as i8)).boxed(),
                types::I16 => small().prop_map(|v| DataValue::I16(v as i16)).boxed(),
                types::I32 => small().prop_map(|v| DataValue::I32(v as i32)).boxed(),
                types::I64 => small().prop_map(DataValue::I64).boxed(),
                types::I128 => (small(), small())
                    .prop_map(|(hi, lo)| DataValue::I128(((hi as i128) << 64) | lo as u64 as i128))
                    .boxed(),
                // Bit patterns, and ordinary values.
                types::F32 => prop_oneof![
                    small().prop_map(|v| v as u32),
                    small().prop_map(|v| (v as f32 / 4.0).to_bits()),
                ]
                .prop_map(|b| DataValue::F32(Ieee32::with_bits(b)))
                .boxed(),
                types::F64 => prop_oneof![
                    small().prop_map(|v| v as u64),
                    small().prop_map(|v| (v as f64 / 4.0).to_bits()),
                ]
                .prop_map(|b| DataValue::F64(Ieee64::with_bits(b)))
                .boxed(),
                _ => {
                    let lane = t.lane_type();
                    let lb = lane.bytes() as usize;
                    prop::collection::vec(small(), t.lane_count() as usize)
                        .prop_map(move |ls| {
                            let mut b = Vec::new();
                            for &v in &ls {
                                let bits = match lane {
                                    types::F32 if v & 1 == 0 => (v as f32 / 4.0).to_bits() as u64,
                                    types::F64 if v & 1 == 0 => (v as f64 / 4.0).to_bits(),
                                    _ => v as u64,
                                };
                                b.extend_from_slice(&bits.to_le_bytes()[..lb]);
                            }
                            match b.len() {
                                16 => DataValue::V128(b.try_into().unwrap()),
                                8 => DataValue::V64(b.try_into().unwrap()),
                                4 => DataValue::V32(b.try_into().unwrap()),
                                _ => DataValue::V16(b.try_into().unwrap()),
                            }
                        })
                        .boxed()
                }
            }
        }))
        .collect();
    vals
}

/// With `CFG_STATS` set, report what happened to each case on stderr.
fn stat(what: &str) {
    if std::env::var_os("CFG_STATS").is_some() {
        eprintln!("STAT {what}");
    }
}

/// Compare interpreter and wasm results for `func` on `args`.
fn check(func: &Function, args: &[DataValue]) -> Result<(), String> {
    cranelift_codegen::verify_function(
        func,
        &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .map_err(|e| format!("generator made invalid IR: {e}\n{func}"))?;
    let returns: Vec<Type> = func.signature.returns.iter().map(|p| p.value_type).collect();
    let callee = cranelift_reader::parse_functions(CALLEE).unwrap().remove(0);
    let mut funcs = vec![func.clone(), callee];
    funcs.extend(cranelift_reader::parse_functions(BYVAL_CALLEE).unwrap());
    funcs.extend(cranelift_reader::parse_functions(WRITER).unwrap());
    for text in NARROW_CALLEES {
        funcs.extend(cranelift_reader::parse_functions(text).unwrap());
    }
    let expected = match harness::interpret(&funcs, "f", args) {
        Some(Ok(e)) => e,
        Some(Err(e)) => return Err(format!("interpreter error: {e}\n{func}")),
        // The interpreter does not implement every operation on every type.
        None => {
            stat("skipped");
            return Ok(());
        }
    };
    // cranelift-interpreter sometimes returns a value of the wrong type
    // (see `Op::VBitcast`); such a result says nothing about the backend.
    if let Some(e) = &expected
        && e.iter().zip(&returns).any(|(v, t)| {
            v.ty().bits() != t.bits() || v.ty().is_vector() != t.is_vector()
        })
    {
        stat("ill-typed-oracle");
        return Ok(());
    }
    stat(if expected.is_some() { "returned" } else { "trapped" });
    for (name, config) in configs() {
        let (wasm, unsupported) = harness::compile_module(&funcs, &config);
        if !unsupported.is_empty() {
            return Err(format!("[{name}] unsupported {unsupported:?}\n{func}"));
        }
        harness::validate(&wasm).map_err(|e| format!("[{name}] invalid wasm: {e}\n{func}"))?;
        let got = harness::Runner::new(&wasm)?.call("f", args, &returns)?;
        let ok = match (&expected, &got) {
            (None, None) => true,
            (Some(e), Some(g)) => harness::same_results(e, g, &returns),
            _ => false,
        };
        if !ok {
            return Err(format!(
                "[{name}] args {args:?}\n  expected {expected:?}\n  got      {got:?}\n{func}"
            ));
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(256),
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::WithSource("regressions"),
        )),
        ..ProptestConfig::default()
    })]

    #[test]
    fn arbitrary_cfgs_match_interpreter(
        (spec, args) in spec().prop_flat_map(|s| {
            let tys = s.tys.clone();
            (Just(s), arg_strategy(tys))
        })
    ) {
        let func = build(&spec);
        if let Err(e) = check(&func, &args) {
            prop_assert!(false, "{}", e);
        }
    }
}
