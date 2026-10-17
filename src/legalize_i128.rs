//! Legalization of `i128` into pairs of `i64`.
//!
//! This is an IR-to-IR pass that runs before everything else, so the rest of
//! the backend never sees `i128`. Every `i128` value becomes a `(lo, hi)`
//! pair of `i64` values: in signatures (an `i128` parameter or return becomes
//! two `i64`s, low half first; see [`crate::wasm_signature`]), block
//! parameters, branch arguments, calls and returns.
//!
//! Most operations expand into straight-line `i64` code:
//!
//! - add/sub propagate a carry computed with an unsigned compare;
//! - multiply is `lo*lo` plus the cross products in the high half, using
//!   `umulhi.i64`;
//! - shifts and rotates select between the "amount < 64" and "amount >= 64"
//!   results, so no branches are needed;
//! - ordered comparisons compare the high halves, and the low halves
//!   (unsigned) when the high halves are equal;
//! - conversions from integers to floats round the top 64 significant bits
//!   to odd (a sticky bit) before converting, which gives correct rounding.
//!
//! Division and remainder become a small shift-subtract loop: the block is
//! split at the division and 128 iterations of a restoring division run in
//! between.

use crate::{Result, unsupported};
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::dominator_tree::DominatorTree;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee64, Offset32};
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, BlockCall, Endianness, Function, Inst, InstBuilder,
    InstructionData as D, Opcode, Signature, TrapCode, Type, Value, types::*,
};
use std::borrow::Cow;

type Pair = (Value, Value);

/// Prepare `func` for the backend pipeline: resolve value aliases and legalize
/// `i128` into pairs of `i64`. Borrows `func` without cloning when neither is
/// present.
pub(crate) fn prepare(func: &Function) -> Result<Cow<'_, Function>> {
    let (aliases, wide) = scan(func);
    if !aliases && !wide {
        return Ok(Cow::Borrowed(func));
    }
    let mut f = func.clone();
    f.dfg.resolve_all_aliases();
    if wide {
        legalize(&mut f)?;
    }
    Ok(Cow::Owned(f))
}

/// One pass over the function: does it use value aliases, and does it
/// involve `i128` anywhere (signatures, block parameters or results)?
fn scan(func: &Function) -> (bool, bool) {
    let dfg = &func.dfg;
    let sig_has = |s: &Signature| {
        s.params
            .iter()
            .chain(&s.returns)
            .any(|p| p.value_type == I128)
    };
    let mut aliases = false;
    let mut wide = sig_has(&func.signature) || dfg.signatures.values().any(sig_has);
    for b in func.layout.blocks() {
        wide |= dfg
            .block_params(b)
            .iter()
            .any(|&p| dfg.value_type(p) == I128);
        for inst in func.layout.block_insts(b) {
            aliases |= dfg.inst_values(inst).any(|v| dfg.value_is_alias(v));
            wide |= dfg
                .inst_results(inst)
                .iter()
                .any(|&r| dfg.value_type(r) == I128);
        }
    }
    (aliases, wide)
}

fn legalize_signature(sig: &mut Signature) {
    let expand = |ps: &mut Vec<AbiParam>| {
        *ps = ps
            .iter()
            .flat_map(|&p| {
                if p.value_type == I128 {
                    vec![AbiParam::new(I64), AbiParam::new(I64)]
                } else {
                    vec![p]
                }
            })
            .collect();
    };
    expand(&mut sig.params);
    expand(&mut sig.returns);
}

/// What replaces a result of a legalized instruction.
#[derive(Clone, Copy)]
enum Out {
    Pair(Value, Value),
    Value(Value),
}

/// The function must be alias-free; aliases are created for replaced
/// scalar results and must be resolved afterwards.
fn legalize(func: &mut Function) -> Result<()> {
    legalize_signature(&mut func.signature);
    for sig in func.dfg.signatures.values_mut() {
        legalize_signature(sig);
    }

    let cfg = ControlFlowGraph::with_function(func);
    let domtree = DominatorTree::with_function(func, &cfg);
    // Defs dominate uses, so in reverse postorder every `i128` operand has
    // been replaced by a pair before it is used.
    let rpo: Vec<Block> = domtree.cfg_rpo().copied().collect();

    let mut l = Legalizer {
        func,
        pairs: SecondaryMap::new(),
    };
    for block in rpo {
        l.block_params(block);
        let insts: Vec<Inst> = l.func.layout.block_insts(block).collect();
        for inst in insts {
            l.inst(inst)?;
        }
    }
    l.func.dfg.resolve_all_aliases();
    Ok(())
}

struct Legalizer<'f> {
    func: &'f mut Function,
    pairs: SecondaryMap<Value, Option<Pair>>,
}

impl Legalizer<'_> {
    fn ty(&self, v: Value) -> Type {
        self.func.dfg.value_type(v)
    }

    fn pair(&self, v: Value) -> Result<Pair> {
        match self.pairs[v] {
            Some(p) => Ok(p),
            None => unsupported(format!("i128 value {v} used before its definition")),
        }
    }

    fn block_params(&mut self, block: Block) {
        let dfg = &mut self.func.dfg;
        if !dfg
            .block_params(block)
            .iter()
            .any(|&p| dfg.value_type(p) == I128)
        {
            return;
        }
        let old = dfg.detach_block_params(block);
        let old: Vec<Value> = old.as_slice(&dfg.value_lists).to_vec();
        for p in old {
            if dfg.value_type(p) == I128 {
                let lo = dfg.append_block_param(block, I64);
                let hi = dfg.append_block_param(block, I64);
                self.pairs[p] = Some((lo, hi));
            } else {
                dfg.attach_block_param(block, p);
            }
        }
    }

    /// The `i64` arguments replacing a list of values.
    fn expand(&self, vals: &[Value]) -> Result<Vec<Value>> {
        let mut out = Vec::with_capacity(vals.len());
        for &v in vals {
            if self.ty(v) == I128 {
                let (lo, hi) = self.pair(v)?;
                out.extend([lo, hi]);
            } else {
                out.push(v);
            }
        }
        Ok(out)
    }

    /// Replace the results of `old` and remove it.
    fn finish(&mut self, old: Inst, outs: &[Out]) {
        let results = self.func.dfg.inst_results(old).to_vec();
        debug_assert_eq!(results.len(), outs.len());
        self.func.dfg.detach_inst_results(old);
        for (&r, &out) in results.iter().zip(outs) {
            match out {
                Out::Pair(lo, hi) => self.pairs[r] = Some((lo, hi)),
                Out::Value(v) => self.func.dfg.change_to_alias(r, v),
            }
        }
        FuncCursor::new(self.func).at_inst(old).remove_inst();
    }

    /// Map the results of `old` to those of `new`, which has the same
    /// results except that each `i128` became two `i64`s.
    fn finish_expanded(&mut self, old: Inst, new: Inst) {
        let olds = self.func.dfg.inst_results(old).to_vec();
        let news = self.func.dfg.inst_results(new).to_vec();
        let mut outs = Vec::new();
        let mut i = 0;
        for r in olds {
            if self.ty(r) == I128 {
                outs.push(Out::Pair(news[i], news[i + 1]));
                i += 2;
            } else {
                outs.push(Out::Value(news[i]));
                i += 1;
            }
        }
        self.finish(old, &outs);
    }

    fn inst(&mut self, inst: Inst) -> Result<()> {
        let dfg = &self.func.dfg;
        let data = dfg.insts[inst];
        let op = data.opcode();
        let inst_args = dfg.inst_args(inst);
        let inst_results = dfg.inst_results(inst);
        let wide_arg = inst_args.iter().any(|&a| dfg.value_type(a) == I128);
        let wide_res = inst_results.iter().any(|&r| dfg.value_type(r) == I128);
        let wide_branch_arg = data
            .branch_destination(&dfg.jump_tables, &dfg.exception_tables)
            .iter()
            .any(|c| {
                c.args(&dfg.value_lists)
                    .any(|a| a.as_value().is_some_and(|v| dfg.value_type(v) == I128))
            });
        // Splitting a try_call's i128 results or payload would renumber the
        // `ret`/`exn` arguments of its edges; not supported.
        if let Some(et) = data.exception_table() {
            let sig = &dfg.signatures[dfg.exception_tables[et].signature()];
            let wide_sig = sig
                .params
                .iter()
                .chain(&sig.returns)
                .any(|p| p.value_type == I128);
            if wide_sig || wide_arg || wide_branch_arg {
                return unsupported("i128 in try_call");
            }
        }
        if !wide_arg && !wide_res && !wide_branch_arg {
            return Ok(());
        }
        let ty = inst_results.first().map_or(INVALID, |&r| dfg.value_type(r));
        let args: Vec<Value> = inst_args.to_vec();

        // Operations that keep their non-i128 shape: replace the i128
        // operand in place.
        match op {
            Opcode::Brif
            | Opcode::Trapz
            | Opcode::Trapnz
            | Opcode::Select
            | Opcode::SelectSpectreGuard
                if wide_arg && self.ty(args[0]) == I128 && !wide_res =>
            {
                let (lo, hi) = self.pair(args[0])?;
                let c = FuncCursor::new(self.func).at_inst(inst).ins().bor(lo, hi);
                self.func.dfg.inst_args_mut(inst)[0] = c;
                return self.branch_args(inst);
            }
            // A narrower value shifted or rotated by an i128 amount: only
            // the low bits of the amount matter.
            Opcode::Ishl | Opcode::Ushr | Opcode::Sshr | Opcode::Rotl | Opcode::Rotr
                if ty != I128 =>
            {
                let (lo, _) = self.pair(args[1])?;
                self.func.dfg.inst_args_mut(inst)[1] = lo;
                return Ok(());
            }
            Opcode::Jump | Opcode::Brif | Opcode::BrTable => return self.branch_args(inst),
            _ => {}
        }

        let arg_pairs: Vec<Option<Pair>> = args.iter().map(|&a| self.pairs[a]).collect();
        let arg_tys: Vec<Type> = args.iter().map(|&a| self.ty(a)).collect();
        let p = |i: usize| -> Result<Pair> {
            match arg_pairs[i] {
                Some(pair) => Ok(pair),
                None => unsupported(format!("i128 value {} used before its definition", args[i])),
            }
        };

        // Calls, returns, and division/remainder manage their own cursor or results.
        match (op, data) {
            (Opcode::Udiv | Opcode::Sdiv | Opcode::Urem | Opcode::Srem, _) => {
                let (x, d) = (p(0)?, p(1)?);
                let signed = matches!(op, Opcode::Sdiv | Opcode::Srem);
                let rem = matches!(op, Opcode::Urem | Opcode::Srem);
                let (lo, hi) = self.divmod(inst, signed, rem, x, d);
                self.finish(inst, &[Out::Pair(lo, hi)]);
                return Ok(());
            }
            (Opcode::Return, _) => {
                let vals = self.expand(&args)?;
                let mut pos = FuncCursor::new(self.func).at_inst(inst);
                pos.ins().return_(&vals);
                pos.remove_inst();
                return Ok(());
            }
            (Opcode::Call | Opcode::ReturnCall, D::Call { func_ref, .. }) => {
                let vals = self.expand(&args)?;
                let mut pos = FuncCursor::new(self.func).at_inst(inst);
                let new = if op == Opcode::Call {
                    pos.ins().call(func_ref, &vals)
                } else {
                    pos.ins().return_call(func_ref, &vals)
                };
                self.finish_expanded(inst, new);
                return Ok(());
            }
            (
                Opcode::CallIndirect | Opcode::ReturnCallIndirect,
                D::CallIndirect { sig_ref, .. },
            ) => {
                let vals = self.expand(&args[1..])?;
                let mut pos = FuncCursor::new(self.func).at_inst(inst);
                let new = if op == Opcode::CallIndirect {
                    pos.ins().call_indirect(sig_ref, args[0], &vals)
                } else {
                    pos.ins().return_call_indirect(sig_ref, args[0], &vals)
                };
                self.finish_expanded(inst, new);
                return Ok(());
            }
            _ => {}
        }

        let mut pos = FuncCursor::new(self.func).at_inst(inst);
        let outs: Vec<Out> = match (op, data) {
            (Opcode::Iconst, D::UnaryImm { imm, .. }) => {
                // The immediate is zero-extended.
                let lo = pos.ins().iconst(I64, imm.bits());
                let hi = pos.ins().iconst(I64, 0);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Iconcat, _) => vec![Out::Pair(args[0], args[1])],
            (Opcode::Isplit, _) => {
                let (lo, hi) = p(0)?;
                vec![Out::Value(lo), Out::Value(hi)]
            }
            (Opcode::Uextend | Opcode::Sextend, _) => {
                let x = args[0];
                let xty = arg_tys[0];
                let lo = if xty == I64 {
                    x
                } else if op == Opcode::Uextend {
                    pos.ins().uextend(I64, x)
                } else {
                    pos.ins().sextend(I64, x)
                };
                let hi = if op == Opcode::Uextend {
                    pos.ins().iconst(I64, 0)
                } else {
                    pos.ins().sshr_imm_u(lo, 63)
                };
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Ireduce, _) => {
                let (lo, _) = p(0)?;
                let v = if ty == I64 {
                    lo
                } else {
                    pos.ins().ireduce(ty, lo)
                };
                vec![Out::Value(v)]
            }
            (Opcode::Iadd | Opcode::Isub | Opcode::Imul, _) => {
                let (a, b) = (p(0)?, p(1)?);
                let (lo, hi) = match op {
                    Opcode::Iadd => add(&mut pos, a, b),
                    Opcode::Isub => sub(&mut pos, a, b),
                    _ => mul(&mut pos, a, b),
                };
                vec![Out::Pair(lo, hi)]
            }
            (
                Opcode::UaddOverflow
                | Opcode::SaddOverflow
                | Opcode::UaddOverflowCin
                | Opcode::SaddOverflowCin
                | Opcode::UsubOverflow
                | Opcode::SsubOverflow
                | Opcode::UsubOverflowBin
                | Opcode::SsubOverflowBin,
                _,
            ) => {
                // The low halves' carry or borrow feeds the high halves,
                // whose own flag is the result's.
                let ((al, ah), (bl, bh)) = (p(0)?, p(1)?);
                let add = matches!(
                    op,
                    Opcode::UaddOverflow
                        | Opcode::SaddOverflow
                        | Opcode::UaddOverflowCin
                        | Opcode::SaddOverflowCin
                );
                let signed = matches!(
                    op,
                    Opcode::SaddOverflow
                        | Opcode::SaddOverflowCin
                        | Opcode::SsubOverflow
                        | Opcode::SsubOverflowBin
                );
                let c_in = args.get(2).copied();
                let (lo, c) = match (add, c_in) {
                    (true, None) => pos.ins().uadd_overflow(al, bl),
                    (true, Some(c)) => pos.ins().uadd_overflow_cin(al, bl, c),
                    (false, None) => pos.ins().usub_overflow(al, bl),
                    (false, Some(c)) => pos.ins().usub_overflow_bin(al, bl, c),
                };
                let (hi, of) = match (add, signed) {
                    (true, false) => pos.ins().uadd_overflow_cin(ah, bh, c),
                    (true, true) => pos.ins().sadd_overflow_cin(ah, bh, c),
                    (false, false) => pos.ins().usub_overflow_bin(ah, bh, c),
                    (false, true) => pos.ins().ssub_overflow_bin(ah, bh, c),
                };
                vec![Out::Pair(lo, hi), Out::Value(of)]
            }
            (Opcode::Band | Opcode::Bor | Opcode::Bxor, _) => {
                let ((al, ah), (bl, bh)) = (p(0)?, p(1)?);
                let f = |pos: &mut FuncCursor, x, y| match op {
                    Opcode::Band => pos.ins().band(x, y),
                    Opcode::Bor => pos.ins().bor(x, y),
                    _ => pos.ins().bxor(x, y),
                };
                let lo = f(&mut pos, al, bl);
                let hi = f(&mut pos, ah, bh);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Bnot, _) => {
                let (l, h) = p(0)?;
                let lo = pos.ins().bnot(l);
                let hi = pos.ins().bnot(h);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Bitselect, _) => {
                let ((cl, ch), (xl, xh), (yl, yh)) = (p(0)?, p(1)?, p(2)?);
                let lo = pos.ins().bitselect(cl, xl, yl);
                let hi = pos.ins().bitselect(ch, xh, yh);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Ineg, _) => {
                let x = p(0)?;
                let (lo, hi) = neg(&mut pos, x);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Iabs, _) => {
                let x = p(0)?;
                let n = neg(&mut pos, x);
                let is_neg = pos.ins().icmp_imm_s(IntCC::SignedLessThan, x.1, 0);
                let (lo, hi) = select(&mut pos, is_neg, n, x);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Smin | Opcode::Smax | Opcode::Umin | Opcode::Umax, _) => {
                let (a, b) = (p(0)?, p(1)?);
                let cc = match op {
                    Opcode::Smin => IntCC::SignedLessThan,
                    Opcode::Smax => IntCC::SignedGreaterThan,
                    Opcode::Umin => IntCC::UnsignedLessThan,
                    _ => IntCC::UnsignedGreaterThan,
                };
                let c = icmp(&mut pos, cc, a, b);
                let (lo, hi) = select(&mut pos, c, a, b);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Select | Opcode::SelectSpectreGuard, _) => {
                let c = if arg_tys[0] == I128 {
                    let (l, h) = p(0)?;
                    pos.ins().bor(l, h)
                } else {
                    args[0]
                };
                let (x, y) = (p(1)?, p(2)?);
                let (lo, hi) = select(&mut pos, c, x, y);
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Icmp, D::IntCompare { cond, .. }) => {
                let (a, b) = (p(0)?, p(1)?);
                vec![Out::Value(icmp(&mut pos, cond, a, b))]
            }
            (Opcode::Bmask, _) => {
                let x = if arg_tys[0] == I128 {
                    let (l, h) = p(0)?;
                    pos.ins().bor(l, h)
                } else {
                    args[0]
                };
                if ty == I128 {
                    let m = pos.ins().bmask(I64, x);
                    vec![Out::Pair(m, m)]
                } else {
                    vec![Out::Value(pos.ins().bmask(ty, x))]
                }
            }
            (Opcode::Ishl | Opcode::Ushr | Opcode::Sshr | Opcode::Rotl | Opcode::Rotr, _) => {
                let x = p(0)?;
                let amt = if arg_tys[1] == I128 { p(1)?.0 } else { args[1] };
                let amt = to_i64(&mut pos, amt);
                let (lo, hi) = match op {
                    Opcode::Ishl | Opcode::Ushr | Opcode::Sshr => shift(&mut pos, op, x, amt),
                    _ => {
                        // rot(x, n) = (x << n) | (x >> (-n & 127)), or the
                        // mirror image for rotr.
                        let (first, second) = if op == Opcode::Rotl {
                            (Opcode::Ishl, Opcode::Ushr)
                        } else {
                            (Opcode::Ushr, Opcode::Ishl)
                        };
                        let a = shift(&mut pos, first, x, amt);
                        let neg_amt = pos.ins().ineg(amt);
                        let b = shift(&mut pos, second, x, neg_amt);
                        (pos.ins().bor(a.0, b.0), pos.ins().bor(a.1, b.1))
                    }
                };
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::Clz | Opcode::Ctz | Opcode::Popcnt | Opcode::Cls, _) => {
                let (l, h) = p(0)?;
                let n = match op {
                    Opcode::Clz => clz(&mut pos, (l, h)),
                    Opcode::Ctz => {
                        let lz = pos.ins().icmp_imm_s(IntCC::Equal, l, 0);
                        let ch = pos.ins().ctz(h);
                        let ch = pos.ins().iadd_imm_s(ch, 64);
                        let cl = pos.ins().ctz(l);
                        pos.ins().select(lz, ch, cl)
                    }
                    Opcode::Popcnt => {
                        let a = pos.ins().popcnt(l);
                        let b = pos.ins().popcnt(h);
                        pos.ins().iadd(a, b)
                    }
                    _ => {
                        // Leading bits equal to the sign bit, excluding it.
                        let s = pos.ins().sshr_imm_u(h, 63);
                        let a = pos.ins().bxor(l, s);
                        let b = pos.ins().bxor(h, s);
                        let c = clz(&mut pos, (a, b));
                        pos.ins().iadd_imm_s(c, -1)
                    }
                };
                let zero = pos.ins().iconst(I64, 0);
                vec![Out::Pair(n, zero)]
            }
            (Opcode::Bswap | Opcode::Bitrev, _) => {
                let (l, h) = p(0)?;
                let (lo, hi) = if op == Opcode::Bswap {
                    (pos.ins().bswap(h), pos.ins().bswap(l))
                } else {
                    (pos.ins().bitrev(h), pos.ins().bitrev(l))
                };
                vec![Out::Pair(lo, hi)]
            }
            (Opcode::UaddOverflowTrap, D::IntAddTrap { code, .. }) => {
                let (a, b) = (p(0)?, p(1)?);
                let r = add(&mut pos, a, b);
                let overflow = icmp(&mut pos, IntCC::UnsignedLessThan, r, a);
                pos.ins().trapnz(overflow, code);
                vec![Out::Pair(r.0, r.1)]
            }

            // Memory.
            (Opcode::Load, D::Load { flags, offset, .. }) if ty == I128 => {
                let flags = pos.func.dfg.mem_flags[flags];
                let (lo_off, hi_off) = half_offsets(flags.endianness(Endianness::Little), offset)?;
                let lo = pos.ins().load(I64, flags, args[0], lo_off);
                let hi = pos.ins().load(I64, flags, args[0], hi_off);
                vec![Out::Pair(lo, hi)]
            }
            (
                Opcode::Store | Opcode::Istore8 | Opcode::Istore16 | Opcode::Istore32,
                D::Store { flags, offset, .. },
            ) => {
                let (lo, hi) = p(0)?;
                let addr = args[1];
                let flags = pos.func.dfg.mem_flags[flags];
                match op {
                    Opcode::Store => {
                        let (lo_off, hi_off) =
                            half_offsets(flags.endianness(Endianness::Little), offset)?;
                        pos.ins().store(flags, lo, addr, lo_off);
                        pos.ins().store(flags, hi, addr, hi_off);
                    }
                    Opcode::Istore8 => {
                        pos.ins().istore8(flags, lo, addr, offset);
                    }
                    Opcode::Istore16 => {
                        pos.ins().istore16(flags, lo, addr, offset);
                    }
                    _ => {
                        pos.ins().istore32(flags, lo, addr, offset);
                    }
                }
                vec![]
            }

            // Bitcasts to and from 128-bit vectors.
            (Opcode::Bitcast, D::LoadNoOffset { flags, .. }) => {
                let flags = pos.func.dfg.mem_flags[flags];
                if flags.endianness(Endianness::Little) == Endianness::Big {
                    return unsupported("big-endian i128 bitcast");
                }
                let aty = arg_tys[0];
                if aty == I128 && ty == I128 {
                    let (lo, hi) = p(0)?;
                    vec![Out::Pair(lo, hi)]
                } else if ty == I128 && aty.is_vector() && aty.bits() == 128 {
                    let v = if aty == I64X2 {
                        args[0]
                    } else {
                        pos.ins().bitcast(I64X2, flags, args[0])
                    };
                    let lo = pos.ins().extractlane(v, 0);
                    let hi = pos.ins().extractlane(v, 1);
                    vec![Out::Pair(lo, hi)]
                } else if aty == I128 && ty.is_vector() && ty.bits() == 128 {
                    let (lo, hi) = p(0)?;
                    let v = pos.ins().scalar_to_vector(I64X2, lo);
                    let v = pos.ins().insertlane(v, hi, 1);
                    let v = if ty == I64X2 {
                        v
                    } else {
                        pos.ins().bitcast(ty, flags, v)
                    };
                    vec![Out::Value(v)]
                } else {
                    return unsupported(format!("bitcast {aty} -> {ty}"));
                }
            }

            // Conversions to and from floats.
            (Opcode::FcvtFromSint | Opcode::FcvtFromUint, _) => {
                let x = p(0)?;
                let v = if op == Opcode::FcvtFromUint {
                    u128_to_float(&mut pos, ty, x)
                } else {
                    let is_neg = pos.ins().icmp_imm_s(IntCC::SignedLessThan, x.1, 0);
                    let n = neg(&mut pos, x);
                    let abs = select(&mut pos, is_neg, n, x);
                    let f = u128_to_float(&mut pos, ty, abs);
                    let nf = pos.ins().fneg(f);
                    pos.ins().select(is_neg, nf, f)
                };
                vec![Out::Value(v)]
            }
            (
                Opcode::FcvtToSint
                | Opcode::FcvtToUint
                | Opcode::FcvtToSintSat
                | Opcode::FcvtToUintSat,
                _,
            ) => {
                let (lo, hi) = float_to_128(&mut pos, op, args[0]);
                vec![Out::Pair(lo, hi)]
            }

            _ => {
                return unsupported(format!("i128 in `{}`", pos.func.dfg.display_inst(inst)));
            }
        };
        self.finish(inst, &outs);
        Ok(())
    }

    /// Expand the `i128` arguments of a branch's block calls.
    fn branch_args(&mut self, inst: Inst) -> Result<()> {
        let dfg = &self.func.dfg;
        let calls: Vec<BlockCall> = dfg.insts[inst]
            .branch_destination(&dfg.jump_tables, &dfg.exception_tables)
            .to_vec();
        let mut new_calls = Vec::with_capacity(calls.len());
        for call in calls {
            let dfg = &self.func.dfg;
            let block = call.block(&dfg.value_lists);
            let mut args: Vec<BlockArg> = Vec::new();
            for a in call.args(&dfg.value_lists) {
                match a.as_value() {
                    Some(v) if dfg.value_type(v) == I128 => {
                        let (lo, hi) = self.pair(v)?;
                        args.extend([BlockArg::Value(lo), BlockArg::Value(hi)]);
                    }
                    _ => args.push(a),
                }
            }
            new_calls.push(self.func.dfg.block_call(block, &args));
        }
        let dfg = &mut self.func.dfg;
        let slots =
            dfg.insts[inst].branch_destination_mut(&mut dfg.jump_tables, &mut dfg.exception_tables);
        for (slot, new) in slots.iter_mut().zip(new_calls) {
            *slot = new;
        }
        Ok(())
    }

    /// Replace a 128-bit division or remainder with a shift-subtract loop.
    ///
    /// ```text
    /// block:  <checks; absolute values>          jump body(n, 0, 128)
    /// body(n, r, i):
    ///         (r, n) = (r, n) << 1               ; 256-bit shift
    ///         if r >= d { r -= d; n |= 1 }       ; branchless
    ///         brif i - 1, body(n, r, i - 1), cont(n, r)
    /// cont(q, r):
    ///         <sign fix-ups>; the rest of the original block
    /// ```
    fn divmod(&mut self, inst: Inst, signed: bool, rem: bool, x: Pair, d: Pair) -> Pair {
        let orig = self.func.layout.inst_block(inst).unwrap();
        let mut pos = FuncCursor::new(self.func).at_inst(inst);

        let nonzero = pos.ins().bor(d.0, d.1);
        pos.ins().trapz(nonzero, TrapCode::INTEGER_DIVISION_BY_ZERO);
        let (x, d, x_neg, d_neg) = if signed {
            if !rem {
                // MIN / -1 overflows.
                let zero = pos.ins().iconst(I64, 0);
                let min_hi = pos.ins().iconst(I64, i64::MIN);
                let all = pos.ins().iconst(I64, -1);
                let is_min = icmp(&mut pos, IntCC::Equal, x, (zero, min_hi));
                let is_m1 = icmp(&mut pos, IntCC::Equal, d, (all, all));
                let both = pos.ins().band(is_min, is_m1);
                pos.ins().trapnz(both, TrapCode::INTEGER_OVERFLOW);
            }
            let x_neg = pos.ins().icmp_imm_s(IntCC::SignedLessThan, x.1, 0);
            let d_neg = pos.ins().icmp_imm_s(IntCC::SignedLessThan, d.1, 0);
            let nx = neg(&mut pos, x);
            let nd = neg(&mut pos, d);
            let ax = select(&mut pos, x_neg, nx, x);
            let ad = select(&mut pos, d_neg, nd, d);
            (ax, ad, Some(x_neg), Some(d_neg))
        } else {
            (x, d, None, None)
        };

        let func = &mut *pos.func;
        let cont = func.dfg.make_block();
        func.layout.split_block(cont, inst);
        let body = func.dfg.make_block();
        func.layout.insert_block(body, cont);
        let [b_nlo, b_nhi, b_rlo, b_rhi, b_i] =
            [(); 5].map(|_| func.dfg.append_block_param(body, I64));
        let [q_lo, q_hi, r_lo, r_hi] = [(); 4].map(|_| func.dfg.append_block_param(cont, I64));

        let mut pos = FuncCursor::new(func).at_bottom(orig);
        let zero = pos.ins().iconst(I64, 0);
        let count = pos.ins().iconst(I64, 128);
        let args = [x.0, x.1, zero, zero, count].map(BlockArg::Value);
        pos.ins().jump(body, &args);

        pos.goto_bottom(body);
        let top = pos.ins().ushr_imm_u(b_nhi, 63);
        let rh = pos.ins().ishl_imm_u(b_rhi, 1);
        let rl_top = pos.ins().ushr_imm_u(b_rlo, 63);
        let rh = pos.ins().bor(rh, rl_top);
        let rl = pos.ins().ishl_imm_u(b_rlo, 1);
        let rl = pos.ins().bor(rl, top);
        let nh = pos.ins().ishl_imm_u(b_nhi, 1);
        let nl_top = pos.ins().ushr_imm_u(b_nlo, 63);
        let nh = pos.ins().bor(nh, nl_top);
        let nl = pos.ins().ishl_imm_u(b_nlo, 1);
        let ge = icmp(&mut pos, IntCC::UnsignedGreaterThanOrEqual, (rl, rh), d);
        let diff = sub(&mut pos, (rl, rh), d);
        let (rl, rh) = select(&mut pos, ge, diff, (rl, rh));
        let bit = pos.ins().uextend(I64, ge);
        let nl = pos.ins().bor(nl, bit);
        let i = pos.ins().iadd_imm_s(b_i, -1);
        let again = [nl, nh, rl, rh, i].map(BlockArg::Value);
        let done = [nl, nh, rl, rh].map(BlockArg::Value);
        pos.ins().brif(i, body, &again, cont, &done);

        let mut pos = FuncCursor::new(func).at_inst(inst);
        match (rem, x_neg, d_neg) {
            (false, Some(xn), Some(dn)) => {
                let q = (q_lo, q_hi);
                let sign = pos.ins().bxor(xn, dn);
                let nq = neg(&mut pos, q);
                select(&mut pos, sign, nq, q)
            }
            (true, Some(xn), _) => {
                // The remainder has the sign of the dividend.
                let r = (r_lo, r_hi);
                let nr = neg(&mut pos, r);
                select(&mut pos, xn, nr, r)
            }
            (false, _, _) => (q_lo, q_hi),
            (true, _, _) => (r_lo, r_hi),
        }
    }
}

/// Offsets of the low and high halves of an `i128` in memory.
fn half_offsets(endianness: Endianness, offset: Offset32) -> Result<(i32, i32)> {
    let off = i32::from(offset);
    let Some(next) = off.checked_add(8) else {
        return unsupported("i128 memory offset overflow");
    };
    Ok(match endianness {
        Endianness::Little => (off, next),
        Endianness::Big => (next, off),
    })
}

fn to_i64(pos: &mut FuncCursor, v: Value) -> Value {
    match pos.func.dfg.value_type(v) {
        I64 => v,
        _ => pos.ins().uextend(I64, v),
    }
}

fn add(pos: &mut FuncCursor, (al, ah): Pair, (bl, bh): Pair) -> Pair {
    let lo = pos.ins().iadd(al, bl);
    let carry = pos.ins().icmp(IntCC::UnsignedLessThan, lo, al);
    let carry = pos.ins().uextend(I64, carry);
    let hi = pos.ins().iadd(ah, bh);
    let hi = pos.ins().iadd(hi, carry);
    (lo, hi)
}

fn sub(pos: &mut FuncCursor, (al, ah): Pair, (bl, bh): Pair) -> Pair {
    let lo = pos.ins().isub(al, bl);
    let borrow = pos.ins().icmp(IntCC::UnsignedLessThan, al, bl);
    let borrow = pos.ins().uextend(I64, borrow);
    let hi = pos.ins().isub(ah, bh);
    let hi = pos.ins().isub(hi, borrow);
    (lo, hi)
}

fn neg(pos: &mut FuncCursor, x: Pair) -> Pair {
    let zero = pos.ins().iconst(I64, 0);
    sub(pos, (zero, zero), x)
}

fn mul(pos: &mut FuncCursor, (al, ah): Pair, (bl, bh): Pair) -> Pair {
    let lo = pos.ins().imul(al, bl);
    let h0 = pos.ins().umulhi(al, bl);
    let h1 = pos.ins().imul(al, bh);
    let h2 = pos.ins().imul(ah, bl);
    let hi = pos.ins().iadd(h0, h1);
    let hi = pos.ins().iadd(hi, h2);
    (lo, hi)
}

fn select(pos: &mut FuncCursor, c: Value, (xl, xh): Pair, (yl, yh): Pair) -> Pair {
    (pos.ins().select(c, xl, yl), pos.ins().select(c, xh, yh))
}

fn icmp(pos: &mut FuncCursor, cc: IntCC, (al, ah): Pair, (bl, bh): Pair) -> Value {
    match cc {
        IntCC::Equal | IntCC::NotEqual => {
            let x = pos.ins().bxor(al, bl);
            let y = pos.ins().bxor(ah, bh);
            let any = pos.ins().bor(x, y);
            pos.ins().icmp_imm_s(cc, any, 0)
        }
        _ => {
            // The high halves decide unless they are equal.
            let hi_eq = pos.ins().icmp(IntCC::Equal, ah, bh);
            let hi = pos.ins().icmp(cc.without_equal(), ah, bh);
            let lo = pos.ins().icmp(cc.unsigned(), al, bl);
            pos.ins().select(hi_eq, lo, hi)
        }
    }
}

fn clz(pos: &mut FuncCursor, (l, h): Pair) -> Value {
    let hz = pos.ins().icmp_imm_s(IntCC::Equal, h, 0);
    let cl = pos.ins().clz(l);
    let cl = pos.ins().iadd_imm_s(cl, 64);
    let ch = pos.ins().clz(h);
    pos.ins().select(hz, cl, ch)
}

/// A 128-bit shift by an `i64` amount, taken modulo 128.
fn shift(pos: &mut FuncCursor, op: Opcode, (lo, hi): Pair, amt: Value) -> Pair {
    let amt = pos.ins().band_imm_u(amt, 127);
    let s = pos.ins().band_imm_u(amt, 63);
    let big = pos
        .ins()
        .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, amt, 64);
    // Bits crossing between the halves: x >> (64 - s), written as
    // (x >> 1) >> (63 - s) so that s = 0 gives 0.
    let sixty_three = pos.ins().iconst(I64, 63);
    let inv = pos.ins().isub(sixty_three, s);
    let zero = pos.ins().iconst(I64, 0);
    match op {
        Opcode::Ishl => {
            let lo_s = pos.ins().ishl(lo, s);
            let hi_s = pos.ins().ishl(hi, s);
            let carry = pos.ins().ushr_imm_u(lo, 1);
            let carry = pos.ins().ushr(carry, inv);
            let hi_small = pos.ins().bor(hi_s, carry);
            (
                pos.ins().select(big, zero, lo_s),
                pos.ins().select(big, lo_s, hi_small),
            )
        }
        _ => {
            let arith = op == Opcode::Sshr;
            let lo_s = pos.ins().ushr(lo, s);
            let hi_s = if arith {
                pos.ins().sshr(hi, s)
            } else {
                pos.ins().ushr(hi, s)
            };
            let carry = pos.ins().ishl_imm_u(hi, 1);
            let carry = pos.ins().ishl(carry, inv);
            let lo_small = pos.ins().bor(lo_s, carry);
            let fill = if arith {
                pos.ins().sshr_imm_u(hi, 63)
            } else {
                zero
            };
            (
                pos.ins().select(big, hi_s, lo_small),
                pos.ins().select(big, fill, hi_s),
            )
        }
    }
}

/// Convert an unsigned 128-bit integer to `f32` or `f64`, correctly
/// rounded.
fn u128_to_float(pos: &mut FuncCursor, ty: Type, (lo, hi): Pair) -> Value {
    let hz = pos.ins().icmp_imm_s(IntCC::Equal, hi, 0);
    let small = pos.ins().fcvt_from_uint(ty, lo);
    // Normalize so the top 64 significant bits are in one word, and OR the
    // rest into its lowest bit (round to odd). Rounding that word to the
    // float's precision then rounds the full value correctly.
    let n = pos.ins().clz(hi);
    let sixty_three = pos.ins().iconst(I64, 63);
    let inv = pos.ins().isub(sixty_three, n);
    let top = pos.ins().ishl(hi, n);
    let from_lo = pos.ins().ushr_imm_u(lo, 1);
    let from_lo = pos.ins().ushr(from_lo, inv);
    let top = pos.ins().bor(top, from_lo);
    let rest = pos.ins().ishl(lo, n);
    let sticky = pos.ins().icmp_imm_s(IntCC::NotEqual, rest, 0);
    let sticky = pos.ins().uextend(I64, sticky);
    let word = pos.ins().bor(top, sticky);
    let f = pos.ins().fcvt_from_uint(ty, word);
    // Scale by 2^(64 - n), built from its exponent bits.
    let sixty_four = pos.ins().iconst(I64, 64);
    let k = pos.ins().isub(sixty_four, n);
    let scale = if ty == F64 {
        let e = pos.ins().iadd_imm_s(k, 1023);
        let bits = pos.ins().ishl_imm_u(e, 52);
        pos.ins()
            .bitcast(F64, cranelift_codegen::ir::MemFlagsData::new(), bits)
    } else {
        let k = pos.ins().ireduce(I32, k);
        let e = pos.ins().iadd_imm_s(k, 127);
        let bits = pos.ins().ishl_imm_u(e, 23);
        pos.ins()
            .bitcast(F32, cranelift_codegen::ir::MemFlagsData::new(), bits)
    };
    let big = pos.ins().fmul(f, scale);
    pos.ins().select(hz, small, big)
}

/// `fcvt_to_{s,u}int{,_sat}` to `i128`.
fn float_to_128(pos: &mut FuncCursor, op: Opcode, x: Value) -> Pair {
    let signed = matches!(op, Opcode::FcvtToSint | Opcode::FcvtToSintSat);
    let sat = matches!(op, Opcode::FcvtToSintSat | Opcode::FcvtToUintSat);
    let x = if pos.func.dfg.value_type(x) == F32 {
        pos.ins().fpromote(F64, x)
    } else {
        x
    };
    let c = |pos: &mut FuncCursor, v: f64| pos.ins().f64const(Ieee64::with_float(v));
    let two64 = 18446744073709551616.0f64;
    let two127 = 170141183460469231731687303715884105728.0f64;

    // Range: (-1, 2^128) unsigned, [-2^127, 2^127) signed (after truncation).
    let (lo_bound, lo_cc, hi_bound) = if signed {
        (-two127, FloatCC::LessThan, two127)
    } else {
        (-1.0, FloatCC::LessThanOrEqual, two64 * two64)
    };
    let lo_b = c(pos, lo_bound);
    let hi_b = c(pos, hi_bound);
    let below = pos.ins().fcmp(lo_cc, x, lo_b);
    let above = pos.ins().fcmp(FloatCC::GreaterThanOrEqual, x, hi_b);
    if !sat {
        let nan = pos.ins().fcmp(FloatCC::Unordered, x, x);
        pos.ins().trapnz(nan, TrapCode::BAD_CONVERSION_TO_INTEGER);
        let out = pos.ins().bor(below, above);
        pos.ins().trapnz(out, TrapCode::INTEGER_OVERFLOW);
    }

    // Magnitude, split at 2^64. Saturating conversions turn NaN and
    // negative parts into 0.
    let a = if signed { pos.ins().fabs(x) } else { x };
    let inv64 = c(pos, 1.0 / two64);
    let scaled = pos.ins().fmul(a, inv64);
    let hf = pos.ins().trunc(scaled);
    let hi = pos.ins().fcvt_to_uint_sat(I64, hf);
    let two64v = c(pos, two64);
    let hpart = pos.ins().fmul(hf, two64v);
    let rest = pos.ins().fsub(a, hpart);
    let lo = pos.ins().fcvt_to_uint_sat(I64, rest);
    let mut r = (lo, hi);
    if signed {
        let zero = c(pos, 0.0);
        let is_neg = pos.ins().fcmp(FloatCC::LessThan, x, zero);
        let n = neg(pos, r);
        r = select(pos, is_neg, n, r);
    }
    if sat {
        let (min, max) = if signed {
            let zero = pos.ins().iconst(I64, 0);
            let all = pos.ins().iconst(I64, -1);
            let min_hi = pos.ins().iconst(I64, i64::MIN);
            let max_hi = pos.ins().iconst(I64, i64::MAX);
            ((zero, min_hi), (all, max_hi))
        } else {
            let zero = pos.ins().iconst(I64, 0);
            let all = pos.ins().iconst(I64, -1);
            ((zero, zero), (all, all))
        };
        r = select(pos, below, min, r);
        r = select(pos, above, max, r);
    }
    r
}
