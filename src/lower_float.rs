//! Floating-point operations that wasm lacks natively (`f16` and `fma`),
//! plus shared floating-point comparison lowering for scalars and vectors.
//!
//! An `f16` is represented by its bits, zero-extended in an `i32`. Loads,
//! stores, bitcasts and `select` handle it like an `i16`; `fneg`, `fabs`
//! and `fcopysign` are bit operations. Everything else converts to `f32`,
//! which holds every `f16` exactly, computes there and rounds back:
//!
//! - Comparisons, `fmin`/`fmax` and the rounding operations give exact
//!   results in `f32`.
//! - For `+`, `-`, `*`, `/` and `sqrt`, rounding to `f32` and then to `f16`
//!   gives the correctly rounded `f16` result, because `f32` has at least
//!   `2 * 11 + 2` significand bits (Figueroa, "When is double rounding
//!   innocuous?", 1995).
//!
//! Rounding to `f16` starts from an `f64`, so that `fdemote.f16` of an
//! `f64` rounds once.
//!
//! `fma` on `f32` and `f16` computes in `f64`: the product of two such
//! values is exact there, and the sum is rounded *to odd* (truncated, with
//! the last bit set if anything was lost). A round-to-odd result with at
//! least two more bits than the target rounds to the correctly rounded
//! target result (Boldo and Melquiond, "Emulation of FMA and correctly
//! rounded sums: proved algorithms using rounding to odd", 2008).
//!
//! `fma` on `f64` (and `f64x2`) unpacks finite non-zero operands into
//! 53-bit integer significands, computes the exact 106-bit product in a
//! 128-bit `(lo, hi)` pair, aligns and adds/subtracts the third operand with
//! a sticky bit, and rounds once to nearest-even.

use crate::emit::Emitter;
use crate::locals::Local;
use crate::{Result, fixed_type, unsupported};
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{Inst, Opcode, Type, types};
use wasm_encoder::{BlockType, Ieee32, Ieee64, Instruction as I, ValType};

/// The instruction set for lowering a [`FloatCC`] comparison on a specific
/// scalar or vector float type.
pub(crate) struct FloatCmpOps {
    pub vt: ValType,
    pub eq: I<'static>,
    pub ne: I<'static>,
    pub lt: I<'static>,
    pub le: I<'static>,
    pub gt: I<'static>,
    pub ge: I<'static>,
    pub and: I<'static>,
    pub or: I<'static>,
    pub not: I<'static>,
}

impl FloatCmpOps {
    pub const F32: Self = Self {
        vt: ValType::F32,
        eq: I::F32Eq,
        ne: I::F32Ne,
        lt: I::F32Lt,
        le: I::F32Le,
        gt: I::F32Gt,
        ge: I::F32Ge,
        and: I::I32And,
        or: I::I32Or,
        not: I::I32Eqz,
    };

    pub const F64: Self = Self {
        vt: ValType::F64,
        eq: I::F64Eq,
        ne: I::F64Ne,
        lt: I::F64Lt,
        le: I::F64Le,
        gt: I::F64Gt,
        ge: I::F64Ge,
        and: I::I32And,
        or: I::I32Or,
        not: I::I32Eqz,
    };

    pub const F32X4: Self = Self {
        vt: ValType::V128,
        eq: I::F32x4Eq,
        ne: I::F32x4Ne,
        lt: I::F32x4Lt,
        le: I::F32x4Le,
        gt: I::F32x4Gt,
        ge: I::F32x4Ge,
        and: I::V128And,
        or: I::V128Or,
        not: I::V128Not,
    };

    pub const F64X2: Self = Self {
        vt: ValType::V128,
        eq: I::F64x2Eq,
        ne: I::F64x2Ne,
        lt: I::F64x2Lt,
        le: I::F64x2Le,
        gt: I::F64x2Gt,
        ge: I::F64x2Ge,
        and: I::V128And,
        or: I::V128Or,
        not: I::V128Not,
    };
}

impl Emitter<'_> {
    /// Push the next operand, a float, widening an `f16` to `f32`. Returns
    /// the type of the pushed value.
    pub(crate) fn float_operand(&mut self) -> Result<Type> {
        let v = self.operand()?;
        match self.ty(v) {
            types::F16 => {
                self.f16_to_f32();
                Ok(types::F32)
            }
            t @ (types::F32 | types::F64) => Ok(t),
            t if fixed_type(t).is_vector() => Ok(t),
            t => unsupported(format!("float operand of type {t}")),
        }
    }

    /// Lower a floating-point comparison (scalar or vector) using `ops`.
    pub(crate) fn emit_fcmp(&mut self, cond: FloatCC, ops: FloatCmpOps) -> Result<()> {
        use FloatCC::*;
        // Unordered-or-X is the negation of the ordered inverse.
        let simple = match cond {
            Equal => Some((ops.eq.clone(), false)),
            NotEqual => Some((ops.ne.clone(), false)),
            LessThan => Some((ops.lt.clone(), false)),
            LessThanOrEqual => Some((ops.le.clone(), false)),
            GreaterThan => Some((ops.gt.clone(), false)),
            GreaterThanOrEqual => Some((ops.ge.clone(), false)),
            UnorderedOrLessThan => Some((ops.ge.clone(), true)),
            UnorderedOrLessThanOrEqual => Some((ops.gt.clone(), true)),
            UnorderedOrGreaterThan => Some((ops.le.clone(), true)),
            UnorderedOrGreaterThanOrEqual => Some((ops.lt.clone(), true)),
            _ => None,
        };
        if let Some((i, negate)) = simple {
            self.float_operand()?;
            self.float_operand()?;
            self.ins(i);
            if negate {
                self.ins(ops.not);
            }
            return Ok(());
        }
        let [sa, sb] = self.scratches(ops.vt);
        match cond {
            Ordered | Unordered => {
                // a == a (and|or) b == b, or the negation.
                let (cmp, comb) = if cond == Ordered {
                    (ops.eq, ops.and)
                } else {
                    (ops.ne, ops.or)
                };
                self.float_operand()?;
                self.local_tee(sa);
                self.local_get(sa);
                self.ins(cmp.clone());
                self.float_operand()?;
                self.local_tee(sb);
                self.local_get(sb);
                self.ins(cmp);
                self.ins(comb);
            }
            OrderedNotEqual | UnorderedOrEqual => {
                self.float_operand()?;
                self.local_tee(sa);
                self.float_operand()?;
                self.local_tee(sb);
                self.ins(ops.lt);
                self.local_get(sa);
                self.local_get(sb);
                self.ins(ops.gt);
                self.ins(ops.or);
                if cond == UnorderedOrEqual {
                    self.ins(ops.not);
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Lower an instruction with an `f16` result or operand, if it needs
    /// anything beyond the generic lowering. Returns whether it did.
    pub(crate) fn lower_f16(&mut self, inst: Inst) -> Result<bool> {
        let dfg = &self.func.dfg;
        let op = dfg.insts[inst].opcode();
        let ty = dfg
            .inst_results(inst)
            .first()
            .map_or(types::INVALID, |&r| dfg.value_type(r));
        let aty = dfg
            .inst_args(inst)
            .first()
            .map_or(types::INVALID, |&a| dfg.value_type(a));
        match op {
            Opcode::Fadd | Opcode::Fsub | Opcode::Fmul | Opcode::Fdiv | Opcode::Fmin
            | Opcode::Fmax => {
                self.float_operand()?;
                self.float_operand()?;
                self.ins(match op {
                    Opcode::Fadd => I::F32Add,
                    Opcode::Fsub => I::F32Sub,
                    Opcode::Fmul => I::F32Mul,
                    Opcode::Fdiv => I::F32Div,
                    Opcode::Fmin => I::F32Min,
                    _ => I::F32Max,
                });
                self.f32_to_f16();
            }
            Opcode::Sqrt | Opcode::Ceil | Opcode::Floor | Opcode::Trunc | Opcode::Nearest => {
                self.float_operand()?;
                self.ins(match op {
                    Opcode::Sqrt => I::F32Sqrt,
                    Opcode::Ceil => I::F32Ceil,
                    Opcode::Floor => I::F32Floor,
                    Opcode::Trunc => I::F32Trunc,
                    _ => I::F32Nearest,
                });
                self.f32_to_f16();
            }
            Opcode::Fneg => {
                self.operand()?;
                self.ins(I::I32Const(0x8000));
                self.ins(I::I32Xor);
            }
            Opcode::Fabs => {
                self.operand()?;
                self.ins(I::I32Const(0x7fff));
                self.ins(I::I32And);
            }
            Opcode::Fcopysign => {
                self.operand()?;
                self.ins(I::I32Const(0x7fff));
                self.ins(I::I32And);
                self.operand()?;
                self.ins(I::I32Const(0x8000));
                self.ins(I::I32And);
                self.ins(I::I32Or);
            }
            Opcode::Fpromote => {
                self.float_operand()?;
                if ty == types::F64 {
                    self.ins(I::F64PromoteF32);
                }
            }
            Opcode::Fdemote => {
                self.operand()?;
                match aty {
                    types::F32 => self.f32_to_f16(),
                    types::F64 => self.f64_to_f16(),
                    _ => return unsupported(format!("fdemote {aty} -> {ty}")),
                }
            }
            Opcode::FcvtFromSint | Opcode::FcvtFromUint if ty == types::F16 => {
                // Through f64, which holds any i32 exactly. An i64 that f64
                // rounds is far beyond the f16 range, so the result is
                // infinity either way.
                let signed = op == Opcode::FcvtFromSint;
                self.operand()?;
                if signed {
                    self.sext(aty);
                }
                self.ins(match (aty == types::I64, signed) {
                    (false, true) => I::F64ConvertI32S,
                    (false, false) => I::F64ConvertI32U,
                    (true, true) => I::F64ConvertI64S,
                    (true, false) => I::F64ConvertI64U,
                });
                self.f64_to_f16();
            }
            Opcode::Fma if ty == types::F16 => {
                self.fma_f64(&mut |e, _| {
                    e.float_operand()?;
                    e.ins(I::F64PromoteF32);
                    Ok(())
                })?;
                self.f64_to_f16();
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Convert the `f16` bits on the stack to an `f32`.
    pub(crate) fn f16_to_f32(&mut self) {
        let mark = self.scratch_mark();
        let h = self.scratch(ValType::I32);
        self.local_tee(h);
        // Finite values: placing the exponent and fraction bits in an f32
        // gives the value scaled by 2^-112 (subnormals included), and
        // scaling back is exact.
        self.ins(I::I32Const(0x7fff));
        self.ins(I::I32And);
        self.ins(I::I32Const(13));
        self.ins(I::I32Shl);
        self.ins(I::F32ReinterpretI32);
        self.ins(I::F32Const(Ieee32::new(0x7780_0000))); // 2^112
        self.ins(I::F32Mul);
        // Infinities and NaNs: the maximum exponent, keeping the payload.
        self.local_get(h);
        self.ins(I::I32Const(0x3ff));
        self.ins(I::I32And);
        self.ins(I::I32Const(13));
        self.ins(I::I32Shl);
        self.ins(I::I32Const(0x7f80_0000));
        self.ins(I::I32Or);
        self.ins(I::F32ReinterpretI32);
        self.local_get(h);
        self.ins(I::I32Const(0x7c00));
        self.ins(I::I32And);
        self.ins(I::I32Const(0x7c00));
        self.ins(I::I32Ne);
        self.ins(I::Select);
        // The sign, from bit 15.
        self.local_get(h);
        self.ins(I::I32Const(16));
        self.ins(I::I32Shl);
        self.ins(I::F32ReinterpretI32);
        self.ins(I::F32Copysign);
        self.free_scratches(mark);
    }

    /// Round the `f32` on the stack to `f16` bits.
    pub(crate) fn f32_to_f16(&mut self) {
        // Exact, so this rounds once.
        self.ins(I::F64PromoteF32);
        self.f64_to_f16();
    }

    /// Round the `f64` on the stack to `f16` bits, to nearest with ties to
    /// even.
    pub(crate) fn f64_to_f16(&mut self) {
        let [b, e, sh, full, bits] = self.scratches(ValType::I64);
        let k = |s: &mut Self, v: i64| s.ins(I::I64Const(v));
        self.ins(I::I64ReinterpretF64);
        self.local_set(b);
        // e: the f16 biased exponent, ((b >> 52) & 0x7ff) - 1023 + 15.
        self.local_get(b);
        k(self, 52);
        self.ins(I::I64ShrU);
        k(self, 0x7ff);
        self.ins(I::I64And);
        k(self, 1023 - 15);
        self.ins(I::I64Sub);
        self.local_set(e);
        // full: the 53-bit significand with its implicit bit. (For f64
        // subnormals the bit is wrong, but they round to zero anyway.)
        self.local_get(b);
        k(self, (1 << 52) - 1);
        self.ins(I::I64And);
        k(self, 1 << 52);
        self.ins(I::I64Or);
        self.local_set(full);
        // sh: how far to shift `full` down to the f16's 10 or fewer
        // fraction bits: 42, plus 1 - e for subnormals, at most 54 (which
        // leaves nothing, and a remainder below half).
        k(self, 1);
        self.local_get(e);
        self.ins(I::I64Sub);
        self.max0(sh);
        k(self, 42);
        self.ins(I::I64Add);
        self.local_tee(sh);
        k(self, 54);
        self.local_get(sh);
        k(self, 54);
        self.ins(I::I64LtS);
        self.ins(I::Select);
        self.local_set(sh);
        // bits = (max(e - 1, 0) << 10) + (full >> sh): for normal results
        // the implicit bit carries into the exponent field.
        self.local_get(e);
        k(self, 1);
        self.ins(I::I64Sub);
        self.max0(bits);
        k(self, 10);
        self.ins(I::I64Shl);
        self.local_get(full);
        self.local_get(sh);
        self.ins(I::I64ShrU);
        self.ins(I::I64Add);
        self.local_set(bits);
        // Round to nearest, ties to even: add 1 if remainder + (bits & 1)
        // exceeds half. A carry may reach the exponent, or infinity.
        self.local_get(bits);
        self.local_get(full);
        k(self, 1);
        self.local_get(sh);
        self.ins(I::I64Shl);
        k(self, 1);
        self.ins(I::I64Sub);
        self.ins(I::I64And);
        self.local_get(bits);
        k(self, 1);
        self.ins(I::I64And);
        self.ins(I::I64Add);
        k(self, 1);
        self.local_get(sh);
        k(self, 1);
        self.ins(I::I64Sub);
        self.ins(I::I64Shl);
        self.ins(I::I64GtU);
        self.ins(I::I64ExtendI32U);
        self.ins(I::I64Add);
        self.local_set(bits);
        // Too large: infinity.
        self.local_get(bits);
        k(self, 0x7c00);
        self.local_get(e);
        k(self, 31);
        self.ins(I::I64LtS);
        self.ins(I::Select);
        self.local_set(bits);
        // Infinity or NaN input (e == 0x7ff - 1008): keep the top of a NaN
        // payload, made quiet.
        k(self, 0x7c00);
        self.local_get(full);
        k(self, 1 << 52);
        self.ins(I::I64Xor);
        k(self, 42);
        self.ins(I::I64ShrU);
        k(self, 0x200);
        self.ins(I::I64Or);
        k(self, 0);
        self.local_get(full);
        k(self, 1 << 52);
        self.ins(I::I64Ne);
        self.ins(I::Select);
        self.ins(I::I64Or);
        self.local_get(bits);
        self.local_get(e);
        k(self, 0x7ff - (1023 - 15));
        self.ins(I::I64Eq);
        self.ins(I::Select);
        // The sign.
        self.local_get(b);
        k(self, 48);
        self.ins(I::I64ShrU);
        k(self, 0x8000);
        self.ins(I::I64And);
        self.ins(I::I64Or);
        self.ins(I::I32WrapI64);
    }

    /// max(x, 0) of the `i64` on the stack, using `s` as scratch.
    fn max0(&mut self, s: Local) {
        self.local_tee(s);
        self.ins(I::I64Const(0));
        self.local_get(s);
        self.ins(I::I64Const(0));
        self.ins(I::I64GtS);
        self.ins(I::Select);
    }

    /// Push `x * y + z` for operands whose product is exact in `f64` (from
    /// `f32` or `f16`), rounded to odd, as an `f64`. `push(e, i)` pushes the
    /// `i`th operand converted to `f64`.
    pub(crate) fn fma_f64(
        &mut self,
        push: &mut dyn FnMut(&mut Self, usize) -> Result<()>,
    ) -> Result<()> {
        let mark = self.scratch_mark();
        let [p, c, s, bb, err] = self.scratches(ValType::F64);
        let sb = self.scratch(ValType::I64);
        push(self, 0)?;
        push(self, 1)?;
        self.ins(I::F64Mul);
        self.local_set(p);
        push(self, 2)?;
        self.local_set(c);
        // s = p + c, rounded to nearest; err = (p + c) - s exactly (2Sum).
        self.local_get(p);
        self.local_get(c);
        self.ins(I::F64Add);
        self.local_set(s);
        self.local_get(s);
        self.local_get(p);
        self.ins(I::F64Sub);
        self.local_set(bb);
        self.local_get(p);
        self.local_get(s);
        self.local_get(bb);
        self.ins(I::F64Sub);
        self.ins(I::F64Sub);
        self.local_get(c);
        self.local_get(bb);
        self.ins(I::F64Sub);
        self.ins(I::F64Add);
        self.local_set(err);
        // Round to odd: if s is inexact and even, the odd neighbor on err's
        // side is the result: one step up in magnitude if err has s's sign,
        // one step down otherwise. Infinities and NaNs stay.
        self.local_get(s);
        self.ins(I::I64ReinterpretF64);
        self.local_tee(sb);
        // step = 1 - 2 * ((bits(err) ^ bits(s)) >>> 63)
        self.ins(I::I64Const(1));
        self.local_get(err);
        self.ins(I::I64ReinterpretF64);
        self.local_get(sb);
        self.ins(I::I64Xor);
        self.ins(I::I64Const(63));
        self.ins(I::I64ShrU);
        self.ins(I::I64Const(1));
        self.ins(I::I64Shl);
        self.ins(I::I64Sub);
        self.ins(I::I64Const(0));
        self.local_get(err);
        self.ins(I::F64Const(Ieee64::new(0)));
        self.ins(I::F64Ne);
        self.local_get(sb);
        self.ins(I::I64Const(1));
        self.ins(I::I64And);
        self.ins(I::I64Eqz);
        self.ins(I::I32And);
        self.local_get(s);
        self.ins(I::F64Abs);
        self.ins(I::F64Const(Ieee64::new(f64::INFINITY.to_bits())));
        self.ins(I::F64Lt);
        self.ins(I::I32And);
        self.ins(I::Select);
        self.ins(I::I64Add);
        self.ins(I::F64ReinterpretI64);
        self.free_scratches(mark);
        Ok(())
    }

    /// `fma` on `f32` vector lanes, one lane at a time.
    pub(crate) fn fma_f32_lanes(&mut self, ty: Type) -> Result<()> {
        let vs = self.scratches::<3>(ValType::V128);
        for &v in &vs {
            self.operand()?;
            self.local_set(v);
        }
        for lane in 0..ty.lane_count() as u8 {
            self.fma_f64(&mut |e, i| {
                e.local_get(vs[i]);
                e.ins(I::F32x4ExtractLane(lane));
                e.ins(I::F64PromoteF32);
                Ok(())
            })?;
            self.ins(I::F32DemoteF64);
            self.ins(if lane == 0 {
                I::F32x4Splat
            } else {
                I::F32x4ReplaceLane(lane)
            });
        }
        Ok(())
    }

    /// `fma` on `f64x2` vector lanes, one lane at a time.
    pub(crate) fn fma_f64_lanes(&mut self) -> Result<()> {
        let vs = self.scratches::<3>(ValType::V128);
        for &v in &vs {
            self.operand()?;
            self.local_set(v);
        }
        for lane in 0..2u8 {
            for &v in &vs {
                self.local_get(v);
                self.ins(I::F64x2ExtractLane(lane));
            }
            self.fma_f64_exact();
            self.ins(if lane == 0 {
                I::F64x2Splat
            } else {
                I::F64x2ReplaceLane(lane)
            });
        }
        Ok(())
    }

    /// Correctly rounded `a * b + c` on three `f64` values already on the
    /// operand stack (`a` bottom, `b` middle, `c` top).
    pub(crate) fn fma_f64_exact(&mut self) {
        let mark = self.scratch_mark();
        let [fa, fb, fc] = self.scratches(ValType::F64);
        let [aa, ba, ca] = self.scratches(ValType::I64);
        self.local_tee(fc);
        self.ins(I::I64ReinterpretF64);
        self.local_set(ca);
        self.local_tee(fb);
        self.ins(I::I64ReinterpretF64);
        self.local_set(ba);
        self.local_tee(fa);
        self.ins(I::I64ReinterpretF64);
        self.local_set(aa);

        // Fast path 1: if a or b is ±0, ±Inf, or NaN, `a * b` has no rounding
        // error, so `(a * b) + c` in standard f64 arithmetic is exact.
        self.is_special_f64(aa);
        self.is_special_f64(ba);
        self.ins(I::I32Or);
        self.ins(I::If(BlockType::Result(ValType::F64)));
        self.local_get(fa);
        self.local_get(fb);
        self.ins(I::F64Mul);
        self.local_get(fc);
        self.ins(I::F64Add);
        self.ins(I::Else);

        // Fast path 2: a and b are finite and non-zero. If c is ±0, the exact
        // real sum is `a * b != 0`, so `f64.mul(a, b)` rounds once with the
        // right sign; if c is ±Inf or NaN, the result is `c + c`.
        self.is_special_f64(ca);
        self.ins(I::If(BlockType::Result(ValType::F64)));
        self.local_get(fa);
        self.local_get(fb);
        self.ins(I::F64Mul);
        self.local_get(fc);
        self.local_get(fc);
        self.ins(I::F64Add);
        self.local_get(ca);
        self.ins(I::I64Const(0x7fff_ffff_ffff_ffff));
        self.ins(I::I64And);
        self.ins(I::I64Eqz);
        self.ins(I::Select);
        self.ins(I::Else);

        // Main path: a, b, and c are all finite and non-zero.
        let (a_sign, a_exp, a_sig) = self.unpack_f64(aa);
        let (b_sign, b_exp, b_sig) = self.unpack_f64(ba);
        let (c_sign, c_exp, c_sig) = self.unpack_f64(ca);

        // Exact 106-bit product p = a_sig * b_sig in [2^104, 2^106), shifted
        // left by 19 or 20 bits so its leading 1-bit sits at bit 124 (bit 60
        // of p_hi).
        let [p_sign, p_exp, p_lo, p_hi, top, sh] = self.scratches(ValType::I64);
        self.local_get(a_sign);
        self.local_get(b_sign);
        self.ins(I::I64Xor);
        self.local_set(p_sign);

        self.local_get(a_sig);
        self.local_get(b_sig);
        self.ins(I::I64Mul);
        self.local_set(p_lo);
        self.mulhi64_of(a_sig, b_sig, false);
        self.local_tee(p_hi);
        self.ins(I::I64Const(41));
        self.ins(I::I64ShrU);
        self.ins(I::I64Const(1));
        self.ins(I::I64And);
        self.local_set(top);

        self.local_get(a_exp);
        self.local_get(b_exp);
        self.ins(I::I64Add);
        self.ins(I::I64Const(1023));
        self.ins(I::I64Sub);
        self.local_get(top);
        self.ins(I::I64Add);
        self.local_set(p_exp);

        self.ins(I::I64Const(20));
        self.local_get(top);
        self.ins(I::I64Sub);
        self.local_set(sh);

        self.local_get(p_hi);
        self.local_get(sh);
        self.ins(I::I64Shl);
        self.local_get(p_lo);
        self.ins(I::I64Const(44));
        self.local_get(top);
        self.ins(I::I64Add);
        self.ins(I::I64ShrU);
        self.ins(I::I64Or);
        self.local_set(p_hi);

        self.local_get(p_lo);
        self.local_get(sh);
        self.ins(I::I64Shl);
        self.local_set(p_lo);

        // Place c's 53-bit significand with its leading 1-bit at bit 124 too
        // (c_hi = c_sig << 8, c_lo = 0).
        let c_hi = c_sig;
        self.local_get(c_sig);
        self.ins(I::I64Const(8));
        self.ins(I::I64Shl);
        self.local_set(c_hi);

        // Order (p, c) into (x, y) so |x| >= |y|. Since c_lo == 0,
        // |p| < |c| iff p_exp < c_exp || (p_exp == c_exp && p_hi < c_hi).
        let swap = self.scratch(ValType::I32);
        self.local_get(p_exp);
        self.local_get(c_exp);
        self.ins(I::I64LtS);
        self.local_get(p_exp);
        self.local_get(c_exp);
        self.ins(I::I64Eq);
        self.local_get(p_hi);
        self.local_get(c_hi);
        self.ins(I::I64LtU);
        self.ins(I::I32And);
        self.ins(I::I32Or);
        self.local_set(swap);

        let [x_sign, y_sign, x_exp, y_exp, x_hi, y_hi, x_lo, y_lo] =
            self.scratches(ValType::I64);
        for (x, y, pv, cv) in [
            (x_sign, y_sign, p_sign, c_sign),
            (x_exp, y_exp, p_exp, c_exp),
            (x_hi, y_hi, p_hi, c_hi),
        ] {
            self.local_get(cv);
            self.local_get(pv);
            self.local_get(swap);
            self.ins(I::Select);
            self.local_set(x);

            self.local_get(pv);
            self.local_get(cv);
            self.local_get(swap);
            self.ins(I::Select);
            self.local_set(y);
        }
        self.ins(I::I64Const(0));
        self.local_get(p_lo);
        self.local_get(swap);
        self.ins(I::Select);
        self.local_set(x_lo);

        self.local_get(p_lo);
        self.ins(I::I64Const(0));
        self.local_get(swap);
        self.ins(I::Select);
        self.local_set(y_lo);

        // Align y to x's exponent, folding shifted-out bits into bit 0 of y_lo.
        let d = self.scratch(ValType::I64);
        self.local_get(x_exp);
        self.local_get(y_exp);
        self.ins(I::I64Sub);
        self.local_set(d);
        self.shr128_sticky(y_lo, y_hi, d);

        // Add (if x_sign == y_sign) or subtract (if x_sign != y_sign).
        let same_sign = self.scratch(ValType::I32);
        let [add_lo, res_lo, res_hi] = self.scratches(ValType::I64);
        self.local_get(x_sign);
        self.local_get(y_sign);
        self.ins(I::I64Eq);
        self.local_set(same_sign);

        self.local_get(x_lo);
        self.local_get(y_lo);
        self.ins(I::I64Add);
        self.local_tee(add_lo);
        self.local_get(x_lo);
        self.local_get(y_lo);
        self.ins(I::I64Sub);
        self.local_get(same_sign);
        self.ins(I::Select);
        self.local_set(res_lo);

        // add_hi = x_hi + y_hi + (add_lo <u x_lo)
        self.local_get(x_hi);
        self.local_get(y_hi);
        self.ins(I::I64Add);
        self.local_get(add_lo);
        self.local_get(x_lo);
        self.ins(I::I64LtU);
        self.ins(I::I64ExtendI32U);
        self.ins(I::I64Add);
        // sub_hi = x_hi - y_hi - (x_lo <u y_lo)
        self.local_get(x_hi);
        self.local_get(y_hi);
        self.ins(I::I64Sub);
        self.local_get(x_lo);
        self.local_get(y_lo);
        self.ins(I::I64LtU);
        self.ins(I::I64ExtendI32U);
        self.ins(I::I64Sub);
        self.local_get(same_sign);
        self.ins(I::Select);
        self.local_set(res_hi);

        // Exact cancellation `x - x` yields +0.0 under round-to-nearest-even.
        self.local_get(res_lo);
        self.local_get(res_hi);
        self.ins(I::I64Or);
        self.ins(I::I64Eqz);
        self.ins(I::If(BlockType::Result(ValType::F64)));
        self.ins(I::F64Const(Ieee64::new(0)));
        self.ins(I::Else);

        // Normalize (res_lo, res_hi) so its MSB is at bit 124 (lz == 3).
        let [lz, res_exp, lsh, sub_sh, exp_base, sig53, bits] = self.scratches(ValType::I64);
        self.local_get(res_hi);
        self.ins(I::I64Clz);
        self.local_get(res_lo);
        self.ins(I::I64Clz);
        self.ins(I::I64Const(64));
        self.ins(I::I64Add);
        self.local_get(res_hi);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.ins(I::Select);
        self.local_set(lz);

        self.local_get(x_exp);
        self.ins(I::I64Const(3));
        self.ins(I::I64Add);
        self.local_get(lz);
        self.ins(I::I64Sub);
        self.local_set(res_exp);

        // If lz == 2 (addition carried into bit 125), shift right by 1 with sticky.
        self.local_get(lz);
        self.ins(I::I64Const(2));
        self.ins(I::I64Eq);
        self.ins(I::I64ExtendI32U);
        self.local_set(d);
        self.shr128_sticky(res_lo, res_hi, d);

        // If lz > 3 (cancellation), shift left by lz - 3.
        self.local_get(lz);
        self.ins(I::I64Const(3));
        self.ins(I::I64Sub);
        self.max0(lsh);
        self.local_set(lsh);
        self.shl128(res_lo, res_hi, lsh);

        // Denormalize if res_exp <= 0: shift right by 1 - res_exp with sticky.
        self.ins(I::I64Const(1));
        self.local_get(res_exp);
        self.ins(I::I64Sub);
        self.max0(sub_sh);
        self.local_set(sub_sh);
        self.shr128_sticky(res_lo, res_hi, sub_sh);

        self.local_get(res_exp);
        self.ins(I::I64Const(1));
        self.ins(I::I64Sub);
        self.max0(exp_base);
        self.local_set(exp_base);

        // Extract 53-bit significand (bits 8..=60 of res_hi), guard bit (bit 7),
        // and sticky bit (bits 0..=6 of res_hi or any bit of res_lo).
        self.local_get(res_hi);
        self.ins(I::I64Const(8));
        self.ins(I::I64ShrU);
        self.local_set(sig53);

        self.local_get(exp_base);
        self.ins(I::I64Const(52));
        self.ins(I::I64Shl);
        self.local_get(sig53);
        self.ins(I::I64Add);
        // + (guard & (sticky | (sig53 & 1)))
        self.local_get(res_hi);
        self.ins(I::I64Const(7));
        self.ins(I::I64ShrU);
        self.ins(I::I64Const(1));
        self.ins(I::I64And);
        self.local_get(res_hi);
        self.ins(I::I64Const(0x7f));
        self.ins(I::I64And);
        self.local_get(res_lo);
        self.ins(I::I64Or);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.ins(I::I64ExtendI32U);
        self.local_get(sig53);
        self.ins(I::I64Const(1));
        self.ins(I::I64And);
        self.ins(I::I64Or);
        self.ins(I::I64And);
        self.ins(I::I64Add);
        self.local_set(bits);

        // Overflow to +Inf (0x7ff0_0000_0000_0000).
        self.ins(I::I64Const(0x7ff0_0000_0000_0000));
        self.local_get(bits);
        self.local_get(res_exp);
        self.ins(I::I64Const(2047));
        self.ins(I::I64GeS);
        self.local_get(bits);
        self.ins(I::I64Const(0x7ff0_0000_0000_0000));
        self.ins(I::I64GeU);
        self.ins(I::I32Or);
        self.ins(I::Select);
        self.local_get(x_sign);
        self.ins(I::I64Or);
        self.ins(I::F64ReinterpretI64);

        self.ins(I::End);
        self.ins(I::End);
        self.ins(I::End);
        self.free_scratches(mark);
    }

    /// Push `1` (`i32`) if the `f64` bit pattern in `bits` is `±0`, `±Inf`, or `NaN`.
    fn is_special_f64(&mut self, bits: Local) {
        self.local_get(bits);
        self.ins(I::I64Const(0x7fff_ffff_ffff_ffff));
        self.ins(I::I64And);
        self.ins(I::I64Const(1));
        self.ins(I::I64Sub);
        self.ins(I::I64Const(0x7fef_ffff_ffff_ffff));
        self.ins(I::I64GeU);
    }

    /// Unpack a finite non-zero `f64` bit pattern into `(sign, exp, sig)` locals,
    /// where `sign` is `bits & (1 << 63)`, `sig` is normalized in `[2^52, 2^53)`,
    /// and `exp` is the biased exponent (`1023` for `1.0`).
    fn unpack_f64(&mut self, bits: Local) -> (Local, Local, Local) {
        let [sign, exp, sig, is_norm, sh] = self.scratches(ValType::I64);
        self.local_get(bits);
        self.ins(I::I64Const(i64::MIN)); // 0x8000_0000_0000_0000
        self.ins(I::I64And);
        self.local_set(sign);

        self.local_get(bits);
        self.ins(I::I64Const(52));
        self.ins(I::I64ShrU);
        self.ins(I::I64Const(0x7ff));
        self.ins(I::I64And);
        self.local_tee(exp);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.ins(I::I64ExtendI32U);
        self.local_set(is_norm);

        self.local_get(bits);
        self.ins(I::I64Const((1 << 52) - 1));
        self.ins(I::I64And);
        self.local_get(is_norm);
        self.ins(I::I64Const(52));
        self.ins(I::I64Shl);
        self.ins(I::I64Or);
        self.local_tee(sig);
        self.ins(I::I64Clz);
        self.ins(I::I64Const(11));
        self.ins(I::I64Sub);
        self.local_set(sh);

        self.local_get(sig);
        self.local_get(sh);
        self.ins(I::I64Shl);
        self.local_set(sig);

        self.local_get(exp);
        self.ins(I::I64Const(1));
        self.ins(I::I64Add);
        self.local_get(is_norm);
        self.ins(I::I64Sub);
        self.local_get(sh);
        self.ins(I::I64Sub);
        self.local_set(exp);

        (sign, exp, sig)
    }

    /// Shift `(lo, hi)` right in place by `d >= 0` bits, OR-ing any shifted-out
    /// bits into bit 0 of `lo`. Assumes the MSB of `(lo, hi)` is at most bit 125.
    fn shr128_sticky(&mut self, lo: Local, hi: Local, d: Local) {
        let [sh, inv] = self.scratches(ValType::I64);
        let [ge64, nz] = self.scratches(ValType::I32);

        // Clamp d to at most 127.
        self.local_get(d);
        self.ins(I::I64Const(127));
        self.local_get(d);
        self.ins(I::I64Const(127));
        self.ins(I::I64LtS);
        self.ins(I::Select);
        self.local_tee(sh);
        self.ins(I::I64Const(64));
        self.ins(I::I64GeU);
        self.local_set(ge64);

        // Step 1: if sh >= 64, shift right by 64 with sticky.
        self.local_get(hi);
        self.local_get(lo);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.ins(I::I64ExtendI32U);
        self.ins(I::I64Or);
        self.local_get(lo);
        self.local_get(ge64);
        self.ins(I::Select);
        self.local_set(lo);

        self.ins(I::I64Const(0));
        self.local_get(hi);
        self.local_get(ge64);
        self.ins(I::Select);
        self.local_set(hi);

        // Step 2: shift right by (sh & 63) in 0..=63 with sticky.
        self.local_get(sh);
        self.ins(I::I64Const(63));
        self.ins(I::I64And);
        self.local_tee(sh);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.local_set(nz);

        self.ins(I::I64Const(64));
        self.local_get(sh);
        self.ins(I::I64Sub);
        self.local_set(inv);

        self.local_get(lo);
        self.local_get(sh);
        self.ins(I::I64ShrU);
        self.local_get(hi);
        self.local_get(inv);
        self.ins(I::I64Shl);
        self.ins(I::I64Or);
        self.local_get(lo);
        self.local_get(inv);
        self.ins(I::I64Shl);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.ins(I::I64ExtendI32U);
        self.ins(I::I64Or);
        self.local_get(lo);
        self.local_get(nz);
        self.ins(I::Select);
        self.local_set(lo);

        self.local_get(hi);
        self.local_get(sh);
        self.ins(I::I64ShrU);
        self.local_get(hi);
        self.local_get(nz);
        self.ins(I::Select);
        self.local_set(hi);
    }

    /// Shift `(lo, hi)` left in place by `d` in `0..=127` bits.
    fn shl128(&mut self, lo: Local, hi: Local, d: Local) {
        let [sh, inv] = self.scratches(ValType::I64);
        let [ge64, nz] = self.scratches(ValType::I32);

        self.local_get(d);
        self.ins(I::I64Const(64));
        self.ins(I::I64GeU);
        self.local_set(ge64);

        // Step 1: if d >= 64, shift left by 64.
        self.local_get(lo);
        self.local_get(hi);
        self.local_get(ge64);
        self.ins(I::Select);
        self.local_set(hi);

        self.ins(I::I64Const(0));
        self.local_get(lo);
        self.local_get(ge64);
        self.ins(I::Select);
        self.local_set(lo);

        // Step 2: shift left by (d & 63) in 0..=63.
        self.local_get(d);
        self.ins(I::I64Const(63));
        self.ins(I::I64And);
        self.local_tee(sh);
        self.ins(I::I64Const(0));
        self.ins(I::I64Ne);
        self.local_set(nz);

        self.ins(I::I64Const(64));
        self.local_get(sh);
        self.ins(I::I64Sub);
        self.local_set(inv);

        self.local_get(hi);
        self.local_get(sh);
        self.ins(I::I64Shl);
        self.local_get(lo);
        self.local_get(inv);
        self.ins(I::I64ShrU);
        self.ins(I::I64Or);
        self.local_get(hi);
        self.local_get(nz);
        self.ins(I::Select);
        self.local_set(hi);

        self.local_get(lo);
        self.local_get(sh);
        self.ins(I::I64Shl);
        self.local_get(lo);
        self.local_get(nz);
        self.ins(I::Select);
        self.local_set(lo);
    }
}
