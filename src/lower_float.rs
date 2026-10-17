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
//! rounded sums: proved algorithms using rounding to odd", 2008). `f64`
//! `fma` would need a wider type and is not supported.

use crate::emit::Emitter;
use crate::{Result, fixed_type, unsupported};
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{Inst, Opcode, Type, types};
use wasm_encoder::{Ieee32, Ieee64, Instruction as I, ValType};

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
                self.ins(I::LocalTee(sa));
                self.ins(I::LocalGet(sa));
                self.ins(cmp.clone());
                self.float_operand()?;
                self.ins(I::LocalTee(sb));
                self.ins(I::LocalGet(sb));
                self.ins(cmp);
                self.ins(comb);
            }
            OrderedNotEqual | UnorderedOrEqual => {
                self.float_operand()?;
                self.ins(I::LocalTee(sa));
                self.float_operand()?;
                self.ins(I::LocalTee(sb));
                self.ins(ops.lt);
                self.ins(I::LocalGet(sa));
                self.ins(I::LocalGet(sb));
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
        self.ins(I::LocalTee(h));
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
        self.ins(I::LocalGet(h));
        self.ins(I::I32Const(0x3ff));
        self.ins(I::I32And);
        self.ins(I::I32Const(13));
        self.ins(I::I32Shl);
        self.ins(I::I32Const(0x7f80_0000));
        self.ins(I::I32Or);
        self.ins(I::F32ReinterpretI32);
        self.ins(I::LocalGet(h));
        self.ins(I::I32Const(0x7c00));
        self.ins(I::I32And);
        self.ins(I::I32Const(0x7c00));
        self.ins(I::I32Ne);
        self.ins(I::Select);
        // The sign, from bit 15.
        self.ins(I::LocalGet(h));
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
        let get = |s: &mut Self, l: u32| s.ins(I::LocalGet(l));
        let k = |s: &mut Self, v: i64| s.ins(I::I64Const(v));
        self.ins(I::I64ReinterpretF64);
        self.ins(I::LocalSet(b));
        // e: the f16 biased exponent, ((b >> 52) & 0x7ff) - 1023 + 15.
        get(self, b);
        k(self, 52);
        self.ins(I::I64ShrU);
        k(self, 0x7ff);
        self.ins(I::I64And);
        k(self, 1023 - 15);
        self.ins(I::I64Sub);
        self.ins(I::LocalSet(e));
        // full: the 53-bit significand with its implicit bit. (For f64
        // subnormals the bit is wrong, but they round to zero anyway.)
        get(self, b);
        k(self, (1 << 52) - 1);
        self.ins(I::I64And);
        k(self, 1 << 52);
        self.ins(I::I64Or);
        self.ins(I::LocalSet(full));
        // sh: how far to shift `full` down to the f16's 10 or fewer
        // fraction bits: 42, plus 1 - e for subnormals, at most 54 (which
        // leaves nothing, and a remainder below half).
        k(self, 1);
        get(self, e);
        self.ins(I::I64Sub);
        self.max0(sh);
        k(self, 42);
        self.ins(I::I64Add);
        self.ins(I::LocalTee(sh));
        k(self, 54);
        get(self, sh);
        k(self, 54);
        self.ins(I::I64LtS);
        self.ins(I::Select);
        self.ins(I::LocalSet(sh));
        // bits = (max(e - 1, 0) << 10) + (full >> sh): for normal results
        // the implicit bit carries into the exponent field.
        get(self, e);
        k(self, 1);
        self.ins(I::I64Sub);
        self.max0(bits);
        k(self, 10);
        self.ins(I::I64Shl);
        get(self, full);
        get(self, sh);
        self.ins(I::I64ShrU);
        self.ins(I::I64Add);
        self.ins(I::LocalSet(bits));
        // Round to nearest, ties to even: add 1 if remainder + (bits & 1)
        // exceeds half. A carry may reach the exponent, or infinity.
        get(self, bits);
        get(self, full);
        k(self, 1);
        get(self, sh);
        self.ins(I::I64Shl);
        k(self, 1);
        self.ins(I::I64Sub);
        self.ins(I::I64And);
        get(self, bits);
        k(self, 1);
        self.ins(I::I64And);
        self.ins(I::I64Add);
        k(self, 1);
        get(self, sh);
        k(self, 1);
        self.ins(I::I64Sub);
        self.ins(I::I64Shl);
        self.ins(I::I64GtU);
        self.ins(I::I64ExtendI32U);
        self.ins(I::I64Add);
        self.ins(I::LocalSet(bits));
        // Too large: infinity.
        get(self, bits);
        k(self, 0x7c00);
        get(self, e);
        k(self, 31);
        self.ins(I::I64LtS);
        self.ins(I::Select);
        self.ins(I::LocalSet(bits));
        // Infinity or NaN input (e == 0x7ff - 1008): keep the top of a NaN
        // payload, made quiet.
        k(self, 0x7c00);
        get(self, full);
        k(self, 1 << 52);
        self.ins(I::I64Xor);
        k(self, 42);
        self.ins(I::I64ShrU);
        k(self, 0x200);
        self.ins(I::I64Or);
        k(self, 0);
        get(self, full);
        k(self, 1 << 52);
        self.ins(I::I64Ne);
        self.ins(I::Select);
        self.ins(I::I64Or);
        get(self, bits);
        get(self, e);
        k(self, 0x7ff - (1023 - 15));
        self.ins(I::I64Eq);
        self.ins(I::Select);
        // The sign.
        get(self, b);
        k(self, 48);
        self.ins(I::I64ShrU);
        k(self, 0x8000);
        self.ins(I::I64And);
        self.ins(I::I64Or);
        self.ins(I::I32WrapI64);
    }

    /// max(x, 0) of the `i64` on the stack, using `s` as scratch.
    fn max0(&mut self, s: u32) {
        self.ins(I::LocalTee(s));
        self.ins(I::I64Const(0));
        self.ins(I::LocalGet(s));
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
        let get = |e: &mut Self, l: u32| e.ins(I::LocalGet(l));
        push(self, 0)?;
        push(self, 1)?;
        self.ins(I::F64Mul);
        self.ins(I::LocalSet(p));
        push(self, 2)?;
        self.ins(I::LocalSet(c));
        // s = p + c, rounded to nearest; err = (p + c) - s exactly (2Sum).
        get(self, p);
        get(self, c);
        self.ins(I::F64Add);
        self.ins(I::LocalSet(s));
        get(self, s);
        get(self, p);
        self.ins(I::F64Sub);
        self.ins(I::LocalSet(bb));
        get(self, p);
        get(self, s);
        get(self, bb);
        self.ins(I::F64Sub);
        self.ins(I::F64Sub);
        get(self, c);
        get(self, bb);
        self.ins(I::F64Sub);
        self.ins(I::F64Add);
        self.ins(I::LocalSet(err));
        // Round to odd: if s is inexact and even, the odd neighbor on err's
        // side is the result: one step up in magnitude if err has s's sign,
        // one step down otherwise. Infinities and NaNs stay.
        get(self, s);
        self.ins(I::I64ReinterpretF64);
        self.ins(I::LocalTee(sb));
        // step = 1 - 2 * ((bits(err) ^ bits(s)) >>> 63)
        self.ins(I::I64Const(1));
        get(self, err);
        self.ins(I::I64ReinterpretF64);
        get(self, sb);
        self.ins(I::I64Xor);
        self.ins(I::I64Const(63));
        self.ins(I::I64ShrU);
        self.ins(I::I64Const(1));
        self.ins(I::I64Shl);
        self.ins(I::I64Sub);
        self.ins(I::I64Const(0));
        get(self, err);
        self.ins(I::F64Const(Ieee64::new(0)));
        self.ins(I::F64Ne);
        get(self, sb);
        self.ins(I::I64Const(1));
        self.ins(I::I64And);
        self.ins(I::I64Eqz);
        self.ins(I::I32And);
        get(self, s);
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
            self.ins(I::LocalSet(v));
        }
        for lane in 0..ty.lane_count() as u8 {
            self.fma_f64(&mut |e, i| {
                e.ins(I::LocalGet(vs[i]));
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
}
