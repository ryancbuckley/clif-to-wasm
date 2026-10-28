//! Lowering of vector instructions to wasm SIMD.
//!
//! Most Cranelift vector operations map directly to wasm SIMD. Operations
//! without a direct counterpart are emulated:
//!
//! - `imul.i8x16` and `umulhi`/`smulhi`: widening multiplies (`extmul`) of
//!   the low and high halves, followed by a shuffle picking the low or high
//!   half of each product.
//! - `iadd_pairwise`: shuffles gathering the even and odd lanes of both inputs,
//!   followed by an add.
//! - `i64x2` unsigned comparisons and min/max: flip the sign bits and use the
//!   signed comparison; min/max then `bitselect` on the comparison mask.
//! - `uunarrow`: clamp each lane with an unsigned min before narrowing.
//! - `popcnt` on 16-, 32-, and 64-bit lanes: byte popcounts summed pairwise.
//!
//! Vectors narrower than 128 bits (`i32x2`, `i8x4`, ...) occupy the low bytes
//! of a `v128` with upper bytes unspecified. Lane-wise operations ignore the
//! upper bytes; cross-lane operations (`vall_true`, `vany_true`, `vhigh_bits`,
//! `iadd_pairwise`, `*widen_high`, and narrowing ops), loads, and stores
//! account for the active lane count.

use crate::emit::Emitter;
use crate::locals::Local;
use crate::lower_float::FloatCmpOps;
use crate::{Result, error, wasm_types};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{Function, Inst, InstructionData as D, Opcode, Type, types};
use wasm_encoder::{Instruction as I, MemArg, ValType};

/// The lane type of a 128-bit vector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lane {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Lane {
    fn of(ty: Type) -> Result<Lane> {
        if !ty.is_vector() || !(16..=128).contains(&ty.bits()) {
            return error::unsupported(format!("vector type {ty}"));
        }
        Self::from_scalar(ty.lane_type())
            .ok_or_else(|| crate::Error::Unsupported(format!("vector type {ty}")))
    }

    fn from_scalar(ty: Type) -> Option<Lane> {
        Some(match ty {
            types::I8 => Lane::I8,
            // f16 lanes can be moved around like i16 lanes; arithmetic
            // on them is rejected below as on any integer lane.
            types::I16 | types::F16 => Lane::I16,
            types::I32 => Lane::I32,
            types::I64 => Lane::I64,
            types::F32 => Lane::F32,
            types::F64 => Lane::F64,
            _ => return None,
        })
    }

    fn bytes(self) -> u32 {
        match self {
            Lane::I8 => 1,
            Lane::I16 => 2,
            Lane::I32 | Lane::F32 => 4,
            Lane::I64 | Lane::F64 => 8,
        }
    }

    /// The integer lane type of the same width.
    fn int(self) -> Lane {
        match self {
            Lane::F32 => Lane::I32,
            Lane::F64 => Lane::I64,
            l => l,
        }
    }

    fn splat(self) -> I<'static> {
        match self {
            Lane::I8 => I::I8x16Splat,
            Lane::I16 => I::I16x8Splat,
            Lane::I32 => I::I32x4Splat,
            Lane::I64 => I::I64x2Splat,
            Lane::F32 => I::F32x4Splat,
            Lane::F64 => I::F64x2Splat,
        }
    }

    /// Extract lane `idx`, zero-extending narrow integer lanes to match the
    /// scalar representation of `i8`/`i16`.
    fn extract_lane(self, idx: u8) -> I<'static> {
        match self {
            Lane::I8 => I::I8x16ExtractLaneU(idx),
            Lane::I16 => I::I16x8ExtractLaneU(idx),
            Lane::I32 => I::I32x4ExtractLane(idx),
            Lane::I64 => I::I64x2ExtractLane(idx),
            Lane::F32 => I::F32x4ExtractLane(idx),
            Lane::F64 => I::F64x2ExtractLane(idx),
        }
    }

    fn replace_lane(self, idx: u8) -> I<'static> {
        match self {
            Lane::I8 => I::I8x16ReplaceLane(idx),
            Lane::I16 => I::I16x8ReplaceLane(idx),
            Lane::I32 => I::I32x4ReplaceLane(idx),
            Lane::I64 => I::I64x2ReplaceLane(idx),
            Lane::F32 => I::F32x4ReplaceLane(idx),
            Lane::F64 => I::F64x2ReplaceLane(idx),
        }
    }
}

/// Pick a per-lane instruction; `None` entries do not exist in wasm.
macro_rules! lanes {
    ($lane:expr, $op:expr; $($l:ident => $i:expr),* $(,)?) => {
        match $lane {
            $(Lane::$l => $i,)*
            #[allow(unreachable_patterns)]
            l => return error::unsupported(format!("{} on {:?} lanes", $op, l)),
        }
    };
}

/// Does this instruction operate on vectors (and so belong here)?
pub(crate) fn is_simd(func: &Function, inst: Inst) -> bool {
    let dfg = &func.dfg;
    // Almost every vector operation has a vector result; check the
    // vector-in, scalar-out ones by their operand.
    match dfg.inst_results(inst).first() {
        Some(&r) if wasm_types::fixed_type(dfg.value_type(r)).is_vector() => {
            let op = dfg.insts[inst].opcode();
            !op.is_call()
        }
        _ => match dfg.insts[inst].opcode() {
            Opcode::Extractlane
            | Opcode::VanyTrue
            | Opcode::VallTrue
            | Opcode::VhighBits
            | Opcode::Store
            | Opcode::Bitcast => {
                wasm_types::fixed_type(dfg.value_type(dfg.inst_args(inst)[0])).is_vector()
            }
            _ => false,
        },
    }
}

/// Byte indices for `i8x16.shuffle` that build result lane `i` (of
/// `lane_bytes` bytes) from lane `pick(i)` of the concatenated inputs.
fn shuffle_bytes(lane_bytes: u32, pick: impl Fn(u32) -> u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 / lane_bytes {
        for k in 0..lane_bytes {
            out[(i * lane_bytes + k) as usize] = (pick(i) * lane_bytes + k) as u8;
        }
    }
    out
}

/// A `v128` mask with ones in the low `bytes` bytes.
fn low_bytes(bytes: u32) -> u128 {
    (1u128 << (bytes * 8)) - 1
}

/// A `v128` with every lane of `lane` set to `bits`.
fn splat_const(lane: Lane, bits: u64) -> I<'static> {
    let mut v: u128 = 0;
    let width = lane.bytes() * 8;
    for i in 0..128 / width {
        v |= (bits as u128 & ((1u128 << width) - 1)) << (i * width);
    }
    I::V128Const(v as i128)
}

impl Emitter<'_> {
    pub(crate) fn lower_simd(&mut self, inst: Inst) -> Result<()> {
        let func = self.func;
        let dfg = &func.dfg;
        let data = dfg.insts[inst];
        let op = data.opcode();
        let args = dfg.inst_args(inst);
        // Dynamic vectors are treated as their fixed-size types.
        let ty = dfg.inst_results(inst).first().map_or(types::INVALID, |&r| {
            wasm_types::fixed_type(dfg.value_type(r))
        });
        let aty = args.first().map_or(types::INVALID, |&a| {
            wasm_types::fixed_type(dfg.value_type(a))
        });
        // The vector type the operation works on.
        let vty = if ty.is_vector() { ty } else { aty };
        let lane = Lane::of(vty)?;
        // Lanes in use, and lanes in a `v128`.
        let count = vty.lane_count();
        let full = 16 / lane.bytes();

        match (op, data) {
            (Opcode::Vconst, _) => self.constant(inst)?,

            // Lanes.
            (Opcode::Splat, _) => {
                self.operand()?;
                self.ins(lane.splat());
            }
            (Opcode::ScalarToVector, _) => {
                self.ins(I::V128Const(0));
                self.operand()?;
                self.ins(lane.replace_lane(0));
            }
            (Opcode::Extractlane, D::BinaryImm8 { imm, .. }) => {
                self.operand()?;
                self.ins(lane.extract_lane(imm));
            }
            (Opcode::Insertlane, D::TernaryImm8 { imm, .. }) => {
                self.operand()?;
                self.operand()?;
                self.ins(lane.replace_lane(imm));
            }
            (Opcode::Shuffle, D::Shuffle { imm, .. }) => {
                self.operand()?;
                self.operand()?;
                let bytes: [u8; 16] = dfg.immediates[imm]
                    .as_slice()
                    .try_into()
                    .expect("16-byte shuffle mask");
                self.ins(I::I8x16Shuffle(bytes));
            }
            // With a dynamic vector the size of its fixed type, only the
            // first fixed-size piece exists.
            (Opcode::ExtractVector, D::BinaryImm8 { imm: 0, .. }) => {
                self.operand()?;
            }
            (Opcode::Swizzle, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(I::I8x16Swizzle);
            }

            // Integer arithmetic.
            (Opcode::Iadd | Opcode::Isub, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(if op == Opcode::Iadd {
                    lanes!(lane, op; I8 => I::I8x16Add, I16 => I::I16x8Add,
                        I32 => I::I32x4Add, I64 => I::I64x2Add)
                } else {
                    lanes!(lane, op; I8 => I::I8x16Sub, I16 => I::I16x8Sub,
                        I32 => I::I32x4Sub, I64 => I::I64x2Sub)
                });
            }
            (Opcode::Imul, _) if lane == Lane::I8 => {
                // The low byte of each 16-bit product.
                self.widening_mul(Lane::I8, /* signed= */ false)?;
                self.ins(I::I8x16Shuffle(shuffle_bytes(1, |i| 2 * i)));
            }
            (Opcode::Imul, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(lanes!(lane, op;
                    I16 => I::I16x8Mul, I32 => I::I32x4Mul, I64 => I::I64x2Mul));
            }
            (Opcode::Umulhi | Opcode::Smulhi, _) if lane == Lane::I64 => {
                self.mulhi_i64x2(/* signed= */ op == Opcode::Smulhi)?;
            }
            (Opcode::Umulhi | Opcode::Smulhi, _) => {
                // The high half of each double-width product.
                self.widening_mul(lane, op == Opcode::Smulhi)?;
                self.ins(I::I8x16Shuffle(shuffle_bytes(lane.bytes(), |i| 2 * i + 1)));
            }
            (Opcode::Ineg | Opcode::Iabs, _) => {
                self.operand()?;
                self.ins(if op == Opcode::Ineg {
                    lanes!(lane, op; I8 => I::I8x16Neg, I16 => I::I16x8Neg,
                        I32 => I::I32x4Neg, I64 => I::I64x2Neg)
                } else {
                    lanes!(lane, op; I8 => I::I8x16Abs, I16 => I::I16x8Abs,
                        I32 => I::I32x4Abs, I64 => I::I64x2Abs)
                });
            }
            (Opcode::Smin | Opcode::Smax | Opcode::Umin | Opcode::Umax, _) if lane == Lane::I64 => {
                self.i64x2_min_max(op)?;
            }
            (Opcode::Smin | Opcode::Smax | Opcode::Umin | Opcode::Umax, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::Smin => lanes!(lane, op; I8 => I::I8x16MinS, I16 => I::I16x8MinS,
                        I32 => I::I32x4MinS),
                    Opcode::Smax => lanes!(lane, op; I8 => I::I8x16MaxS, I16 => I::I16x8MaxS,
                        I32 => I::I32x4MaxS),
                    Opcode::Umin => lanes!(lane, op; I8 => I::I8x16MinU, I16 => I::I16x8MinU,
                        I32 => I::I32x4MinU),
                    _ => lanes!(lane, op; I8 => I::I8x16MaxU, I16 => I::I16x8MaxU,
                        I32 => I::I32x4MaxU),
                });
            }
            (Opcode::UaddSat | Opcode::SaddSat | Opcode::UsubSat | Opcode::SsubSat, _)
                if matches!(lane, Lane::I32 | Lane::I64) =>
            {
                self.wide_sat(op, lane)?;
            }
            (Opcode::UaddSat | Opcode::SaddSat | Opcode::UsubSat | Opcode::SsubSat, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::UaddSat => lanes!(lane, op; I8 => I::I8x16AddSatU,
                        I16 => I::I16x8AddSatU),
                    Opcode::SaddSat => lanes!(lane, op; I8 => I::I8x16AddSatS,
                        I16 => I::I16x8AddSatS),
                    Opcode::UsubSat => lanes!(lane, op; I8 => I::I8x16SubSatU,
                        I16 => I::I16x8SubSatU),
                    _ => lanes!(lane, op; I8 => I::I8x16SubSatS, I16 => I::I16x8SubSatS),
                });
            }
            (Opcode::AvgRound, _) if matches!(lane, Lane::I32 | Lane::I64) => {
                // (a + b + 1) >> 1 without overflow: (a | b) - ((a ^ b) >> 1).
                let (shr, sub) = if lane == Lane::I32 {
                    (I::I32x4ShrU, I::I32x4Sub)
                } else {
                    (I::I64x2ShrU, I::I64x2Sub)
                };
                let [a, b] = self.scratches(ValType::V128);
                self.operand()?;
                self.local_tee(a);
                self.operand()?;
                self.local_tee(b);
                self.ins(I::V128Or);
                self.local_get(a);
                self.local_get(b);
                self.ins(I::V128Xor);
                self.ins(I::I32Const(1));
                self.ins(shr);
                self.ins(sub);
            }
            (Opcode::AvgRound, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(lanes!(lane, op; I8 => I::I8x16AvgrU, I16 => I::I16x8AvgrU));
            }
            (Opcode::SqmulRoundSat, _) if lane == Lane::I32 => {
                // (a * b + 2^30) >> 31 on 64-bit products, then the low
                // halves. Only MIN * MIN overflows; it wraps to MIN, which
                // no other inputs produce, so turn MIN into MAX.
                let [a, b] = self.scratches(ValType::V128);
                self.operand()?;
                self.local_set(a);
                self.operand()?;
                self.local_set(b);
                for mul in [I::I64x2ExtMulLowI32x4S, I::I64x2ExtMulHighI32x4S] {
                    self.local_get(a);
                    self.local_get(b);
                    self.ins(mul);
                    self.ins(splat_const(Lane::I64, 1 << 30));
                    self.ins(I::I64x2Add);
                    self.ins(I::I32Const(31));
                    self.ins(I::I64x2ShrS);
                }
                self.ins(I::I8x16Shuffle(shuffle_bytes(4, |i| 2 * i)));
                self.local_tee(a);
                self.local_get(a);
                self.ins(splat_const(Lane::I32, 0x8000_0000));
                self.ins(I::I32x4Eq);
                self.ins(I::V128Xor);
            }
            (Opcode::SqmulRoundSat, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(lanes!(lane, op; I16 => I::I16x8Q15MulrSatS));
            }
            (Opcode::IaddPairwise, _) => {
                // [x0+x1, x2+x3, ..., y0+y1, ...]: add the even lanes of
                // x then y to the odd ones. Lane `full + k` of the shuffle's
                // input is lane `k` of y.
                let [x, y] = self.scratches(ValType::V128);
                let b = lane.bytes();
                let half = count / 2;
                let pick = move |i: u32, odd: u32| match i {
                    i if i < half => 2 * i + odd,
                    i if i < count => full + 2 * (i - half) + odd,
                    _ => 0,
                };
                self.operand()?;
                self.local_tee(x);
                self.operand()?;
                self.local_tee(y);
                self.ins(I::I8x16Shuffle(shuffle_bytes(b, |i| pick(i, 0))));
                self.local_get(x);
                self.local_get(y);
                self.ins(I::I8x16Shuffle(shuffle_bytes(b, |i| pick(i, 1))));
                self.ins(lanes!(lane, op; I8 => I::I8x16Add, I16 => I::I16x8Add,
                    I32 => I::I32x4Add, I64 => I::I64x2Add));
            }
            (Opcode::Popcnt, _) => {
                self.operand()?;
                self.ins(I::I8x16Popcnt);
                match lane {
                    Lane::I8 => {}
                    Lane::I16 => self.ins(I::I16x8ExtAddPairwiseI8x16U),
                    Lane::I32 => {
                        self.ins(I::I16x8ExtAddPairwiseI8x16U);
                        self.ins(I::I32x4ExtAddPairwiseI16x8U);
                    }
                    Lane::I64 => {
                        // Per 32-bit half, then add the halves of each lane.
                        self.ins(I::I16x8ExtAddPairwiseI8x16U);
                        self.ins(I::I32x4ExtAddPairwiseI16x8U);
                        let s = self.scratch(ValType::V128);
                        self.local_tee(s);
                        self.local_get(s);
                        self.ins(I::I32Const(32));
                        self.ins(I::I64x2ShrU);
                        self.ins(I::I64x2Add);
                        self.ins(splat_const(Lane::I64, 0xffff_ffff));
                        self.ins(I::V128And);
                    }
                    _ => return error::unsupported(format!("{op} on {vty}")),
                }
            }

            // Shifts. The amount is a scalar, taken modulo the lane width by
            // both Cranelift and wasm.
            (Opcode::Ishl | Opcode::Ushr | Opcode::Sshr, _) => {
                self.operand()?;
                let amt = self.operand()?;
                if dfg.value_type(amt) == types::I64 {
                    self.ins(I::I32WrapI64);
                }
                self.ins(match op {
                    Opcode::Ishl => lanes!(lane, op; I8 => I::I8x16Shl, I16 => I::I16x8Shl,
                        I32 => I::I32x4Shl, I64 => I::I64x2Shl),
                    Opcode::Ushr => lanes!(lane, op; I8 => I::I8x16ShrU, I16 => I::I16x8ShrU,
                        I32 => I::I32x4ShrU, I64 => I::I64x2ShrU),
                    _ => lanes!(lane, op; I8 => I::I8x16ShrS, I16 => I::I16x8ShrS,
                        I32 => I::I32x4ShrS, I64 => I::I64x2ShrS),
                });
            }

            (Opcode::Rotl | Opcode::Rotr, _) => {
                // (x << n) | (x >> (width - n)). Wasm takes shift amounts
                // modulo the lane width, so n = 0 gives x | x.
                let (shl, shr) = match lane {
                    Lane::I8 => (I::I8x16Shl, I::I8x16ShrU),
                    Lane::I16 => (I::I16x8Shl, I::I16x8ShrU),
                    Lane::I32 => (I::I32x4Shl, I::I32x4ShrU),
                    Lane::I64 => (I::I64x2Shl, I::I64x2ShrU),
                    _ => return error::unsupported(format!("{op} on {vty}")),
                };
                let (first, second) = if op == Opcode::Rotl {
                    (shl, shr)
                } else {
                    (shr, shl)
                };
                let x = self.scratch(ValType::V128);
                let n = self.scratch(ValType::I32);
                self.operand()?;
                self.local_set(x);
                let amt = self.operand()?;
                if dfg.value_type(amt) == types::I64 {
                    self.ins(I::I32WrapI64);
                }
                self.local_set(n);
                self.local_get(x);
                self.local_get(n);
                self.ins(first);
                self.local_get(x);
                self.ins(I::I32Const((lane.bytes() * 8) as i32));
                self.local_get(n);
                self.ins(I::I32Sub);
                self.ins(second);
                self.ins(I::V128Or);
            }

            // Bitwise.
            (Opcode::Band | Opcode::Bor | Opcode::Bxor, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::Band => I::V128And,
                    Opcode::Bor => I::V128Or,
                    _ => I::V128Xor,
                });
            }
            (Opcode::Bnot, _) => {
                self.operand()?;
                self.ins(I::V128Not);
            }
            (Opcode::Bitselect, _) => {
                // Operands are (x, y, mask); see `expr_trees::operands`.
                self.operand()?;
                self.operand()?;
                self.operand()?;
                self.ins(I::V128Bitselect);
            }
            (Opcode::Select | Opcode::SelectSpectreGuard, _) => {
                self.operand()?;
                self.operand()?;
                let c = self.operand()?;
                self.int_to_cond(dfg.value_type(c));
                self.ins(I::Select);
            }

            // Comparisons and reductions.
            (Opcode::Icmp, D::IntCompare { cond, .. }) => self.vector_icmp(lane, cond)?,
            (Opcode::Fcmp, D::FloatCompare { cond, .. }) => {
                let ops = lanes!(Lane::of(aty)?, op;
                    F32 => FloatCmpOps::F32X4, F64 => FloatCmpOps::F64X2);
                self.emit_fcmp(cond, ops)?;
            }
            (Opcode::VanyTrue, _) => {
                self.operand()?;
                if count < full {
                    // Clear the unused bytes.
                    self.ins(I::V128Const(low_bytes(vty.bytes()) as i128));
                    self.ins(I::V128And);
                }
                self.ins(I::V128AnyTrue);
            }
            (Opcode::VallTrue, _) => {
                self.operand()?;
                if count < full {
                    // Set the unused bytes, so their lanes count as true.
                    self.ins(I::V128Const(!low_bytes(vty.bytes()) as i128));
                    self.ins(I::V128Or);
                }
                self.ins(
                    lanes!(lane.int(), op; I8 => I::I8x16AllTrue, I16 => I::I16x8AllTrue,
                    I32 => I::I32x4AllTrue, I64 => I::I64x2AllTrue),
                );
            }
            (Opcode::VhighBits, _) => {
                self.operand()?;
                self.ins(
                    lanes!(lane.int(), op; I8 => I::I8x16Bitmask, I16 => I::I16x8Bitmask,
                    I32 => I::I32x4Bitmask, I64 => I::I64x2Bitmask),
                );
                if count < full {
                    self.ins(I::I32Const((1 << count) - 1));
                    self.ins(I::I32And);
                }
                match ty {
                    types::I64 => self.ins(I::I64ExtendI32U),
                    _ => self.mask(ty),
                }
            }

            // Narrowing and widening.
            (Opcode::Snarrow | Opcode::Unarrow | Opcode::Uunarrow, _)
                if Lane::of(aty)? == Lane::I64 =>
            {
                // Clamp each 64-bit lane to the 32-bit range, then keep the
                // low halves: [x0, x1, y0, y1].
                for _ in 0..2 {
                    self.operand()?;
                    self.clamp_i64x2(op)?;
                }
                self.ins(I::I8x16Shuffle(shuffle_bytes(4, |i| 2 * i)));
            }
            (Opcode::Snarrow | Opcode::Unarrow, _) => {
                self.operand()?;
                self.operand()?;
                let signed = op == Opcode::Snarrow;
                self.ins(match (Lane::of(aty)?, signed) {
                    (Lane::I16, true) => I::I8x16NarrowI16x8S,
                    (Lane::I16, false) => I::I8x16NarrowI16x8U,
                    (Lane::I32, true) => I::I16x8NarrowI32x4S,
                    (Lane::I32, false) => I::I16x8NarrowI32x4U,
                    _ => return error::unsupported(format!("{op} on {vty}")),
                });
                self.join_narrowed(aty)?;
            }
            (Opcode::Uunarrow, _) => {
                // Clamp as unsigned first; the narrowing then sees only
                // non-negative lanes.
                let src = Lane::of(aty)?;
                let (max, min, narrow) = match src {
                    Lane::I16 => (0xff, I::I16x8MinU, I::I8x16NarrowI16x8U),
                    Lane::I32 => (0xffff, I::I32x4MinU, I::I16x8NarrowI32x4U),
                    _ => return error::unsupported(format!("{op} on {vty}")),
                };
                for _ in 0..2 {
                    self.operand()?;
                    self.ins(splat_const(src, max));
                    self.ins(min.clone());
                }
                self.ins(narrow);
                self.join_narrowed(aty)?;
            }
            (
                Opcode::SwidenLow | Opcode::SwidenHigh | Opcode::UwidenLow | Opcode::UwidenHigh,
                _,
            ) => {
                self.operand()?;
                let src = Lane::of(aty)?;
                let n = aty.lane_count();
                let op = match op {
                    // The high half of a narrow vector is not the high half
                    // of the v128: move it down and widen the low half.
                    Opcode::SwidenHigh | Opcode::UwidenHigh if n < 16 / src.bytes() => {
                        let s = self.scratch(ValType::V128);
                        self.local_tee(s);
                        self.local_get(s);
                        self.ins(I::I8x16Shuffle(shuffle_bytes(src.bytes(), |i| i + n / 2)));
                        if op == Opcode::SwidenHigh {
                            Opcode::SwidenLow
                        } else {
                            Opcode::UwidenLow
                        }
                    }
                    op => op,
                };
                self.ins(match (op, src) {
                    (Opcode::SwidenLow, Lane::I8) => I::I16x8ExtendLowI8x16S,
                    (Opcode::SwidenHigh, Lane::I8) => I::I16x8ExtendHighI8x16S,
                    (Opcode::UwidenLow, Lane::I8) => I::I16x8ExtendLowI8x16U,
                    (Opcode::UwidenHigh, Lane::I8) => I::I16x8ExtendHighI8x16U,
                    (Opcode::SwidenLow, Lane::I16) => I::I32x4ExtendLowI16x8S,
                    (Opcode::SwidenHigh, Lane::I16) => I::I32x4ExtendHighI16x8S,
                    (Opcode::UwidenLow, Lane::I16) => I::I32x4ExtendLowI16x8U,
                    (Opcode::UwidenHigh, Lane::I16) => I::I32x4ExtendHighI16x8U,
                    (Opcode::SwidenLow, Lane::I32) => I::I64x2ExtendLowI32x4S,
                    (Opcode::SwidenHigh, Lane::I32) => I::I64x2ExtendHighI32x4S,
                    (Opcode::UwidenLow, Lane::I32) => I::I64x2ExtendLowI32x4U,
                    (Opcode::UwidenHigh, Lane::I32) => I::I64x2ExtendHighI32x4U,
                    _ => return error::unsupported(format!("{op} on {aty}")),
                });
            }

            // Floating point.
            (
                Opcode::Fadd
                | Opcode::Fsub
                | Opcode::Fmul
                | Opcode::Fdiv
                | Opcode::Fmin
                | Opcode::Fmax,
                _,
            ) => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::Fadd => lanes!(lane, op; F32 => I::F32x4Add, F64 => I::F64x2Add),
                    Opcode::Fsub => lanes!(lane, op; F32 => I::F32x4Sub, F64 => I::F64x2Sub),
                    Opcode::Fmul => lanes!(lane, op; F32 => I::F32x4Mul, F64 => I::F64x2Mul),
                    Opcode::Fdiv => lanes!(lane, op; F32 => I::F32x4Div, F64 => I::F64x2Div),
                    Opcode::Fmin => lanes!(lane, op; F32 => I::F32x4Min, F64 => I::F64x2Min),
                    _ => lanes!(lane, op; F32 => I::F32x4Max, F64 => I::F64x2Max),
                });
            }
            (
                Opcode::Fneg
                | Opcode::Fabs
                | Opcode::Sqrt
                | Opcode::Ceil
                | Opcode::Floor
                | Opcode::Trunc
                | Opcode::Nearest,
                _,
            ) => {
                self.operand()?;
                self.ins(match op {
                    Opcode::Fneg => lanes!(lane, op; F32 => I::F32x4Neg, F64 => I::F64x2Neg),
                    Opcode::Fabs => lanes!(lane, op; F32 => I::F32x4Abs, F64 => I::F64x2Abs),
                    Opcode::Sqrt => lanes!(lane, op; F32 => I::F32x4Sqrt, F64 => I::F64x2Sqrt),
                    Opcode::Ceil => lanes!(lane, op; F32 => I::F32x4Ceil, F64 => I::F64x2Ceil),
                    Opcode::Floor => lanes!(lane, op; F32 => I::F32x4Floor, F64 => I::F64x2Floor),
                    Opcode::Trunc => lanes!(lane, op; F32 => I::F32x4Trunc, F64 => I::F64x2Trunc),
                    _ => lanes!(lane, op; F32 => I::F32x4Nearest, F64 => I::F64x2Nearest),
                });
            }
            (Opcode::Fma, _) if lane == Lane::F32 => self.fma_f32_lanes(vty)?,
            (Opcode::Fma, _) if lane == Lane::F64 => self.fma_f64_lanes()?,
            (Opcode::Fcopysign, _) => {
                // (x & ~sign) | (y & sign) == bitselect(x, y, ~sign).
                self.operand()?;
                self.operand()?;
                let magnitude = lanes!(lane, op;
                    F32 => splat_const(Lane::I32, 0x7fff_ffff),
                    F64 => splat_const(Lane::I64, 0x7fff_ffff_ffff_ffff),
                );
                self.ins(magnitude);
                self.ins(I::V128Bitselect);
            }
            (Opcode::FcvtFromSint | Opcode::FcvtFromUint, _) => {
                self.operand()?;
                match (op, Lane::of(aty)?, lane) {
                    (Opcode::FcvtFromSint, Lane::I32, Lane::F32) => self.ins(I::F32x4ConvertI32x4S),
                    (Opcode::FcvtFromUint, Lane::I32, Lane::F32) => self.ins(I::F32x4ConvertI32x4U),
                    // No 64-bit lane conversions in wasm: go through scalars.
                    (Opcode::FcvtFromSint, Lane::I64, Lane::F64) => {
                        self.per_lane_64(Lane::I64, Lane::F64, I::F64ConvertI64S)
                    }
                    (Opcode::FcvtFromUint, Lane::I64, Lane::F64) => {
                        self.per_lane_64(Lane::I64, Lane::F64, I::F64ConvertI64U)
                    }
                    _ => return error::unsupported(format!("{op} {aty} -> {ty}")),
                }
            }
            (Opcode::FcvtToSintSat | Opcode::FcvtToUintSat, _) => {
                self.operand()?;
                match (op, Lane::of(aty)?, lane) {
                    (Opcode::FcvtToSintSat, Lane::F32, Lane::I32) => {
                        self.ins(I::I32x4TruncSatF32x4S)
                    }
                    (Opcode::FcvtToUintSat, Lane::F32, Lane::I32) => {
                        self.ins(I::I32x4TruncSatF32x4U)
                    }
                    (Opcode::FcvtToSintSat, Lane::F64, Lane::I64) => {
                        self.per_lane_64(Lane::F64, Lane::I64, I::I64TruncSatF64S)
                    }
                    (Opcode::FcvtToUintSat, Lane::F64, Lane::I64) => {
                        self.per_lane_64(Lane::F64, Lane::I64, I::I64TruncSatF64U)
                    }
                    _ => return error::unsupported(format!("{op} {aty} -> {ty}")),
                }
            }
            (Opcode::Fvdemote, _) => {
                self.operand()?;
                self.ins(I::F32x4DemoteF64x2Zero);
            }
            (Opcode::FvpromoteLow, _) => {
                self.operand()?;
                self.ins(I::F64x2PromoteLowF32x4);
            }

            (Opcode::Bitcast, D::LoadNoOffset { flags, .. }) => {
                // Vectors of the same size share a representation, with
                // lanes in little-endian order.
                if self.big_endian(flags) && aty.lane_bits() != ty.lane_bits() {
                    return error::unsupported(format!("big-endian bitcast {aty} -> {ty}"));
                }
                self.operand()?;
                match (aty.is_vector(), ty.is_vector()) {
                    (true, true) => {
                        Lane::of(aty)?;
                    }
                    // A scalar is lane 0 of a vector of its width.
                    (true, false) => match Lane::from_scalar(ty) {
                        Some(l) if l != Lane::I8 => self.ins(l.extract_lane(0)),
                        _ => return error::unsupported(format!("bitcast {aty} -> {ty}")),
                    },
                    (false, _) => match Lane::from_scalar(aty) {
                        Some(l) if l != Lane::I8 => self.ins(l.splat()),
                        _ => return error::unsupported(format!("bitcast {aty} -> {ty}")),
                    },
                }
            }

            // Memory.
            (
                Opcode::Load
                | Opcode::Uload8x8
                | Opcode::Sload8x8
                | Opcode::Uload16x4
                | Opcode::Sload16x4
                | Opcode::Uload32x2
                | Opcode::Sload32x2,
                D::Load { flags, offset, .. },
            ) => {
                if self.big_endian(flags) {
                    return error::unsupported("big-endian vector load");
                }
                let off = self.mem_address(offset.into())?;
                let (size, i): (u32, fn(MemArg) -> I<'static>) = match op {
                    Opcode::Load => match ty.bytes() {
                        16 => (16, I::V128Load),
                        8 => (8, I::V128Load64Zero),
                        4 => (4, I::V128Load32Zero),
                        _ => (2, I::V128Load16Splat),
                    },
                    Opcode::Uload8x8 => (8, I::V128Load8x8U),
                    Opcode::Sload8x8 => (8, I::V128Load8x8S),
                    Opcode::Uload16x4 => (8, I::V128Load16x4U),
                    Opcode::Sload16x4 => (8, I::V128Load16x4S),
                    Opcode::Uload32x2 => (8, I::V128Load32x2U),
                    _ => (8, I::V128Load32x2S),
                };
                let m = self.memarg(off, flags, size);
                self.ins(i(m));
            }
            (Opcode::Store, D::Store { flags, offset, .. }) => {
                if self.big_endian(flags) {
                    return error::unsupported("big-endian vector store");
                }
                let off = self.mem_address(offset.into())?;
                self.operand()?;
                let size = aty.bytes();
                let memarg = self.memarg(off, flags, size);
                self.ins(match size {
                    16 => I::V128Store(memarg),
                    8 => I::V128Store64Lane { memarg, lane: 0 },
                    4 => I::V128Store32Lane { memarg, lane: 0 },
                    _ => I::V128Store16Lane { memarg, lane: 0 },
                });
            }

            _ => return error::unsupported(format!("instruction `{}`", dfg.display_inst(inst))),
        }
        Ok(())
    }

    /// After a wasm narrowing of two vectors of `src` type, which places
    /// the lanes of each input at the start of a v128 half, gather the used
    /// lanes when `src` is narrower than 128 bits: `[x.., y..]`.
    fn join_narrowed(&mut self, src: Type) -> Result<()> {
        let n = src.lane_count();
        let full = 16 / Lane::of(src)?.bytes();
        if n < full {
            let s = self.scratch(ValType::V128);
            // The result's lanes are half the size: `full` of them per
            // input.
            let bytes = Lane::of(src)?.bytes() / 2;
            self.local_tee(s);
            self.local_get(s);
            self.ins(I::I8x16Shuffle(shuffle_bytes(bytes, |i| {
                if i < n { i } else { full + i - n }
            })));
        }
        Ok(())
    }

    /// Apply a scalar conversion to both lanes of the 64-bit-lane vector on
    /// the stack.
    fn per_lane_64(&mut self, from: Lane, to: Lane, convert: I<'static>) {
        let s = self.scratch(ValType::V128);
        self.local_tee(s);
        self.ins(from.extract_lane(0));
        self.ins(convert.clone());
        self.ins(to.splat());
        self.local_get(s);
        self.ins(from.extract_lane(1));
        self.ins(convert);
        self.ins(to.replace_lane(1));
    }

    /// Push both operands and leave the double-width products of their low
    /// and high halves: `[lo, hi]`.
    fn widening_mul(&mut self, lane: Lane, signed: bool) -> Result<()> {
        let (lo, hi) = match (lane, signed) {
            (Lane::I8, false) => (I::I16x8ExtMulLowI8x16U, I::I16x8ExtMulHighI8x16U),
            (Lane::I8, true) => (I::I16x8ExtMulLowI8x16S, I::I16x8ExtMulHighI8x16S),
            (Lane::I16, false) => (I::I32x4ExtMulLowI16x8U, I::I32x4ExtMulHighI16x8U),
            (Lane::I16, true) => (I::I32x4ExtMulLowI16x8S, I::I32x4ExtMulHighI16x8S),
            (Lane::I32, false) => (I::I64x2ExtMulLowI32x4U, I::I64x2ExtMulHighI32x4U),
            (Lane::I32, true) => (I::I64x2ExtMulLowI32x4S, I::I64x2ExtMulHighI32x4S),
            _ => return error::unsupported(format!("widening multiply of {lane:?} lanes")),
        };
        let [x, y] = self.scratches(ValType::V128);
        self.operand()?;
        self.local_tee(x);
        self.operand()?;
        self.local_tee(y);
        self.ins(lo);
        self.local_get(x);
        self.local_get(y);
        self.ins(hi);
        Ok(())
    }

    /// Saturating add or subtract on 32- or 64-bit lanes, which wasm only
    /// has for 8- and 16-bit lanes: compute the wrapping result, then
    /// replace the lanes that overflowed.
    fn wide_sat(&mut self, op: Opcode, lane: Lane) -> Result<()> {
        let (add, sub, shr_s, max): (I<'static>, I<'static>, I<'static>, u64) = match lane {
            Lane::I32 => (I::I32x4Add, I::I32x4Sub, I::I32x4ShrS, i32::MAX as u64),
            _ => (I::I64x2Add, I::I64x2Sub, I::I64x2ShrS, i64::MAX as u64),
        };
        let sign_shift = (lane.bytes() * 8 - 1) as i32;
        let [a, b, r] = self.scratches(ValType::V128);
        self.operand()?;
        self.local_set(a);
        self.operand()?;
        self.local_set(b);
        let adding = matches!(op, Opcode::UaddSat | Opcode::SaddSat);
        self.local_get(a);
        self.local_get(b);
        self.ins(if adding { add } else { sub });
        self.local_set(r);
        match op {
            // A carry out shows as r < a; saturate to all ones.
            Opcode::UaddSat => {
                self.local_get(r);
                self.lt_u(lane, r, a);
                self.ins(I::V128Or);
            }
            // A borrow happens when a < b; saturate to zero.
            Opcode::UsubSat => {
                self.local_get(r);
                self.lt_u(lane, a, b);
                self.ins(I::V128AndNot);
            }
            _ => {
                // The saturated value: MIN where a < 0, MAX elsewhere.
                self.local_get(a);
                self.ins(I::I32Const(sign_shift));
                self.ins(shr_s.clone());
                self.ins(splat_const(lane, max));
                self.ins(I::V128Xor);
                self.local_get(r);
                // Overflow, as an all-ones lane mask:
                //   add: (a ^ r) & (b ^ r) < 0
                //   sub: (a ^ b) & (a ^ r) < 0
                let pairs = if adding {
                    [(a, r), (b, r)]
                } else {
                    [(a, b), (a, r)]
                };
                for (x, y) in pairs {
                    self.local_get(x);
                    self.local_get(y);
                    self.ins(I::V128Xor);
                }
                self.ins(I::V128And);
                self.ins(I::I32Const(sign_shift));
                self.ins(shr_s);
                self.ins(I::V128Bitselect);
            }
        }
        Ok(())
    }

    /// Push the lane mask of `x < y`, unsigned, for the locals `x` and `y`.
    fn lt_u(&mut self, lane: Lane, x: Local, y: Local) {
        for l in [x, y] {
            self.local_get(l);
            if lane == Lane::I64 {
                // No unsigned 64-bit lane comparison: flip the sign bits.
                self.ins(splat_const(Lane::I64, 1 << 63));
                self.ins(I::V128Xor);
            }
        }
        self.ins(if lane == Lane::I64 {
            I::I64x2LtS
        } else {
            I::I32x4LtU
        });
    }

    /// Clamp the `i64x2` on the stack to the result range of a 64-to-32-bit
    /// narrowing `op`.
    fn clamp_i64x2(&mut self, op: Opcode) -> Result<()> {
        const SIGN: u64 = 1 << 63;
        let mark = self.scratch_mark();
        let s = self.scratch(ValType::V128);
        self.local_set(s);
        // (bound, replace the lane when `lane cmp bound`, compare unsigned)
        let bounds: &[(u64, I<'static>, bool)] = match op {
            Opcode::Snarrow => &[
                (i32::MAX as u64, I::I64x2GtS, false),
                (i32::MIN as i64 as u64, I::I64x2LtS, false),
            ],
            Opcode::Unarrow => &[
                (u32::MAX as u64, I::I64x2GtS, false),
                (0, I::I64x2LtS, false),
            ],
            _ => &[(u32::MAX as u64, I::I64x2GtS, true)],
        };
        for (bound, cmp, unsigned) in bounds.iter().cloned() {
            // s = bitselect(bound, s, s cmp bound); an unsigned comparison
            // flips both sides' sign bits and compares signed.
            self.ins(splat_const(Lane::I64, bound));
            self.local_get(s);
            self.local_get(s);
            if unsigned {
                self.ins(splat_const(Lane::I64, SIGN));
                self.ins(I::V128Xor);
            }
            self.ins(splat_const(
                Lane::I64,
                if unsigned { bound ^ SIGN } else { bound },
            ));
            self.ins(cmp);
            self.ins(I::V128Bitselect);
            self.local_set(s);
        }
        self.local_get(s);
        self.free_scratches(mark);
        Ok(())
    }

    /// `umulhi`/`smulhi` on `i64x2`, one lane at a time with the scalar
    /// algorithm.
    fn mulhi_i64x2(&mut self, signed: bool) -> Result<()> {
        let [a, b] = self.scratches(ValType::V128);
        let [sa, sb] = self.scratches(ValType::I64);
        self.operand()?;
        self.local_set(a);
        self.operand()?;
        self.local_set(b);
        for i in 0..2 {
            self.local_get(a);
            self.ins(I::I64x2ExtractLane(i));
            self.local_set(sa);
            self.local_get(b);
            self.ins(I::I64x2ExtractLane(i));
            self.local_set(sb);
            self.mulhi64_of(sa, sb, signed);
            self.ins(if i == 0 {
                I::I64x2Splat
            } else {
                I::I64x2ReplaceLane(1)
            });
        }
        Ok(())
    }

    fn i64x2_min_max(&mut self, op: Opcode) -> Result<()> {
        let unsigned = matches!(op, Opcode::Umin | Opcode::Umax);
        let cmp = if matches!(op, Opcode::Smin | Opcode::Umin) {
            I::I64x2LtS
        } else {
            I::I64x2GtS
        };
        let [a, b] = self.scratches(ValType::V128);
        self.operand()?;
        self.local_tee(a);
        self.operand()?;
        self.local_tee(b);
        // Stack: a, b. Select a where cmp(a, b), else b.
        self.local_get(a);
        if unsigned {
            self.ins(splat_const(Lane::I64, 1 << 63));
            self.ins(I::V128Xor);
        }
        self.local_get(b);
        if unsigned {
            self.ins(splat_const(Lane::I64, 1 << 63));
            self.ins(I::V128Xor);
        }
        self.ins(cmp);
        self.ins(I::V128Bitselect);
        Ok(())
    }

    fn vector_icmp(&mut self, lane: Lane, cond: IntCC) -> Result<()> {
        use IntCC::*;
        // wasm has no unsigned 64-bit lane comparisons: flip the sign bits
        // and compare signed.
        let flip = lane == Lane::I64
            && matches!(
                cond,
                UnsignedLessThan
                    | UnsignedLessThanOrEqual
                    | UnsignedGreaterThan
                    | UnsignedGreaterThanOrEqual
            );
        for _ in 0..2 {
            self.operand()?;
            if flip {
                self.ins(splat_const(Lane::I64, 1 << 63));
                self.ins(I::V128Xor);
            }
        }
        let cond = match (flip, cond) {
            (true, UnsignedLessThan) => SignedLessThan,
            (true, UnsignedLessThanOrEqual) => SignedLessThanOrEqual,
            (true, UnsignedGreaterThan) => SignedGreaterThan,
            (true, UnsignedGreaterThanOrEqual) => SignedGreaterThanOrEqual,
            _ => cond,
        };
        let i = match lane {
            Lane::I8 => match cond {
                Equal => I::I8x16Eq,
                NotEqual => I::I8x16Ne,
                SignedLessThan => I::I8x16LtS,
                SignedLessThanOrEqual => I::I8x16LeS,
                SignedGreaterThan => I::I8x16GtS,
                SignedGreaterThanOrEqual => I::I8x16GeS,
                UnsignedLessThan => I::I8x16LtU,
                UnsignedLessThanOrEqual => I::I8x16LeU,
                UnsignedGreaterThan => I::I8x16GtU,
                UnsignedGreaterThanOrEqual => I::I8x16GeU,
            },
            Lane::I16 => match cond {
                Equal => I::I16x8Eq,
                NotEqual => I::I16x8Ne,
                SignedLessThan => I::I16x8LtS,
                SignedLessThanOrEqual => I::I16x8LeS,
                SignedGreaterThan => I::I16x8GtS,
                SignedGreaterThanOrEqual => I::I16x8GeS,
                UnsignedLessThan => I::I16x8LtU,
                UnsignedLessThanOrEqual => I::I16x8LeU,
                UnsignedGreaterThan => I::I16x8GtU,
                UnsignedGreaterThanOrEqual => I::I16x8GeU,
            },
            Lane::I32 => match cond {
                Equal => I::I32x4Eq,
                NotEqual => I::I32x4Ne,
                SignedLessThan => I::I32x4LtS,
                SignedLessThanOrEqual => I::I32x4LeS,
                SignedGreaterThan => I::I32x4GtS,
                SignedGreaterThanOrEqual => I::I32x4GeS,
                UnsignedLessThan => I::I32x4LtU,
                UnsignedLessThanOrEqual => I::I32x4LeU,
                UnsignedGreaterThan => I::I32x4GtU,
                UnsignedGreaterThanOrEqual => I::I32x4GeU,
            },
            Lane::I64 => match cond {
                Equal => I::I64x2Eq,
                NotEqual => I::I64x2Ne,
                SignedLessThan => I::I64x2LtS,
                SignedLessThanOrEqual => I::I64x2LeS,
                SignedGreaterThan => I::I64x2GtS,
                SignedGreaterThanOrEqual => I::I64x2GeS,
                _ => unreachable!("flipped to signed above"),
            },
            _ => return error::unsupported(format!("icmp on {lane:?} lanes")),
        };
        self.ins(i);
        Ok(())
    }
}
