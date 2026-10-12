//! A seeded generator of random, valid CLIF functions, used by the
//! differential fuzzer and as a benchmark corpus.

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee32, Ieee64};
use cranelift_codegen::ir::{
    AbiParam, ConstantData, Endianness, Function, InstBuilder, MemFlagsData, Signature,
    StackSlotData, StackSlotKind, Type, UserFuncName, Value, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
    pub fn int(&mut self) -> i64 {
        match self.below(4) {
            0 => self.below(8) as i64 - 2,
            1 => self.pick(&[
                i64::MIN,
                i64::MAX,
                0x7f,
                0x80,
                0xff,
                0x7fff,
                0x8000,
                0xffff_ffff,
            ]),
            _ => self.next() as i64,
        }
    }
}

const TYPES: [Type; 6] = [
    types::I8,
    types::I16,
    types::I32,
    types::I64,
    types::F32,
    types::F64,
];
const VTYPES: [Type; 6] = [
    types::I8X16,
    types::I16X8,
    types::I32X4,
    types::I64X2,
    types::F32X4,
    types::F64X2,
];
const SLOT_SIZE: u32 = 64;

fn little() -> MemFlagsData {
    MemFlagsData::new().with_endianness(Endianness::Little)
}

struct Gen<'a, 'b> {
    rng: &'a mut Rng,
    b: FunctionBuilder<'b>,
    vars: Vec<(Variable, Type)>,
    slot: cranelift_codegen::ir::StackSlot,
    budget: u32,
    /// Also generate vector code.
    simd: bool,
    /// Also generate `i128` code.
    wide: bool,
}

/// What kinds of values a generated function may use beyond the basic
/// scalar types.
#[derive(Clone, Copy)]
struct Features {
    simd: bool,
    i128: bool,
}

impl Gen<'_, '_> {
    fn var_of(&mut self, ty: Type) -> Option<Value> {
        let cands: Vec<_> = self
            .vars
            .iter()
            .filter(|(_, t)| *t == ty)
            .map(|(v, _)| *v)
            .collect();
        if cands.is_empty() {
            return None;
        }
        let v = self.rng.pick(&cands);
        Some(self.b.use_var(v))
    }

    fn constant(&mut self, ty: Type) -> Value {
        if ty.is_vector() {
            let bytes: Vec<u8> = (0..16)
                .map(|_| match self.rng.below(4) {
                    0 => 0,
                    1 => 0xff,
                    _ => self.rng.next() as u8,
                })
                .collect();
            let c = self.b.func.dfg.constants.insert(ConstantData::from(bytes));
            return self.b.ins().vconst(ty, c);
        }
        match ty {
            types::F32 => {
                let f = self
                    .rng
                    .pick(&[0.0f32, -0.0, 1.5, -3.25, 1e30, f32::INFINITY]);
                self.b.ins().f32const(Ieee32::with_float(f))
            }
            types::F64 => {
                let f = self
                    .rng
                    .pick(&[0.0f64, -0.0, 2.5, -7.75, 1e300, f64::NEG_INFINITY]);
                self.b.ins().f64const(Ieee64::with_float(f))
            }
            types::I128 => {
                let (lo, hi) = (self.rng.int(), self.rng.int());
                let lo = self.b.ins().iconst(types::I64, lo);
                let hi = self.b.ins().iconst(types::I64, hi);
                self.b.ins().iconcat(lo, hi)
            }
            _ => {
                let i = self.rng.int();
                self.b.ins().iconst(ty, i)
            }
        }
    }

    /// An integer constant of any width (`iconst` cannot make an `i128`).
    fn int_const(&mut self, ty: Type, v: i64) -> Value {
        if ty == types::I128 {
            let lo = self.b.ins().iconst(types::I64, v);
            let hi = self.b.ins().iconst(types::I64, if v < 0 { -1 } else { 0 });
            self.b.ins().iconcat(lo, hi)
        } else {
            self.b.ins().iconst(ty, v)
        }
    }

    fn leaf(&mut self, ty: Type) -> Value {
        if self.rng.chance(75)
            && let Some(v) = self.var_of(ty)
        {
            return v;
        }
        self.constant(ty)
    }

    fn addr(&mut self, size: u32) -> (Value, i32) {
        // Accesses are aligned (they use trusted flags).
        let align = size.clamp(8, 16);
        let slots = SLOT_SIZE / align;
        let off = (self.rng.below(slots as u64) as u32 * align).min(SLOT_SIZE - size) as i32;
        // Split the offset between the address and the immediate.
        let split = if self.rng.chance(50) { off } else { 0 };
        let a = self.b.ins().stack_addr(types::I64, self.slot, split);
        (a, off - split)
    }

    fn expr(&mut self, ty: Type, depth: u32) -> Value {
        if depth == 0 || self.rng.chance(20) {
            return self.leaf(ty);
        }
        let d = depth - 1;
        if ty.is_vector() {
            return self.vexpr(ty, d);
        }
        if self.simd && ty.is_int() && self.rng.chance(12) {
            return self.vector_to_scalar(ty, d);
        }
        if ty.is_float() {
            return match self.rng.below(9) {
                0 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fadd(a, b)
                }
                1 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fmul(a, b)
                }
                2 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fsub(a, b)
                }
                3 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fmin(a, b)
                }
                4 => {
                    let a = self.expr(ty, d);
                    self.b.ins().fneg(a)
                }
                5 => {
                    let src = if self.wide {
                        self.rng
                            .pick(&[types::I8, types::I32, types::I64, types::I128])
                    } else {
                        self.rng.pick(&[types::I8, types::I32, types::I64])
                    };
                    let a = self.expr(src, d);
                    if self.rng.chance(50) {
                        self.b.ins().fcvt_from_sint(ty, a)
                    } else {
                        self.b.ins().fcvt_from_uint(ty, a)
                    }
                }
                6 => {
                    let other = if ty == types::F32 {
                        types::F64
                    } else {
                        types::F32
                    };
                    let a = self.expr(other, d);
                    if ty == types::F32 {
                        self.b.ins().fdemote(ty, a)
                    } else {
                        self.b.ins().fpromote(ty, a)
                    }
                }
                7 => self.load(ty),
                _ => {
                    let c = self.cond(d);
                    let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().select(c, x, y)
                }
            };
        }
        match self.rng.below(24) {
            0..=6 => {
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.iadd(a, b),
                    1 => ins.isub(a, b),
                    2 => ins.imul(a, b),
                    3 => ins.band(a, b),
                    4 => ins.bor(a, b),
                    5 => ins.bxor(a, b),
                    _ => ins.iadd(a, b),
                }
            }
            7 => {
                let a = self.expr(ty, d);
                let amt_ty = if self.wide {
                    self.rng
                        .pick(&[types::I8, types::I32, types::I64, types::I128])
                } else {
                    self.rng.pick(&[types::I8, types::I32, types::I64])
                };
                let b = self.expr(amt_ty, d);
                let ins = self.b.ins();
                match self.rng.below(5) {
                    0 => ins.ishl(a, b),
                    1 => ins.ushr(a, b),
                    2 => ins.sshr(a, b),
                    3 => ins.rotl(a, b),
                    _ => ins.rotr(a, b),
                }
            }
            8 => {
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let ins = self.b.ins();
                match self.rng.below(4) {
                    0 => ins.smin(a, b),
                    1 => ins.smax(a, b),
                    2 => ins.umin(a, b),
                    _ => ins.umax(a, b),
                }
            }
            9 => {
                // Division may trap; both sides must agree.
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let b = if self.rng.chance(80) {
                    let one = self.int_const(ty, 1);
                    self.b.ins().bor(b, one)
                } else {
                    b
                };
                match self.rng.below(4) {
                    0 => self.b.ins().udiv(a, b),
                    1 => self.b.ins().sdiv(a, b),
                    2 => self.b.ins().urem(a, b),
                    _ => {
                        // The interpreter traps on `MIN srem -1`, which
                        // Cranelift defines as 0; keep the divisor positive.
                        let one = self.int_const(ty, 1);
                        let b = self.b.ins().ushr(b, one);
                        let b = self.b.ins().bor(b, one);
                        self.b.ins().srem(a, b)
                    }
                }
            }
            10 if self.wide && ty.bits() >= 16 && self.rng.chance(25) => {
                let a = self.expr(ty, d);
                if self.rng.chance(50) {
                    self.b.ins().bswap(a)
                } else {
                    self.b.ins().bitrev(a)
                }
            }
            10 => {
                let a = self.expr(ty, d);
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.clz(a),
                    1 => ins.ctz(a),
                    2 => ins.popcnt(a),
                    3 => ins.ineg(a),
                    4 => ins.bnot(a),
                    5 => ins.iabs(a),
                    _ => ins.cls(a),
                }
            }
            11 => {
                let c = self.cond(d);
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                self.b.ins().select(c, x, y)
            }
            12 => {
                let c = self.cond(d);
                if ty == types::I8 {
                    c
                } else {
                    self.b.ins().uextend(ty, c)
                }
            }
            13 | 14 => {
                // Conversions from another integer type.
                let src = if self.wide {
                    self.rng
                        .pick(&[types::I8, types::I16, types::I32, types::I64, types::I128])
                } else {
                    self.rng
                        .pick(&[types::I8, types::I16, types::I32, types::I64])
                };
                let a = self.expr(src, d);
                if src == ty {
                    a
                } else if src.bits() > ty.bits() {
                    self.b.ins().ireduce(ty, a)
                } else if self.rng.chance(50) {
                    self.b.ins().uextend(ty, a)
                } else {
                    self.b.ins().sextend(ty, a)
                }
            }
            // The interpreter saturates float-to-i128 conversions instead of
            // trapping, so those are left out.
            15 if ty == types::I128 => self.leaf(ty),
            15 => {
                let f = self.rng.pick(&[types::F32, types::F64]);
                let a = self.expr(f, d);
                if ty == types::I32 || ty == types::I64 || self.rng.chance(50) {
                    if self.rng.chance(50) {
                        self.b.ins().fcvt_to_sint_sat(ty, a)
                    } else {
                        self.b.ins().fcvt_to_uint_sat(ty, a)
                    }
                } else {
                    self.b.ins().fcvt_to_sint(ty, a)
                }
            }
            16 | 17 => self.load(ty),
            18 => {
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                if ty.bits() <= 32 {
                    if self.rng.chance(50) {
                        self.b.ins().umulhi(a, b)
                    } else {
                        self.b.ins().smulhi(a, b)
                    }
                } else {
                    self.b.ins().bitselect(a, b, a)
                }
            }
            19 => {
                // A load whose value is used after a store to the same
                // place: tests effect ordering.
                let v = self.load(ty);
                self.store();
                v
            }
            20 => {
                // Arithmetic with an overflow flag; use the result or the
                // flag. i128 has only the add and subtract forms.
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let n = if ty == types::I128 { 8 } else { 10 };
                let k = self.rng.below(n);
                let c = if (2..4).contains(&k) || (6..8).contains(&k) {
                    self.expr(types::I8, d)
                } else {
                    a
                };
                let ins = self.b.ins();
                let (r, flag) = match k {
                    0 => ins.uadd_overflow(a, b),
                    1 => ins.sadd_overflow(a, b),
                    2 => ins.uadd_overflow_cin(a, b, c),
                    3 => ins.sadd_overflow_cin(a, b, c),
                    4 => ins.usub_overflow(a, b),
                    5 => ins.ssub_overflow(a, b),
                    6 => ins.usub_overflow_bin(a, b, c),
                    7 => ins.ssub_overflow_bin(a, b, c),
                    8 => ins.umul_overflow(a, b),
                    _ => ins.smul_overflow(a, b),
                };
                if self.rng.chance(50) {
                    r
                } else if ty == types::I8 {
                    flag
                } else {
                    self.b.ins().uextend(ty, flag)
                }
            }
            _ => self.leaf(ty),
        }
    }

    /// Replace NaNs with the canonical NaN. Applied wherever float bits are
    /// reinterpreted (stores, bitcasts, shuffles, bitwise ops on floats):
    /// NaN payloads and signs are nondeterministic in wasm, and differ between
    /// the interpreter and real hardware.
    fn canon(&mut self, v: Value) -> Value {
        let ty = self.b.func.dfg.value_type(v);
        if !ty.lane_type().is_float() {
            return v;
        }
        let nan = if ty.lane_type() == types::F32 {
            self.b.ins().f32const(Ieee32::with_bits(0x7fc0_0000))
        } else {
            self.b
                .ins()
                .f64const(Ieee64::with_bits(0x7ff8_0000_0000_0000))
        };
        let is_nan = self.b.ins().fcmp(FloatCC::Unordered, v, v);
        if ty.is_vector() {
            let nan = self.b.ins().splat(ty, nan);
            let mask = self.b.ins().bitcast(ty, little(), is_nan);
            self.b.ins().bitselect(mask, nan, v)
        } else {
            self.b.ins().select(is_nan, nan, v)
        }
    }

    /// A 16-byte-aligned address in the slot, for vector accesses.
    fn vaddr(&mut self) -> (Value, i32) {
        let off = (self.rng.below((SLOT_SIZE / 16) as u64) * 16) as i32;
        let a = self.b.ins().stack_addr(types::I64, self.slot, 0);
        (a, off)
    }

    /// A vector type with the same lane count as `ty` but `bits`-wide lanes.
    fn with_lane_bits(ty: Type, bits: u32, float: bool) -> Type {
        let lane = match (bits, float) {
            (32, true) => types::F32,
            (64, true) => types::F64,
            (8, _) => types::I8,
            (16, _) => types::I16,
            (32, _) => types::I32,
            _ => types::I64,
        };
        lane.by(128 / bits).unwrap_or(ty)
    }

    /// An expression producing a scalar integer from vectors.
    fn vector_to_scalar(&mut self, ty: Type, d: u32) -> Value {
        match self.rng.below(3) {
            0 => {
                // extractlane from a vector with lanes of this type.
                let vty = ty.by(128 / ty.bits()).unwrap();
                let v = self.expr(vty, d);
                let idx = self.rng.below((128 / ty.bits()) as u64) as u8;
                self.b.ins().extractlane(v, idx)
            }
            1 if ty == types::I8 => {
                let vty = self.rng.pick(&VTYPES);
                let v = self.expr(vty, d);
                if vty.is_float() || self.rng.chance(50) {
                    self.b.ins().vany_true(v)
                } else {
                    self.b.ins().vall_true(v)
                }
            }
            _ => {
                let vty = self.rng.pick(&VTYPES);
                let v = self.expr(vty, d);
                let v = self.canon(v);
                self.b.ins().vhigh_bits(ty, v)
            }
        }
    }

    fn vexpr(&mut self, ty: Type, d: u32) -> Value {
        let lane = ty.lane_type();
        let bits = lane.bits();
        let int = lane.is_int();
        match self.rng.below(20) {
            0 => self.leaf(ty),
            1 => {
                let (a, off) = self.vaddr();
                self.b.ins().load(ty, MemFlagsData::trusted(), a, off)
            }
            2 => {
                let x = self.expr(lane, d);
                self.b.ins().splat(ty, x)
            }
            3 => {
                let v = self.expr(ty, d);
                let x = self.expr(lane, d);
                let idx = self.rng.below(ty.lane_count() as u64) as u8;
                self.b.ins().insertlane(v, x, idx)
            }
            4 => {
                // Shuffle or swizzle as bytes.
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                let (x, y) = (self.canon(x), self.canon(y));
                let (x, y) = (
                    self.b.ins().bitcast(types::I8X16, little(), x),
                    self.b.ins().bitcast(types::I8X16, little(), y),
                );
                let r = if self.rng.chance(70) {
                    let mask: Vec<u8> = (0..16).map(|_| self.rng.below(32) as u8).collect();
                    let imm = self.b.func.dfg.immediates.push(ConstantData::from(mask));
                    self.b.ins().shuffle(x, y, imm)
                } else {
                    self.b.ins().swizzle(x, y)
                };
                self.b.ins().bitcast(ty, little(), r)
            }
            5 => {
                // Reinterpret another vector type.
                let other = self.rng.pick(&VTYPES);
                let v = self.expr(other, d);
                let v = self.canon(v);
                if other == ty {
                    v
                } else {
                    self.b.ins().bitcast(ty, little(), v)
                }
            }
            6 => {
                let c = self.expr(ty, d);
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                let (c, x, y) = (self.canon(c), self.canon(x), self.canon(y));
                self.b.ins().bitselect(c, x, y)
            }
            7 => {
                let c = self.cond(1);
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                self.b.ins().select(c, x, y)
            }
            8 => {
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                let (x, y) = (self.canon(x), self.canon(y));
                let ins = self.b.ins();
                match self.rng.below(3) {
                    0 => ins.band(x, y),
                    1 => ins.bor(x, y),
                    _ => ins.bxor(x, y),
                }
            }
            9 => {
                let x = self.expr(ty, d);
                let x = self.canon(x);
                self.b.ins().bnot(x)
            }
            _ if !int => self.vfloat(ty, d),
            10 | 11 => {
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.iadd(x, y),
                    1 => ins.isub(x, y),
                    2 => ins.imul(x, y),
                    3 => ins.smin(x, y),
                    4 => ins.umax(x, y),
                    5 => ins.umin(x, y),
                    _ => ins.smax(x, y),
                }
            }
            12 => {
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                match (self.rng.below(4), bits) {
                    (0, 8 | 16) => self.b.ins().uadd_sat(x, y),
                    (1, 8 | 16) => self.b.ins().ssub_sat(x, y),
                    (2, 8 | 16) => self.b.ins().usub_sat(x, y),
                    (3, 16) => self.b.ins().sqmul_round_sat(x, y),
                    (_, 64) => self.b.ins().imul(x, y),
                    (3, _) => self.b.ins().iadd_pairwise(x, y),
                    (_, _) if self.rng.chance(50) => self.b.ins().umulhi(x, y),
                    _ => self.b.ins().smulhi(x, y),
                }
            }
            13 => {
                let x = self.expr(ty, d);
                let ins = self.b.ins();
                match self.rng.below(3) {
                    0 => ins.ineg(x),
                    1 => ins.iabs(x),
                    _ => ins.popcnt(x),
                }
            }
            14 => {
                let x = self.expr(ty, d);
                let amt_ty = self.rng.pick(&[types::I32, types::I64]);
                let a = self.expr(amt_ty, d);
                let ins = self.b.ins();
                match self.rng.below(3) {
                    0 => ins.ishl(x, a),
                    1 => ins.ushr(x, a),
                    _ => ins.sshr(x, a),
                }
            }
            15 => {
                let cc = self.rng.pick(&[
                    IntCC::Equal,
                    IntCC::NotEqual,
                    IntCC::SignedLessThan,
                    IntCC::SignedGreaterThanOrEqual,
                    IntCC::UnsignedLessThan,
                    IntCC::UnsignedGreaterThan,
                    IntCC::UnsignedLessThanOrEqual,
                ]);
                if bits >= 32 && self.rng.chance(40) {
                    let fty = Self::with_lane_bits(ty, bits, true);
                    let (x, y) = (self.expr(fty, d), self.expr(fty, d));
                    let cc = self.rng.pick(&[
                        FloatCC::Equal,
                        FloatCC::NotEqual,
                        FloatCC::LessThan,
                        FloatCC::GreaterThanOrEqual,
                        FloatCC::Ordered,
                        FloatCC::Unordered,
                        FloatCC::OrderedNotEqual,
                        FloatCC::UnorderedOrEqual,
                        FloatCC::UnorderedOrLessThan,
                    ]);
                    self.b.ins().fcmp(cc, x, y)
                } else {
                    let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().icmp(cc, x, y)
                }
            }
            16 if bits > 8 => {
                // Widen from narrower lanes.
                let src = Self::with_lane_bits(ty, bits / 2, false);
                let x = self.expr(src, d);
                let ins = self.b.ins();
                match self.rng.below(4) {
                    0 => ins.swiden_low(x),
                    1 => ins.swiden_high(x),
                    2 => ins.uwiden_low(x),
                    _ => ins.uwiden_high(x),
                }
            }
            17 if bits <= 16 => {
                // Narrow from wider lanes.
                let src = Self::with_lane_bits(ty, bits * 2, false);
                let (x, y) = (self.expr(src, d), self.expr(src, d));
                let ins = self.b.ins();
                match self.rng.below(3) {
                    0 => ins.snarrow(x, y),
                    1 => ins.unarrow(x, y),
                    _ => ins.uunarrow(x, y),
                }
            }
            18 if bits >= 32 => {
                let fty = Self::with_lane_bits(ty, bits, true);
                let x = self.expr(fty, d);
                if self.rng.chance(50) {
                    self.b.ins().fcvt_to_sint_sat(ty, x)
                } else {
                    self.b.ins().fcvt_to_uint_sat(ty, x)
                }
            }
            _ => self.leaf(ty),
        }
    }

    fn vfloat(&mut self, ty: Type, d: u32) -> Value {
        let bits = ty.lane_bits();
        match self.rng.below(6) {
            0 | 1 => {
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.fadd(x, y),
                    1 => ins.fsub(x, y),
                    2 => ins.fmul(x, y),
                    3 => ins.fdiv(x, y),
                    4 => ins.fmin(x, y),
                    5 => ins.fmax(x, y),
                    _ => {
                        let y = self.canon(y);
                        self.b.ins().fcopysign(x, y)
                    }
                }
            }
            2 => {
                let x = self.expr(ty, d);
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.fneg(x),
                    1 => ins.fabs(x),
                    2 => ins.sqrt(x),
                    3 => ins.ceil(x),
                    4 => ins.floor(x),
                    5 => ins.trunc(x),
                    _ => ins.nearest(x),
                }
            }
            3 => {
                let ity = Self::with_lane_bits(ty, bits, false);
                let x = self.expr(ity, d);
                if self.rng.chance(50) {
                    self.b.ins().fcvt_from_sint(ty, x)
                } else {
                    self.b.ins().fcvt_from_uint(ty, x)
                }
            }
            4 if bits == 32 => {
                let x = self.expr(types::F64X2, d);
                self.b.ins().fvdemote(x)
            }
            4 => {
                let x = self.expr(types::F32X4, d);
                self.b.ins().fvpromote_low(x)
            }
            _ => self.leaf(ty),
        }
    }

    fn load(&mut self, ty: Type) -> Value {
        let (a, off) = self.addr(ty.bytes());
        let ins = self.b.ins();
        if ty.is_int() && ty.bits() < 64 && self.rng.chance(40) {
            let wide = if ty == types::I32 || self.rng.chance(50) {
                types::I64
            } else {
                types::I32
            };
            let w = if ty.bits() == 8 {
                ins.uload8(wide, MemFlagsData::trusted(), a, off)
            } else if ty.bits() == 16 {
                ins.sload16(wide, MemFlagsData::trusted(), a, off)
            } else {
                ins.sload32(MemFlagsData::trusted(), a, off)
            };
            return self.b.ins().ireduce(ty, w);
        }
        ins.load(ty, MemFlagsData::trusted(), a, off)
    }

    fn store(&mut self) {
        if self.simd && self.rng.chance(30) {
            let ty = self.rng.pick(&VTYPES);
            let v = self.expr(ty, 2);
            let v = self.canon(v);
            let (a, off) = self.vaddr();
            self.b.ins().store(MemFlagsData::trusted(), v, a, off);
            return;
        }
        let ty = if self.wide && self.rng.chance(25) {
            types::I128
        } else {
            self.rng.pick(&TYPES)
        };
        let v = self.expr(ty, 2);
        let v = if self.simd { self.canon(v) } else { v };
        let (a, off) = self.addr(ty.bytes());
        if ty == types::I64 && self.rng.chance(30) {
            self.b.ins().istore16(MemFlagsData::trusted(), v, a, off);
        } else {
            self.b.ins().store(MemFlagsData::trusted(), v, a, off);
        }
    }

    fn cond(&mut self, depth: u32) -> Value {
        let ty = if self.wide && self.rng.chance(30) {
            types::I128
        } else {
            self.rng.pick(&TYPES)
        };
        let (a, b) = (self.expr(ty, depth), self.expr(ty, depth));
        if ty.is_float() {
            let cc = self.rng.pick(&[
                FloatCC::Ordered,
                FloatCC::Unordered,
                FloatCC::Equal,
                FloatCC::NotEqual,
                FloatCC::OrderedNotEqual,
                FloatCC::UnorderedOrEqual,
                FloatCC::LessThan,
                FloatCC::LessThanOrEqual,
                FloatCC::GreaterThan,
                FloatCC::GreaterThanOrEqual,
                FloatCC::UnorderedOrLessThan,
                FloatCC::UnorderedOrLessThanOrEqual,
                FloatCC::UnorderedOrGreaterThan,
                FloatCC::UnorderedOrGreaterThanOrEqual,
            ]);
            self.b.ins().fcmp(cc, a, b)
        } else {
            let cc = self.rng.pick(&[
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
            ]);
            self.b.ins().icmp(cc, a, b)
        }
    }

    fn stmts(&mut self, depth: u32) {
        let n = 1 + self.rng.below(4);
        for _ in 0..n {
            if self.budget == 0 {
                return;
            }
            self.budget -= 1;
            self.stmt(depth);
        }
    }

    fn assign(&mut self) {
        let i = self.rng.below(self.vars.len() as u64) as usize;
        let (var, ty) = self.vars[i];
        let v = self.expr(ty, 3);
        self.b.def_var(var, v);
    }

    fn stmt(&mut self, depth: u32) {
        match self.rng.below(if depth == 0 { 3 } else { 7 }) {
            0 | 1 => self.assign(),
            2 => self.store(),
            3 => {
                // if / else
                let c = self.cond(1);
                let (t, e, m) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().brif(c, t, &[], e, &[]);
                self.b.switch_to_block(t);
                self.stmts(depth - 1);
                self.b.ins().jump(m, &[]);
                self.b.switch_to_block(e);
                if self.rng.chance(70) {
                    self.stmts(depth - 1);
                }
                self.b.ins().jump(m, &[]);
                self.b.switch_to_block(m);
            }
            4 => {
                // A counted loop: 0..4 iterations.
                let counter = self.b.declare_var(types::I32);
                let n = self.b.ins().iconst(types::I32, self.rng.below(5) as i64);
                self.b.def_var(counter, n);
                let (h, body, exit) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().jump(h, &[]);
                self.b.switch_to_block(h);
                let c = self.b.use_var(counter);
                self.b.ins().brif(c, body, &[], exit, &[]);
                self.b.switch_to_block(body);
                self.stmts(depth - 1);
                let c = self.b.use_var(counter);
                let c = self.b.ins().iadd_imm_s(c, -1);
                self.b.def_var(counter, c);
                if self.rng.chance(20) {
                    // An early exit from the loop.
                    let k = self.cond(1);
                    let cont = self.b.create_block();
                    self.b.ins().brif(k, exit, &[], cont, &[]);
                    self.b.switch_to_block(cont);
                }
                self.b.ins().jump(h, &[]);
                self.b.switch_to_block(exit);
            }
            5 => {
                // br_table
                let ity = types::I32;
                let idx = self.expr(ity, 1);
                let n = 1 + self.rng.below(4) as usize;
                let merge = self.b.create_block();
                let targets: Vec<_> = (0..n).map(|_| self.b.create_block()).collect();
                let pool = &mut self.b.func.dfg.value_lists;
                let calls: Vec<_> = (0..n)
                    .map(|i| {
                        let t = targets[(self.rng.below(n as u64) as usize).min(i + 1) % n];
                        cranelift_codegen::ir::BlockCall::new(t, [], pool)
                    })
                    .collect();
                let def = cranelift_codegen::ir::BlockCall::new(targets[0], [], pool);
                let jt = self
                    .b
                    .create_jump_table(cranelift_codegen::ir::JumpTableData::new(def, &calls));
                self.b.ins().br_table(idx, jt);
                for t in targets {
                    self.b.switch_to_block(t);
                    self.stmts(depth - 1);
                    self.b.ins().jump(merge, &[]);
                }
                self.b.switch_to_block(merge);
            }
            _ => {
                // A conditional trap that rarely fires.
                let c = self.cond(1);
                let k = self.b.ins().iconst(types::I8, 1);
                let both = self.b.ins().band(c, k);
                let x = self.b.ins().icmp_imm_s(IntCC::Equal, both, 7);
                self.b
                    .ins()
                    .trapnz(x, cranelift_codegen::ir::TrapCode::unwrap_user(1));
            }
        }
    }
}

/// Generate a random scalar function `%f` from a seed. Returns it with its
/// parameter types.
pub fn generate(seed: u64) -> (Function, Vec<Type>) {
    generate_with(
        seed,
        Features {
            simd: false,
            i128: false,
        },
    )
}

/// Like [`generate`], with vector types and operations too.
pub fn generate_simd(seed: u64) -> (Function, Vec<Type>) {
    generate_with(
        seed,
        Features {
            simd: true,
            i128: false,
        },
    )
}

/// Like [`generate`], with `i128` values too.
pub fn generate_i128(seed: u64) -> (Function, Vec<Type>) {
    generate_with(
        seed,
        Features {
            simd: false,
            i128: true,
        },
    )
}

fn generate_with(seed: u64, features: Features) -> (Function, Vec<Type>) {
    let simd = features.simd;
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let mut all: Vec<Type> = TYPES.to_vec();
    if simd {
        all.extend(VTYPES);
    }
    if features.i128 {
        all.extend([types::I128, types::I128]);
    }
    let nparams = 1 + rng.below(4) as usize;
    let params: Vec<Type> = (0..nparams).map(|_| rng.pick(&all)).collect();
    let nvars = 2 + rng.below(5) as usize;
    let var_tys: Vec<Type> = (0..nvars).map(|_| rng.pick(&all)).collect();

    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.extend(params.iter().map(|&t| AbiParam::new(t)));
    sig.returns
        .extend(var_tys.iter().map(|&t| AbiParam::new(t)));
    let mut func = Function::with_name_signature(UserFuncName::testcase("f"), sig);
    let mut ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut func, &mut ctx);
    let slot = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        SLOT_SIZE,
        3,
    ));
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let pvals = b.block_params(entry).to_vec();

    let budget = std::env::var("FUZZ_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let mut g = Gen {
        rng: &mut rng,
        b,
        vars: vec![],
        slot,
        budget,
        simd,
        wide: features.i128,
    };
    // Zero the slot so loads are deterministic.
    for off in (0..SLOT_SIZE).step_by(8) {
        let z = g.b.ins().iconst(types::I64, 0);
        let a = g.b.ins().stack_addr(types::I64, slot, 0);
        g.b.ins().store(MemFlagsData::trusted(), z, a, off as i32);
    }
    for (i, &ty) in var_tys.iter().enumerate() {
        let var = g.b.declare_var(ty);
        let init = match pvals.iter().position(|&p| g.b.func.dfg.value_type(p) == ty) {
            Some(j) if g.rng.chance(70) => pvals[j],
            _ => g.constant(ty),
        };
        g.b.def_var(var, init);
        g.vars.push((var, ty));
        let _ = i;
    }
    for (j, &p) in pvals.iter().enumerate() {
        let var = g.b.declare_var(params[j]);
        g.b.def_var(var, p);
        g.vars.push((var, params[j]));
    }
    g.stmts(3);
    while g.budget > 0 && g.budget < 1000 {
        g.budget -= 1;
        g.stmt(3);
    }
    let rets: Vec<Value> = (0..nvars).map(|i| g.b.use_var(g.vars[i].0)).collect();
    g.b.ins().return_(&rets);
    g.b.seal_all_blocks();
    g.b.finalize(cranelift_codegen::isa::TargetFrontendConfig {
        default_call_conv: CallConv::SystemV,
        pointer_width: target_lexicon::PointerWidth::U64,
        page_size_align_log2: 12,
    });
    (func, params)
}

/// A random argument of type `ty`.
pub fn arg(rng: &mut Rng, ty: Type) -> DataValue {
    let i = rng.int();
    match ty {
        types::I8 => DataValue::I8(i as i8),
        types::I16 => DataValue::I16(i as i16),
        types::I32 => DataValue::I32(i as i32),
        types::I64 => DataValue::I64(i),
        types::I128 => DataValue::I128(((rng.int() as i128) << 64) ^ i as i128),
        types::F32 => DataValue::F32(Ieee32::with_float((i % 1000) as f32 / 8.0)),
        types::F64 => DataValue::F64(Ieee64::with_float((i % 100000) as f64 / 16.0)),
        _ => {
            let mut b = [0u8; 16];
            for chunk in b.chunks_mut(8) {
                chunk.copy_from_slice(&rng.int().to_le_bytes());
            }
            DataValue::V128(b)
        }
    }
}
