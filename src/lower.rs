//! Lowering of individual Cranelift instructions to wasm instructions.
//!
//! Each case takes its operands with [`Emitter::operand`], which pushes them
//! in the order given by [`crate::expr_trees::operands`].

use crate::emit::{Emitter, is64, narrow_bits, w};
use crate::{Name, Result, SymbolValue, unsupported, wasm_type};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{Inst, InstructionData as D, Opcode, Type, types};
use wasm_encoder::{Instruction as I, MemArg, ValType};

impl Emitter<'_> {
    pub(crate) fn lower(&mut self, inst: Inst) -> Result<()> {
        if crate::lower_simd::is_simd(self.func, inst) {
            return self.lower_simd(inst);
        }
        let func = self.func;
        let dfg = &func.dfg;
        let data = dfg.insts[inst];
        let op = data.opcode();
        let args = dfg.inst_args(inst);
        let rty = dfg.inst_results(inst).first().map(|&r| dfg.value_type(r));
        let ty = rty.unwrap_or(types::INVALID);
        let aty = args
            .first()
            .map(|&a| dfg.value_type(a))
            .unwrap_or(types::INVALID);

        match (op, data) {
            (Opcode::Nop | Opcode::Debugtrap | Opcode::SequencePoint, _) => {}

            (Opcode::Iconst | Opcode::F32const | Opcode::F64const, _) => self.constant(inst)?,

            // Integer arithmetic.
            (Opcode::Iadd | Opcode::Isub | Opcode::Imul, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::Iadd => w!(ty, I32Add, I64Add),
                    Opcode::Isub => w!(ty, I32Sub, I64Sub),
                    _ => w!(ty, I32Mul, I64Mul),
                });
                self.mask(ty);
            }
            (Opcode::Band | Opcode::Bor | Opcode::Bxor, _) if ty.is_int() => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::Band => w!(ty, I32And, I64And),
                    Opcode::Bor => w!(ty, I32Or, I64Or),
                    _ => w!(ty, I32Xor, I64Xor),
                });
            }
            (Opcode::Band | Opcode::Bor | Opcode::Bxor | Opcode::Bnot | Opcode::Bitselect, _)
                if matches!(ty, types::F32 | types::F64) =>
            {
                // Bitwise operations on floats act on their bits.
                let wide = ty == types::F64;
                let (ity, vt) = if wide {
                    (types::I64, ValType::I64)
                } else {
                    (types::I32, ValType::I32)
                };
                let bits = |e: &mut Self| -> Result<()> {
                    e.operand()?;
                    e.ins(w!(ty, I32ReinterpretF32, I64ReinterpretF64));
                    Ok(())
                };
                let ones = if wide { I::I64Const(-1) } else { I::I32Const(-1) };
                match op {
                    Opcode::Bnot => {
                        bits(self)?;
                        self.ins(ones);
                        self.ins(w!(ity, I32Xor, I64Xor));
                    }
                    Opcode::Bitselect => {
                        // (c & x) | (~c & y)
                        let c = self.scratch(vt);
                        bits(self)?;
                        self.ins(I::LocalTee(c));
                        bits(self)?;
                        self.ins(w!(ity, I32And, I64And));
                        bits(self)?;
                        self.ins(I::LocalGet(c));
                        self.ins(ones);
                        self.ins(w!(ity, I32Xor, I64Xor));
                        self.ins(w!(ity, I32And, I64And));
                        self.ins(w!(ity, I32Or, I64Or));
                        self.release(vt, c);
                    }
                    _ => {
                        bits(self)?;
                        bits(self)?;
                        self.ins(match op {
                            Opcode::Band => w!(ity, I32And, I64And),
                            Opcode::Bor => w!(ity, I32Or, I64Or),
                            _ => w!(ity, I32Xor, I64Xor),
                        });
                    }
                }
                self.ins(w!(ty, F32ReinterpretI32, F64ReinterpretI64));
            }
            (Opcode::Bnot | Opcode::Ineg, _) if ty.is_int() => {
                self.operand()?;
                if is64(ty) {
                    self.ins(I::I64Const(-1));
                } else {
                    self.ins(I::I32Const(-1));
                }
                self.ins(match (op, is64(ty)) {
                    (Opcode::Bnot, false) => I::I32Xor,
                    (Opcode::Bnot, true) => I::I64Xor,
                    (_, false) => I::I32Mul,
                    (_, true) => I::I64Mul,
                });
                self.mask(ty);
            }
            (Opcode::Ishl | Opcode::Ushr | Opcode::Sshr, _) => {
                self.operand()?;
                if op == Opcode::Sshr {
                    self.sext(ty);
                }
                self.shift_amount(ty)?;
                self.ins(match op {
                    Opcode::Ishl => w!(ty, I32Shl, I64Shl),
                    Opcode::Ushr => w!(ty, I32ShrU, I64ShrU),
                    _ => w!(ty, I32ShrS, I64ShrS),
                });
                if op != Opcode::Ushr {
                    self.mask(ty);
                }
            }
            (Opcode::Rotl | Opcode::Rotr, _) => {
                self.operand()?;
                if let Some(bits) = narrow_bits(ty) {
                    let x = self.scratch(ValType::I32);
                    let n = self.scratch(ValType::I32);
                    self.ins(I::LocalTee(x));
                    self.shift_amount(ty)?;
                    self.ins(I::LocalTee(n));
                    let (first, second) = if op == Opcode::Rotl {
                        (I::I32Shl, I::I32ShrU)
                    } else {
                        (I::I32ShrU, I::I32Shl)
                    };
                    self.ins(first);
                    self.ins(I::LocalGet(x));
                    self.ins(I::I32Const(bits as i32));
                    self.ins(I::LocalGet(n));
                    self.ins(I::I32Sub);
                    self.ins(second);
                    self.ins(I::I32Or);
                    self.mask(ty);
                    self.release(ValType::I32, x);
                    self.release(ValType::I32, n);
                } else {
                    self.shift_amount(ty)?;
                    self.ins(if op == Opcode::Rotl {
                        w!(ty, I32Rotl, I64Rotl)
                    } else {
                        w!(ty, I32Rotr, I64Rotr)
                    });
                }
            }
            (Opcode::Udiv | Opcode::Urem, _) => {
                self.operand()?;
                self.operand()?;
                self.ins(if op == Opcode::Udiv {
                    w!(ty, I32DivU, I64DivU)
                } else {
                    w!(ty, I32RemU, I64RemU)
                });
            }
            (Opcode::Sdiv | Opcode::Srem, _) => {
                self.operand()?;
                self.sext(ty);
                self.operand()?;
                self.sext(ty);
                if op == Opcode::Sdiv {
                    self.ins(w!(ty, I32DivS, I64DivS));
                    if let Some(bits) = narrow_bits(ty) {
                        // MIN / -1 does not overflow in i32; trap explicitly.
                        let s = self.scratch(ValType::I32);
                        self.ins(I::LocalTee(s));
                        self.ins(I::I32Const(1 << (bits - 1)));
                        self.ins(I::I32Eq);
                        self.trap_if();
                        self.ins(I::LocalGet(s));
                        self.release(ValType::I32, s);
                    }
                } else {
                    self.ins(w!(ty, I32RemS, I64RemS));
                }
                self.mask(ty);
            }
            (Opcode::Umulhi | Opcode::Smulhi, _) => {
                let signed = op == Opcode::Smulhi;
                match ty {
                    types::I8 | types::I16 => {
                        let bits = narrow_bits(ty).unwrap();
                        self.operand()?;
                        if signed {
                            self.sext(ty);
                        }
                        self.operand()?;
                        if signed {
                            self.sext(ty);
                        }
                        self.ins(I::I32Mul);
                        self.ins(I::I32Const(bits as i32));
                        self.ins(if signed { I::I32ShrS } else { I::I32ShrU });
                        self.mask(ty);
                    }
                    types::I32 => {
                        let ext = if signed {
                            I::I64ExtendI32S
                        } else {
                            I::I64ExtendI32U
                        };
                        self.operand()?;
                        self.ins(ext.clone());
                        self.operand()?;
                        self.ins(ext);
                        self.ins(I::I64Mul);
                        self.ins(I::I64Const(32));
                        self.ins(I::I64ShrU);
                        self.ins(I::I32WrapI64);
                    }
                    types::I64 => self.mulhi64(signed)?,
                    _ => return unsupported(format!("{op}.{ty}")),
                }
            }
            (Opcode::Bswap | Opcode::Bitrev, _) => {
                self.operand()?;
                if op == Opcode::Bitrev {
                    let wide = is64(ty);
                    self.swap_bits(wide, 1, 0x5555_5555_5555_5555);
                    self.swap_bits(wide, 2, 0x3333_3333_3333_3333);
                    self.swap_bits(wide, 4, 0x0f0f_0f0f_0f0f_0f0f);
                }
                self.bswap(ty)?;
            }
            (Opcode::Iabs, _) => {
                self.operand()?;
                self.sext(ty);
                let vt = wasm_type(ty)?;
                let s = self.scratch(vt);
                let m = self.scratch(vt);
                self.ins(I::LocalTee(s));
                if is64(ty) {
                    self.ins(I::I64Const(63));
                    self.ins(I::I64ShrS);
                } else {
                    self.ins(I::I32Const(31));
                    self.ins(I::I32ShrS);
                }
                self.ins(I::LocalTee(m));
                self.ins(I::LocalGet(s));
                self.ins(w!(ty, I32Xor, I64Xor));
                self.ins(I::LocalGet(m));
                self.ins(w!(ty, I32Sub, I64Sub));
                self.mask(ty);
                self.release(vt, s);
                self.release(vt, m);
            }
            (Opcode::Smin | Opcode::Smax | Opcode::Umin | Opcode::Umax, _) if ty.is_int() => {
                let signed = matches!(op, Opcode::Smin | Opcode::Smax);
                let vt = wasm_type(ty)?;
                let a = self.scratch(vt);
                let b = self.scratch(vt);
                self.operand()?;
                if signed {
                    self.sext(ty);
                }
                self.ins(I::LocalTee(a));
                self.operand()?;
                if signed {
                    self.sext(ty);
                }
                self.ins(I::LocalTee(b));
                self.ins(I::LocalGet(a));
                self.ins(I::LocalGet(b));
                self.ins(match op {
                    Opcode::Smin => w!(ty, I32LtS, I64LtS),
                    Opcode::Smax => w!(ty, I32GtS, I64GtS),
                    Opcode::Umin => w!(ty, I32LtU, I64LtU),
                    _ => w!(ty, I32GtU, I64GtU),
                });
                self.ins(I::Select);
                if signed {
                    self.mask(ty);
                }
                self.release(vt, a);
                self.release(vt, b);
            }
            (Opcode::Clz | Opcode::Ctz | Opcode::Popcnt, _) => {
                self.operand()?;
                let bits = narrow_bits(ty);
                if let (Opcode::Ctz, Some(bits)) = (op, bits) {
                    self.ins(I::I32Const(1 << bits));
                    self.ins(I::I32Or);
                }
                self.ins(match op {
                    Opcode::Clz => w!(ty, I32Clz, I64Clz),
                    Opcode::Ctz => w!(ty, I32Ctz, I64Ctz),
                    _ => w!(ty, I32Popcnt, I64Popcnt),
                });
                if let (Opcode::Clz, Some(bits)) = (op, bits) {
                    self.ins(I::I32Const(32 - bits as i32));
                    self.ins(I::I32Sub);
                }
            }
            (Opcode::Cls, _) => {
                self.operand()?;
                self.sext(ty);
                let vt = wasm_type(ty)?;
                let s = self.scratch(vt);
                self.ins(I::LocalTee(s));
                if is64(ty) {
                    self.ins(I::I64Const(63));
                    self.ins(I::I64ShrS);
                    self.ins(I::LocalGet(s));
                    self.ins(I::I64Xor);
                    self.ins(I::I64Clz);
                    self.ins(I::I64Const(1));
                    self.ins(I::I64Sub);
                } else {
                    self.ins(I::I32Const(31));
                    self.ins(I::I32ShrS);
                    self.ins(I::LocalGet(s));
                    self.ins(I::I32Xor);
                    self.ins(I::I32Clz);
                    self.ins(I::I32Const(
                        1 + narrow_bits(ty).map_or(0, |b| 32 - b as i32),
                    ));
                    self.ins(I::I32Sub);
                }
                self.release(vt, s);
            }
            (Opcode::Icmp, D::IntCompare { cond, .. }) => {
                let signed = matches!(
                    cond,
                    IntCC::SignedLessThan
                        | IntCC::SignedLessThanOrEqual
                        | IntCC::SignedGreaterThan
                        | IntCC::SignedGreaterThanOrEqual
                );
                self.operand()?;
                if signed {
                    self.sext(aty);
                }
                self.operand()?;
                if signed {
                    self.sext(aty);
                }
                self.ins(match cond {
                    IntCC::Equal => w!(aty, I32Eq, I64Eq),
                    IntCC::NotEqual => w!(aty, I32Ne, I64Ne),
                    IntCC::SignedLessThan => w!(aty, I32LtS, I64LtS),
                    IntCC::SignedLessThanOrEqual => w!(aty, I32LeS, I64LeS),
                    IntCC::SignedGreaterThan => w!(aty, I32GtS, I64GtS),
                    IntCC::SignedGreaterThanOrEqual => w!(aty, I32GeS, I64GeS),
                    IntCC::UnsignedLessThan => w!(aty, I32LtU, I64LtU),
                    IntCC::UnsignedLessThanOrEqual => w!(aty, I32LeU, I64LeU),
                    IntCC::UnsignedGreaterThan => w!(aty, I32GtU, I64GtU),
                    IntCC::UnsignedGreaterThanOrEqual => w!(aty, I32GeU, I64GeU),
                });
            }
            (Opcode::Bmask, _) => {
                self.operand()?;
                self.ins(w!(aty, I32Eqz, I64Eqz));
                self.ins(I::I32Eqz);
                if is64(ty) {
                    self.ins(I::I64ExtendI32U);
                    self.ins(I::I64Const(-1));
                    self.ins(I::I64Mul);
                } else {
                    self.ins(I::I32Const(-1));
                    self.ins(I::I32Mul);
                    self.mask(ty);
                }
            }
            (Opcode::Select | Opcode::SelectSpectreGuard, _) if !ty.is_vector() => {
                self.operand()?;
                self.operand()?;
                self.operand()?;
                self.int_to_cond(aty);
                self.ins(I::Select);
            }
            (Opcode::Bitselect, _) if ty.is_int() => {
                let vt = wasm_type(ty)?;
                let c = self.scratch(vt);
                self.operand()?;
                self.ins(I::LocalTee(c));
                self.operand()?;
                self.ins(w!(ty, I32And, I64And));
                self.operand()?;
                self.ins(I::LocalGet(c));
                if is64(ty) {
                    self.ins(I::I64Const(-1));
                    self.ins(I::I64Xor);
                } else {
                    self.ins(I::I32Const(-1));
                    self.ins(I::I32Xor);
                }
                self.ins(w!(ty, I32And, I64And));
                self.ins(w!(ty, I32Or, I64Or));
                self.release(vt, c);
            }
            (Opcode::UaddOverflowTrap, _) => {
                let vt = wasm_type(ty)?;
                let a = self.scratch(vt);
                let r = self.scratch(vt);
                self.operand()?;
                self.ins(I::LocalTee(a));
                self.operand()?;
                self.ins(w!(ty, I32Add, I64Add));
                self.ins(I::LocalTee(r));
                if let Some(bits) = narrow_bits(ty) {
                    self.ins(I::I32Const(((1u32 << bits) - 1) as i32));
                    self.ins(I::I32GtU);
                } else {
                    self.ins(I::LocalGet(a));
                    self.ins(w!(ty, I32LtU, I64LtU));
                }
                self.trap_if();
                self.ins(I::LocalGet(r));
                self.release(vt, a);
                self.release(vt, r);
            }
            (
                Opcode::UaddOverflow
                | Opcode::SaddOverflow
                | Opcode::UaddOverflowCin
                | Opcode::SaddOverflowCin
                | Opcode::UsubOverflow
                | Opcode::SsubOverflow
                | Opcode::UsubOverflowBin
                | Opcode::SsubOverflowBin
                | Opcode::UmulOverflow
                | Opcode::SmulOverflow,
                _,
            ) => self.overflow_op(op, ty)?,

            // Conversions.
            (Opcode::Uextend, _) => {
                self.operand()?;
                if is64(ty) && !is64(aty) {
                    self.ins(I::I64ExtendI32U);
                }
            }
            (Opcode::Sextend, _) => {
                self.operand()?;
                self.sext(aty);
                if is64(ty) && !is64(aty) {
                    self.ins(I::I64ExtendI32S);
                }
                self.mask(ty);
            }
            (Opcode::Ireduce, _) => {
                self.operand()?;
                if is64(aty) {
                    self.ins(I::I32WrapI64);
                }
                self.mask(ty);
            }
            (Opcode::Bitcast, _) if !ty.is_vector() => {
                self.operand()?;
                match (aty, ty) {
                    (a, b) if a == b => {}
                    (types::I32, types::F32) => self.ins(I::F32ReinterpretI32),
                    (types::F32, types::I32) => self.ins(I::I32ReinterpretF32),
                    (types::I64, types::F64) => self.ins(I::F64ReinterpretI64),
                    (types::F64, types::I64) => self.ins(I::I64ReinterpretF64),
                    _ => return unsupported(format!("bitcast {aty} -> {ty}")),
                }
            }
            (Opcode::Fpromote, _) if ty == types::F64 && aty == types::F32 => {
                self.operand()?;
                self.ins(I::F64PromoteF32);
            }
            (Opcode::Fdemote, _) if ty == types::F32 && aty == types::F64 => {
                self.operand()?;
                self.ins(I::F32DemoteF64);
            }
            (Opcode::FcvtFromSint | Opcode::FcvtFromUint, _) if !ty.is_vector() => {
                let signed = op == Opcode::FcvtFromSint;
                self.operand()?;
                if signed {
                    self.sext(aty);
                }
                self.ins(match (ty, is64(aty), signed) {
                    (types::F32, false, true) => I::F32ConvertI32S,
                    (types::F32, false, false) => I::F32ConvertI32U,
                    (types::F32, true, true) => I::F32ConvertI64S,
                    (types::F32, true, false) => I::F32ConvertI64U,
                    (types::F64, false, true) => I::F64ConvertI32S,
                    (types::F64, false, false) => I::F64ConvertI32U,
                    (types::F64, true, true) => I::F64ConvertI64S,
                    (types::F64, true, false) => I::F64ConvertI64U,
                    _ => return unsupported(format!("{op} {aty} -> {ty}")),
                });
            }
            (Opcode::FcvtToSint | Opcode::FcvtToUint, _) => {
                let signed = op == Opcode::FcvtToSint;
                let f64 = match aty {
                    types::F32 => false,
                    types::F64 => true,
                    _ => return unsupported(format!("{op} {aty}")),
                };
                self.operand()?;
                self.ins(match (is64(ty), f64, signed) {
                    (false, false, true) => I::I32TruncF32S,
                    (false, false, false) => I::I32TruncF32U,
                    (false, true, true) => I::I32TruncF64S,
                    (false, true, false) => I::I32TruncF64U,
                    (true, false, true) => I::I64TruncF32S,
                    (true, false, false) => I::I64TruncF32U,
                    (true, true, true) => I::I64TruncF64S,
                    (true, true, false) => I::I64TruncF64U,
                });
                if let Some(bits) = narrow_bits(ty) {
                    // Trap if the result does not fit the narrow type.
                    let s = self.scratch(ValType::I32);
                    self.ins(I::LocalTee(s));
                    if signed {
                        self.ins(I::LocalGet(s));
                        self.sext(ty);
                        self.ins(I::I32Ne);
                    } else {
                        self.ins(I::I32Const(((1u32 << bits) - 1) as i32));
                        self.ins(I::I32GtU);
                    }
                    self.trap_if();
                    self.ins(I::LocalGet(s));
                    self.mask(ty);
                    self.release(ValType::I32, s);
                }
            }
            (Opcode::FcvtToSintSat | Opcode::FcvtToUintSat, _)
                if matches!(ty, types::I32 | types::I64) =>
            {
                let signed = op == Opcode::FcvtToSintSat;
                self.operand()?;
                let f64 = aty == types::F64;
                self.ins(match (is64(ty), f64, signed) {
                    (false, false, true) => I::I32TruncSatF32S,
                    (false, false, false) => I::I32TruncSatF32U,
                    (false, true, true) => I::I32TruncSatF64S,
                    (false, true, false) => I::I32TruncSatF64U,
                    (true, false, true) => I::I64TruncSatF32S,
                    (true, false, false) => I::I64TruncSatF32U,
                    (true, true, true) => I::I64TruncSatF64S,
                    (true, true, false) => I::I64TruncSatF64U,
                });
            }
            (Opcode::FcvtToSintSat | Opcode::FcvtToUintSat, _) if narrow_bits(ty).is_some() => {
                // Saturate to i32, then clamp to the narrow range.
                let signed = op == Opcode::FcvtToSintSat;
                let bits = narrow_bits(ty).unwrap();
                self.operand()?;
                let f64 = aty == types::F64;
                self.ins(match (f64, signed) {
                    (false, true) => I::I32TruncSatF32S,
                    (false, false) => I::I32TruncSatF32U,
                    (true, true) => I::I32TruncSatF64S,
                    (true, false) => I::I32TruncSatF64U,
                });
                let s = self.scratch(ValType::I32);
                if signed {
                    let (min, max) = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1);
                    self.clamp_with(s, max, I::I32LtS);
                    self.clamp_with(s, min, I::I32GtS);
                    self.mask(ty);
                } else {
                    self.clamp_with(s, (1 << bits) - 1, I::I32LtU);
                }
                self.release(ValType::I32, s);
            }

            // Floating point.
            (
                Opcode::Fadd
                | Opcode::Fsub
                | Opcode::Fmul
                | Opcode::Fdiv
                | Opcode::Fmin
                | Opcode::Fmax
                | Opcode::Fcopysign,
                _,
            ) if matches!(ty, types::F32 | types::F64) => {
                self.operand()?;
                self.operand()?;
                self.ins(match op {
                    Opcode::Fadd => w!(ty, F32Add, F64Add),
                    Opcode::Fsub => w!(ty, F32Sub, F64Sub),
                    Opcode::Fmul => w!(ty, F32Mul, F64Mul),
                    Opcode::Fdiv => w!(ty, F32Div, F64Div),
                    Opcode::Fmin => w!(ty, F32Min, F64Min),
                    Opcode::Fmax => w!(ty, F32Max, F64Max),
                    _ => w!(ty, F32Copysign, F64Copysign),
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
            ) if matches!(ty, types::F32 | types::F64) => {
                self.operand()?;
                self.ins(match op {
                    Opcode::Fneg => w!(ty, F32Neg, F64Neg),
                    Opcode::Fabs => w!(ty, F32Abs, F64Abs),
                    Opcode::Sqrt => w!(ty, F32Sqrt, F64Sqrt),
                    Opcode::Ceil => w!(ty, F32Ceil, F64Ceil),
                    Opcode::Floor => w!(ty, F32Floor, F64Floor),
                    Opcode::Trunc => w!(ty, F32Trunc, F64Trunc),
                    _ => w!(ty, F32Nearest, F64Nearest),
                });
            }
            (Opcode::Fcmp, D::FloatCompare { cond, .. }) if !aty.is_vector() => {
                self.fcmp(cond, aty)?;
            }

            // Memory.
            (
                Opcode::Load
                | Opcode::Uload8
                | Opcode::Sload8
                | Opcode::Uload16
                | Opcode::Sload16
                | Opcode::Uload32
                | Opcode::Sload32,
                D::Load { flags, offset, .. },
            ) if self.big_endian(flags) => {
                // Load the bytes zero-extended, reverse them, then extend
                // and convert.
                let (bytes, signed) = match op {
                    Opcode::Load => (ty.bytes(), false),
                    Opcode::Uload8 => (1, false),
                    Opcode::Sload8 => (1, true),
                    Opcode::Uload16 => (2, false),
                    Opcode::Sload16 => (2, true),
                    Opcode::Uload32 => (4, false),
                    _ => (4, true),
                };
                let off = self.mem_address(offset.into())?;
                let m = self.memarg(off, flags, bytes);
                let raw = types::Type::int_with_byte_size(bytes as u16).unwrap();
                self.ins(match bytes {
                    1 => I::I32Load8U(m),
                    2 => I::I32Load16U(m),
                    4 => I::I32Load(m),
                    8 => I::I64Load(m),
                    _ => return unsupported(format!("{op}.{ty}")),
                });
                self.bswap(raw)?;
                if signed {
                    self.sext(raw);
                }
                match (raw, ty) {
                    (types::I32, types::F32) => self.ins(I::F32ReinterpretI32),
                    (types::I64, types::F64) => self.ins(I::F64ReinterpretI64),
                    (types::I64, types::I64) => {}
                    (_, types::I64) if signed => self.ins(I::I64ExtendI32S),
                    (_, types::I64) => self.ins(I::I64ExtendI32U),
                    (_, types::I8 | types::I16 | types::I32) => self.mask(ty),
                    _ => return unsupported(format!("{op}.{ty}")),
                }
            }
            (
                Opcode::Load
                | Opcode::Uload8
                | Opcode::Sload8
                | Opcode::Uload16
                | Opcode::Sload16
                | Opcode::Uload32
                | Opcode::Sload32,
                D::Load { flags, offset, .. },
            ) => {
                let off = self.mem_address(offset.into())?;
                let (size, i) = match (op, ty) {
                    (Opcode::Load, types::I8) => (1, I::I32Load8U as fn(MemArg) -> I<'static>),
                    (Opcode::Load, types::I16) => (2, I::I32Load16U as _),
                    (Opcode::Load, types::I32) => (4, I::I32Load as _),
                    (Opcode::Load, types::I64) => (8, I::I64Load as _),
                    (Opcode::Load, types::F32) => (4, I::F32Load as _),
                    (Opcode::Load, types::F64) => (8, I::F64Load as _),
                    (Opcode::Uload8, types::I16 | types::I32) => (1, I::I32Load8U as _),
                    (Opcode::Uload8, types::I64) => (1, I::I64Load8U as _),
                    (Opcode::Sload8, types::I16 | types::I32) => (1, I::I32Load8S as _),
                    (Opcode::Sload8, types::I64) => (1, I::I64Load8S as _),
                    (Opcode::Uload16, types::I32) => (2, I::I32Load16U as _),
                    (Opcode::Uload16, types::I64) => (2, I::I64Load16U as _),
                    (Opcode::Sload16, types::I32) => (2, I::I32Load16S as _),
                    (Opcode::Sload16, types::I64) => (2, I::I64Load16S as _),
                    (Opcode::Uload32, types::I64) => (4, I::I64Load32U as _),
                    (Opcode::Sload32, types::I64) => (4, I::I64Load32S as _),
                    _ => return unsupported(format!("{op}.{ty}")),
                };
                let m = self.memarg(off, flags, size);
                self.ins(i(m));
                self.mask(ty);
            }
            (
                Opcode::Store | Opcode::Istore8 | Opcode::Istore16 | Opcode::Istore32,
                D::Store {
                    args: [x, _],
                    flags,
                    offset,
                    ..
                },
            ) if self.big_endian(flags) => {
                // Convert to an integer of the stored width, then reverse
                // its bytes.
                let vty = dfg.value_type(x);
                let bytes = match op {
                    Opcode::Store => vty.bytes(),
                    Opcode::Istore8 => 1,
                    Opcode::Istore16 => 2,
                    _ => 4,
                };
                let off = self.mem_address(offset.into())?;
                self.operand()?;
                match vty {
                    types::F32 => self.ins(I::I32ReinterpretF32),
                    types::F64 => self.ins(I::I64ReinterpretF64),
                    types::I64 if bytes < 8 => self.ins(I::I32WrapI64),
                    t if t.is_int() && !t.is_vector() => {}
                    _ => return unsupported(format!("{op}.{vty}")),
                }
                let raw = types::Type::int_with_byte_size(bytes as u16).unwrap();
                self.bswap(raw)?;
                let m = self.memarg(off, flags, bytes);
                self.ins(match bytes {
                    1 => I::I32Store8(m),
                    2 => I::I32Store16(m),
                    4 => I::I32Store(m),
                    _ => I::I64Store(m),
                });
            }
            (
                Opcode::Store | Opcode::Istore8 | Opcode::Istore16 | Opcode::Istore32,
                D::Store {
                    args: [x, _],
                    flags,
                    offset,
                    ..
                },
            ) => {
                let off = self.mem_address(offset.into())?;
                self.operand()?;
                let vty = dfg.value_type(x);
                let (size, i) = match (op, vty) {
                    (Opcode::Store, types::I8) => (1, I::I32Store8 as fn(MemArg) -> I<'static>),
                    (Opcode::Store, types::I16) => (2, I::I32Store16 as _),
                    (Opcode::Store, types::I32) => (4, I::I32Store as _),
                    (Opcode::Store, types::I64) => (8, I::I64Store as _),
                    (Opcode::Store, types::F32) => (4, I::F32Store as _),
                    (Opcode::Store, types::F64) => (8, I::F64Store as _),
                    (Opcode::Istore8, types::I16 | types::I32) => (1, I::I32Store8 as _),
                    (Opcode::Istore8, types::I64) => (1, I::I64Store8 as _),
                    (Opcode::Istore16, types::I32) => (2, I::I32Store16 as _),
                    (Opcode::Istore16, types::I64) => (2, I::I64Store16 as _),
                    (Opcode::Istore32, types::I64) => (4, I::I64Store32 as _),
                    _ => return unsupported(format!("{op}.{vty}")),
                };
                let m = self.memarg(off, flags, size);
                self.ins(i(m));
            }
            (
                Opcode::StackAddr,
                D::StackAddr {
                    stack_slot, offset, ..
                },
            ) => {
                let (fp, _) = self.frame.expect("frame for stack slot");
                self.ins(I::LocalGet(fp));
                let off = self.slot_offset[stack_slot] as i64 + i64::from(i32::from(offset));
                self.ptr_const(off);
                self.ptr_add();
                self.ptr_to(ty)?;
            }
            (Opcode::SymbolValue, D::UnaryGlobalValue { global_value, .. }) => {
                let gv = &func.global_values[global_value];
                let cranelift_codegen::ir::GlobalValueData::Symbol { name, offset, .. } = gv else {
                    return unsupported(format!("global value {gv}"));
                };
                let offset = offset.bits();
                match self.env.symbol_value(Name::resolve(func, name))? {
                    SymbolValue::Const(c) => self.iconst(ty, c as i64 + offset),
                    SymbolValue::Global(g) => {
                        self.ins(I::GlobalGet(g));
                        if offset != 0 {
                            self.iconst(ty, offset);
                            self.ins(w!(ty, I32Add, I64Add));
                        }
                    }
                }
            }
            (Opcode::FuncAddr, D::FuncAddr { func_ref, .. }) => {
                let name = &dfg.ext_funcs[func_ref].name;
                let slot = self.env.func_table_slot(Name::resolve(func, name))?;
                self.iconst(ty, slot as i64);
            }

            (Opcode::GetPinnedReg | Opcode::SetPinnedReg, _) => {
                let Some(g) = self.config.module.pinned_reg_global else {
                    return unsupported(format!("{op} without ModuleLayout::pinned_reg_global"));
                };
                if op == Opcode::GetPinnedReg {
                    self.ins(I::GlobalGet(g));
                    if ty == types::I32 {
                        self.ins(I::I32WrapI64);
                    }
                } else {
                    self.operand()?;
                    if aty == types::I32 {
                        self.ins(I::I64ExtendI32U);
                    }
                    self.ins(I::GlobalSet(g));
                }
            }

            (
                Opcode::AtomicLoad
                | Opcode::AtomicStore
                | Opcode::AtomicRmw
                | Opcode::AtomicCas
                | Opcode::Fence,
                _,
            ) => self.lower_atomic(inst)?,

            (Opcode::GetStackPointer | Opcode::GetFramePointer, _) => {
                match (self.frame, self.config.module.stack_pointer_global) {
                    (Some((fp, _)), _) => self.ins(I::LocalGet(fp)),
                    (None, Some(sp)) => self.ins(I::GlobalGet(sp)),
                    (None, None) => return unsupported(format!("{op} without a stack pointer")),
                }
                self.ptr_to(ty)?;
            }

            // Calls.
            (Opcode::Call | Opcode::ReturnCall, D::Call { func_ref, .. }) => {
                for _ in args {
                    self.operand()?;
                }
                let name = &dfg.ext_funcs[func_ref].name;
                let idx = self.env.func_index(Name::resolve(func, name))?;
                if op == Opcode::ReturnCall {
                    self.epilogue();
                    if self.config.engine.tail_calls {
                        self.ins(I::ReturnCall(idx));
                    } else {
                        self.ins(I::Call(idx));
                        self.ins(I::Return);
                    }
                } else {
                    self.ins(I::Call(idx));
                }
            }
            (
                Opcode::CallIndirect | Opcode::ReturnCallIndirect,
                D::CallIndirect { sig_ref, .. },
            ) => {
                for _ in &args[1..] {
                    self.operand()?;
                }
                self.operand()?;
                if is64(aty) {
                    self.ins(I::I32WrapI64);
                }
                let Some(table_index) = self.config.module.table_index else {
                    return unsupported("call_indirect without ModuleLayout::table_index");
                };
                let type_index = self.env.type_index(&dfg.signatures[sig_ref])?;
                if op == Opcode::ReturnCallIndirect {
                    self.epilogue();
                    if self.config.engine.tail_calls {
                        self.ins(I::ReturnCallIndirect {
                            type_index,
                            table_index,
                        });
                    } else {
                        self.ins(I::CallIndirect {
                            type_index,
                            table_index,
                        });
                        self.ins(I::Return);
                    }
                } else {
                    self.ins(I::CallIndirect {
                        type_index,
                        table_index,
                    });
                }
            }
            (Opcode::Return, _) => {
                for _ in args {
                    self.operand()?;
                }
                self.epilogue();
                self.ins(I::Return);
            }

            // Traps.
            (Opcode::Trap, _) => self.ins(I::Unreachable),
            (Opcode::Trapz | Opcode::Trapnz, _) => {
                self.operand()?;
                self.int_to_cond(aty);
                if op == Opcode::Trapz {
                    self.ins(I::I32Eqz);
                }
                self.trap_if();
            }

            _ => return unsupported(format!("instruction `{}`", dfg.display_inst(inst))),
        }
        Ok(())
    }

    /// Replace the `i32` on the stack by `bound` unless `cmp(value, bound)`
    /// holds; `s` is a scratch local.
    fn clamp_with(&mut self, s: u32, bound: i32, cmp: I<'static>) {
        self.ins(I::LocalTee(s));
        self.ins(I::I32Const(bound));
        self.ins(I::LocalGet(s));
        self.ins(I::I32Const(bound));
        self.ins(cmp);
        self.ins(I::Select);
    }

    /// Reverse the bytes of the integer of type `ty` on the stack. For `i8`
    /// and `i16` only the low bits of the `i32` are reversed, and the result
    /// is zero-extended.
    pub(crate) fn bswap(&mut self, ty: Type) -> Result<()> {
        match ty {
            types::I8 => {}
            // Byte-reverse as 32 bits, then shift a narrow result down.
            types::I16 | types::I32 => {
                let s = self.scratch(ValType::I32);
                self.ins(I::LocalTee(s));
                self.ins(I::I32Const(0x00ff_00ff));
                self.ins(I::I32And);
                self.ins(I::I32Const(8));
                self.ins(I::I32Rotr);
                self.ins(I::LocalGet(s));
                self.ins(I::I32Const(0xff00_ff00u32 as i32));
                self.ins(I::I32And);
                self.ins(I::I32Const(8));
                self.ins(I::I32Rotl);
                self.ins(I::I32Or);
                if ty == types::I16 {
                    self.ins(I::I32Const(16));
                    self.ins(I::I32ShrU);
                }
                self.release(ValType::I32, s);
            }
            types::I64 => {
                self.swap_bits(true, 8, 0x00ff_00ff_00ff_00ff);
                self.swap_bits(true, 16, 0x0000_ffff_0000_ffff);
                self.ins(I::I64Const(32));
                self.ins(I::I64Rotl);
            }
            _ => return unsupported(format!("bswap.{ty}")),
        }
        Ok(())
    }

    /// Swap adjacent `shift`-bit groups of the value on the stack:
    /// `((x >> shift) & mask) | ((x & mask) << shift)`.
    fn swap_bits(&mut self, wide: bool, shift: i64, mask: u64) {
        let vt = if wide { ValType::I64 } else { ValType::I32 };
        let s = self.scratch(vt);
        let (konst, and, or, shl, shr): (fn(i64) -> I<'static>, _, _, _, _) = if wide {
            (I::I64Const, I::I64And, I::I64Or, I::I64Shl, I::I64ShrU)
        } else {
            (
                |v| I::I32Const(v as i32),
                I::I32And,
                I::I32Or,
                I::I32Shl,
                I::I32ShrU,
            )
        };
        self.ins(I::LocalTee(s));
        self.ins(konst(shift));
        self.ins(shr);
        self.ins(konst(mask as i64));
        self.ins(and.clone());
        self.ins(I::LocalGet(s));
        self.ins(konst(mask as i64));
        self.ins(and);
        self.ins(konst(shift));
        self.ins(shl);
        self.ins(or);
        self.release(vt, s);
    }

    /// The high 64 bits of a 64x64-bit product, from 32-bit pieces:
    /// with `p1 = a.lo * b.hi` and `p2 = a.hi * b.lo`,
    /// `hi = a.hi*b.hi + p1.hi + p2.hi + (p0.hi + p1.lo + p2.lo).hi`.
    /// The signed version subtracts `b` if `a < 0` and `a` if `b < 0`.
    fn mulhi64(&mut self, signed: bool) -> Result<()> {
        let [a, b] = [(); 2].map(|_| self.scratch(ValType::I64));
        self.operand()?;
        self.ins(I::LocalSet(a));
        self.operand()?;
        self.ins(I::LocalSet(b));
        self.mulhi64_of(a, b, signed);
        self.release(ValType::I64, a);
        self.release(ValType::I64, b);
        Ok(())
    }

    /// Push the high half of the product of the `i64` locals `a` and `b`.
    fn mulhi64_of(&mut self, a: u32, b: u32, signed: bool) {
        let [p1, p2] = [(); 2].map(|_| self.scratch(ValType::I64));
        let lo32 = |e: &mut Self, l: u32| {
            e.ins(I::LocalGet(l));
            e.ins(I::I64Const(0xffff_ffff));
            e.ins(I::I64And);
        };
        let hi32 = |e: &mut Self, l: u32| {
            e.ins(I::LocalGet(l));
            e.ins(I::I64Const(32));
            e.ins(I::I64ShrU);
        };
        lo32(self, a);
        hi32(self, b);
        self.ins(I::I64Mul);
        self.ins(I::LocalSet(p1));
        hi32(self, a);
        lo32(self, b);
        self.ins(I::I64Mul);
        self.ins(I::LocalSet(p2));
        // a.hi*b.hi + p1.hi + p2.hi
        hi32(self, a);
        hi32(self, b);
        self.ins(I::I64Mul);
        hi32(self, p1);
        self.ins(I::I64Add);
        hi32(self, p2);
        self.ins(I::I64Add);
        // + (p0.hi + p1.lo + p2.lo).hi
        lo32(self, a);
        lo32(self, b);
        self.ins(I::I64Mul);
        self.ins(I::I64Const(32));
        self.ins(I::I64ShrU);
        lo32(self, p1);
        self.ins(I::I64Add);
        lo32(self, p2);
        self.ins(I::I64Add);
        self.ins(I::I64Const(32));
        self.ins(I::I64ShrU);
        self.ins(I::I64Add);
        if signed {
            for (x, y) in [(a, b), (b, a)] {
                // - (x < 0 ? y : 0)
                self.ins(I::LocalGet(y));
                self.ins(I::I64Const(0));
                self.ins(I::LocalGet(x));
                self.ins(I::I64Const(0));
                self.ins(I::I64LtS);
                self.ins(I::Select);
                self.ins(I::I64Sub);
            }
        }
        self.release(ValType::I64, p1);
        self.release(ValType::I64, p2);
    }

    /// Lower an arithmetic operation with an overflow flag: push the
    /// wrapped result, then the flag. The `_cin`/`_bin` forms add a carry or
    /// subtract a borrow, taken as 1 when nonzero.
    ///
    /// Narrow and 32-bit operations compute the exact result in a wider
    /// type; it overflowed if truncating and re-extending changes it. 64-bit
    /// operations derive the flag from the operands and the wrapped result.
    fn overflow_op(&mut self, op: Opcode, ty: Type) -> Result<()> {
        use Opcode::*;
        let signed = matches!(
            op,
            SaddOverflow | SaddOverflowCin | SsubOverflow | SsubOverflowBin | SmulOverflow
        );
        let carry = matches!(
            op,
            UaddOverflowCin | SaddOverflowCin | UsubOverflowBin | SsubOverflowBin
        );
        let kind = match op {
            UaddOverflow | SaddOverflow | UaddOverflowCin | SaddOverflowCin => Iadd,
            UsubOverflow | SsubOverflow | UsubOverflowBin | SsubOverflowBin => Isub,
            _ => Imul,
        };
        if ty == types::I64 {
            return self.overflow_op64(kind, signed, carry);
        }
        let narrow = narrow_bits(ty).is_some();
        if !narrow && ty != types::I32 {
            return unsupported(format!("{op}.{ty}"));
        }
        let (wide, wvt) = if narrow {
            (types::I32, ValType::I32)
        } else {
            (types::I64, ValType::I64)
        };
        let combine = match kind {
            Iadd => w!(wide, I32Add, I64Add),
            Isub => w!(wide, I32Sub, I64Sub),
            _ => w!(wide, I32Mul, I64Mul),
        };
        for _ in 0..2 {
            self.operand()?;
            match (narrow, signed) {
                (true, true) => self.sext(ty),
                (true, false) => {}
                (false, true) => self.ins(I::I64ExtendI32S),
                (false, false) => self.ins(I::I64ExtendI32U),
            }
        }
        self.ins(combine.clone());
        if carry {
            self.operand()?;
            self.ins(I::I32Const(0));
            self.ins(I::I32Ne);
            if !narrow {
                self.ins(I::I64ExtendI32U);
            }
            self.ins(combine);
        }
        let r = self.scratch(wvt);
        self.ins(I::LocalTee(r));
        if narrow {
            self.mask(ty);
        } else {
            self.ins(I::I32WrapI64);
        }
        // The flag: r != extend(truncate(r)).
        self.ins(I::LocalGet(r));
        self.ins(I::LocalGet(r));
        match (narrow, signed) {
            (true, true) => self.sext(ty),
            (true, false) => self.mask(ty),
            (false, true) => self.ins(I::I64Extend32S),
            (false, false) => {
                self.ins(I::I64Const(0xffff_ffff));
                self.ins(I::I64And);
            }
        }
        self.ins(w!(wide, I32Ne, I64Ne));
        self.release(wvt, r);
        Ok(())
    }

    fn overflow_op64(&mut self, kind: Opcode, signed: bool, carry: bool) -> Result<()> {
        let [x, y, r] = [(); 3].map(|_| self.scratch(ValType::I64));
        let c = self.scratch(ValType::I32);
        self.operand()?;
        self.ins(I::LocalSet(x));
        self.operand()?;
        self.ins(I::LocalSet(y));
        if carry {
            self.operand()?;
            self.ins(I::I32Const(0));
            self.ins(I::I32Ne);
            self.ins(I::LocalSet(c));
        }
        let get = |e: &mut Self, l: u32| e.ins(I::LocalGet(l));
        get(self, x);
        get(self, y);
        match kind {
            Opcode::Imul => {
                self.ins(I::I64Mul);
                self.ins(I::LocalTee(r));
                self.mulhi64_of(x, y, signed);
                // The high half must be the sign extension of the low half.
                if signed {
                    get(self, r);
                    self.ins(I::I64Const(63));
                    self.ins(I::I64ShrS);
                } else {
                    self.ins(I::I64Const(0));
                }
                self.ins(I::I64Ne);
            }
            _ => {
                let add = kind == Opcode::Iadd;
                let combine = if add { I::I64Add } else { I::I64Sub };
                self.ins(combine.clone());
                if carry {
                    get(self, c);
                    self.ins(I::I64ExtendI32U);
                    self.ins(combine);
                }
                self.ins(I::LocalTee(r));
                if signed {
                    // Signed overflow: the operands' signs make the true
                    // result's sign differ from r's.
                    //   add: (x ^ r) & (y ^ r) < 0
                    //   sub: (x ^ y) & (x ^ r) < 0
                    let (a, b) = if add { (x, r) } else { (x, y) };
                    get(self, a);
                    get(self, b);
                    self.ins(I::I64Xor);
                    get(self, if add { y } else { x });
                    get(self, r);
                    self.ins(I::I64Xor);
                    self.ins(I::I64And);
                    self.ins(I::I64Const(0));
                    self.ins(I::I64LtS);
                } else {
                    // Carry out: r < x (add) or borrow: x < y (sub). With a
                    // carry or borrow in, equality also overflows.
                    let (a, b) = if add { (r, x) } else { (x, y) };
                    get(self, a);
                    get(self, b);
                    self.ins(I::I64LtU);
                    if carry {
                        get(self, c);
                        get(self, a);
                        get(self, b);
                        self.ins(I::I64Eq);
                        self.ins(I::I32And);
                        self.ins(I::I32Or);
                    }
                }
            }
        }
        for l in [x, y, r] {
            self.release(ValType::I64, l);
        }
        self.release(ValType::I32, c);
        Ok(())
    }

    /// Push both operands and compare them.
    fn fcmp(&mut self, cond: FloatCC, ty: Type) -> Result<()> {
        let vt = wasm_type(ty)?;
        let simple = |cc| -> Option<(I<'static>, bool)> {
            Some(match cc {
                FloatCC::Equal => (w!(ty, F32Eq, F64Eq), false),
                FloatCC::NotEqual => (w!(ty, F32Ne, F64Ne), false),
                FloatCC::LessThan => (w!(ty, F32Lt, F64Lt), false),
                FloatCC::LessThanOrEqual => (w!(ty, F32Le, F64Le), false),
                FloatCC::GreaterThan => (w!(ty, F32Gt, F64Gt), false),
                FloatCC::GreaterThanOrEqual => (w!(ty, F32Ge, F64Ge), false),
                // Unordered-or-X is the negation of the ordered inverse.
                FloatCC::UnorderedOrLessThan => (w!(ty, F32Ge, F64Ge), true),
                FloatCC::UnorderedOrLessThanOrEqual => (w!(ty, F32Gt, F64Gt), true),
                FloatCC::UnorderedOrGreaterThan => (w!(ty, F32Le, F64Le), true),
                FloatCC::UnorderedOrGreaterThanOrEqual => (w!(ty, F32Lt, F64Lt), true),
                _ => return None,
            })
        };
        if let Some((i, negate)) = simple(cond) {
            self.operand()?;
            self.operand()?;
            self.ins(i);
            if negate {
                self.ins(I::I32Eqz);
            }
            return Ok(());
        }
        let sa = self.scratch(vt);
        let sb = self.scratch(vt);
        match cond {
            FloatCC::Ordered | FloatCC::Unordered => {
                // a == a (and|or) b == b, or the negation.
                let (cmp, comb) = if cond == FloatCC::Ordered {
                    (w!(ty, F32Eq, F64Eq), I::I32And)
                } else {
                    (w!(ty, F32Ne, F64Ne), I::I32Or)
                };
                self.operand()?;
                self.ins(I::LocalTee(sa));
                self.ins(I::LocalGet(sa));
                self.ins(cmp.clone());
                self.operand()?;
                self.ins(I::LocalTee(sb));
                self.ins(I::LocalGet(sb));
                self.ins(cmp);
                self.ins(comb);
            }
            FloatCC::OrderedNotEqual | FloatCC::UnorderedOrEqual => {
                self.operand()?;
                self.ins(I::LocalTee(sa));
                self.operand()?;
                self.ins(I::LocalTee(sb));
                self.ins(w!(ty, F32Lt, F64Lt));
                self.ins(I::LocalGet(sa));
                self.ins(I::LocalGet(sb));
                self.ins(w!(ty, F32Gt, F64Gt));
                self.ins(I::I32Or);
                if cond == FloatCC::UnorderedOrEqual {
                    self.ins(I::I32Eqz);
                }
            }
            _ => unreachable!(),
        }
        self.release(vt, sa);
        self.release(vt, sb);
        Ok(())
    }
}
