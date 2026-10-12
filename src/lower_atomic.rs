//! Lowering of atomic memory operations to the atomic instructions of the
//! wasm threads proposal.
//!
//! Wasm has atomic loads and stores, read-modify-write `add`, `sub`, `and`,
//! `or`, `xor` and `xchg`, and compare-exchange, on 8- to 64-bit values. The
//! other read-modify-write operations (`nand`, `umin`/`umax`,
//! `smin`/`smax`) become a compare-exchange loop:
//!
//! ```text
//! loop
//!   old = atomic.load p
//!   seen = cmpxchg p, old, f(old, x)
//!   br_if 0 (seen != old)     ;; another thread wrote in between
//! end
//! ```
//!
//! Big-endian accesses reverse the bytes of the values they load and store.
//! For read-modify-write that only works with a loop (an addition on
//! byte-reversed values carries the wrong way), so they use it too, except
//! for single bytes, where byte order does not matter.

use crate::emit::{Emitter, is64};
use crate::{Result, unsupported};
use cranelift_codegen::ir::{AtomicRmwOp, Inst, InstructionData as D, MemFlags, Opcode, Type, types};
use wasm_encoder::{BlockType, Instruction as I, MemArg, ValType};

impl Emitter<'_> {
    pub(crate) fn lower_atomic(&mut self, inst: Inst) -> Result<()> {
        let dfg = &self.func.dfg;
        let data = dfg.insts[inst];
        // The type of the value in memory.
        let ty = match data {
            D::StoreNoOffset { args: [x, _], .. } => dfg.value_type(x),
            _ => dfg
                .inst_results(inst)
                .first()
                .map_or(types::INVALID, |&r| dfg.value_type(r)),
        };
        match data {
            D::NullAry {
                opcode: Opcode::Fence,
            } => {
                self.ins(I::AtomicFence);
                return Ok(());
            }
            _ if !matches!(ty, types::I8 | types::I16 | types::I32 | types::I64) => {
                return unsupported(format!(
                    "atomic access of type {ty} in `{}`",
                    dfg.display_inst(inst)
                ));
            }
            D::LoadNoOffset {
                opcode: Opcode::AtomicLoad,
                flags,
                ..
            } => {
                self.address()?;
                let m = self.atomic_memarg(ty);
                self.ins(atomic_load(ty, m));
                if self.big_endian(flags) {
                    self.bswap(ty)?;
                }
            }
            D::StoreNoOffset {
                opcode: Opcode::AtomicStore,
                flags,
                ..
            } => {
                self.address()?;
                self.operand()?;
                if self.big_endian(flags) {
                    self.bswap(ty)?;
                }
                let m = self.atomic_memarg(ty);
                self.ins(match ty {
                    types::I8 => I::I32AtomicStore8(m),
                    types::I16 => I::I32AtomicStore16(m),
                    types::I32 => I::I32AtomicStore(m),
                    _ => I::I64AtomicStore(m),
                });
            }
            D::AtomicCas { flags, .. } => {
                let big = self.big_endian(flags);
                self.address()?;
                for _ in 0..2 {
                    self.operand()?;
                    if big {
                        self.bswap(ty)?;
                    }
                }
                let m = self.atomic_memarg(ty);
                self.ins(cmpxchg(ty, m));
                if big {
                    self.bswap(ty)?;
                }
            }
            D::AtomicRmw { op, flags, .. } => self.atomic_rmw(op, flags, ty)?,
            _ => {
                return unsupported(format!("instruction `{}`", dfg.display_inst(inst)));
            }
        }
        Ok(())
    }

    fn atomic_memarg(&self, ty: Type) -> MemArg {
        // Atomic accesses must declare their natural alignment.
        MemArg {
            offset: 0,
            align: ty.bytes().trailing_zeros(),
            memory_index: self.config.module.memory_index,
        }
    }

    fn atomic_rmw(&mut self, op: AtomicRmwOp, flags: MemFlags, ty: Type) -> Result<()> {
        let big = self.big_endian(flags) && ty != types::I8;
        if !big && let Some(native) = native_rmw(op, ty, self.atomic_memarg(ty)) {
            self.address()?;
            self.operand()?;
            self.ins(native);
            return Ok(());
        }

        let vt = if is64(ty) { ValType::I64 } else { ValType::I32 };
        let ptr = self.config.module.ptr_ty();
        let p = self.scratch(ptr);
        let x = self.scratch(vt);
        let old = self.scratch(vt);
        // The value as stored in memory, which differs from `old` when
        // big-endian.
        let raw = if big { self.scratch(vt) } else { old };
        let m = self.atomic_memarg(ty);
        self.address()?;
        self.ins(I::LocalSet(p));
        self.operand()?;
        self.ins(I::LocalSet(x));

        self.ins(I::Loop(BlockType::Empty));
        self.ins(I::LocalGet(p));
        self.ins(atomic_load(ty, m));
        if big {
            self.ins(I::LocalTee(raw));
            self.bswap(ty)?;
        }
        self.ins(I::LocalSet(old));
        self.ins(I::LocalGet(p));
        self.ins(I::LocalGet(raw));
        self.rmw_value(op, ty, old, x);
        if big {
            self.bswap(ty)?;
        }
        self.ins(cmpxchg(ty, m));
        self.ins(I::LocalGet(raw));
        self.ins(if is64(ty) { I::I64Ne } else { I::I32Ne });
        self.ins(I::BrIf(0));
        self.ins(I::End);
        self.ins(I::LocalGet(old));

        self.release(ptr, p);
        self.release(vt, x);
        self.release(vt, old);
        if big {
            self.release(vt, raw);
        }
        Ok(())
    }

    /// Push `op(old, x)` for the locals `old` and `x`, zero-extended for
    /// narrow types.
    fn rmw_value(&mut self, op: AtomicRmwOp, ty: Type, old: u32, x: u32) {
        let wide = is64(ty);
        let pick = |a: I<'static>, b: I<'static>| if wide { b } else { a };
        if op == AtomicRmwOp::Xchg {
            self.ins(I::LocalGet(x));
            return;
        }
        self.ins(I::LocalGet(old));
        self.ins(I::LocalGet(x));
        match op {
            AtomicRmwOp::Add => self.ins(pick(I::I32Add, I::I64Add)),
            AtomicRmwOp::Sub => self.ins(pick(I::I32Sub, I::I64Sub)),
            AtomicRmwOp::And => self.ins(pick(I::I32And, I::I64And)),
            AtomicRmwOp::Or => self.ins(pick(I::I32Or, I::I64Or)),
            AtomicRmwOp::Xor => self.ins(pick(I::I32Xor, I::I64Xor)),
            AtomicRmwOp::Xchg => unreachable!("handled above"),
            AtomicRmwOp::Nand => {
                self.ins(pick(I::I32And, I::I64And));
                self.ins(pick(I::I32Const(-1), I::I64Const(-1)));
                self.ins(pick(I::I32Xor, I::I64Xor));
            }
            AtomicRmwOp::Umin | AtomicRmwOp::Umax | AtomicRmwOp::Smin | AtomicRmwOp::Smax => {
                // select(old, x, old cmp x), comparing sign-extended values
                // for the signed forms.
                let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
                self.ins(I::LocalGet(old));
                if signed {
                    self.sext(ty);
                }
                self.ins(I::LocalGet(x));
                if signed {
                    self.sext(ty);
                }
                self.ins(match (op, wide) {
                    (AtomicRmwOp::Umin, false) => I::I32LtU,
                    (AtomicRmwOp::Umin, true) => I::I64LtU,
                    (AtomicRmwOp::Umax, false) => I::I32GtU,
                    (AtomicRmwOp::Umax, true) => I::I64GtU,
                    (AtomicRmwOp::Smin, false) => I::I32LtS,
                    (AtomicRmwOp::Smin, true) => I::I64LtS,
                    (AtomicRmwOp::Smax, false) => I::I32GtS,
                    _ => I::I64GtS,
                });
                self.ins(I::Select);
            }
        }
        self.mask(ty);
    }
}

fn atomic_load(ty: Type, m: MemArg) -> I<'static> {
    match ty {
        types::I8 => I::I32AtomicLoad8U(m),
        types::I16 => I::I32AtomicLoad16U(m),
        types::I32 => I::I32AtomicLoad(m),
        _ => I::I64AtomicLoad(m),
    }
}

fn cmpxchg(ty: Type, m: MemArg) -> I<'static> {
    match ty {
        types::I8 => I::I32AtomicRmw8CmpxchgU(m),
        types::I16 => I::I32AtomicRmw16CmpxchgU(m),
        types::I32 => I::I32AtomicRmwCmpxchg(m),
        _ => I::I64AtomicRmwCmpxchg(m),
    }
}

/// The wasm instruction for a read-modify-write, if there is one.
fn native_rmw(op: AtomicRmwOp, ty: Type, m: MemArg) -> Option<I<'static>> {
    use AtomicRmwOp::*;
    Some(match (op, ty) {
        (Add, types::I8) => I::I32AtomicRmw8AddU(m),
        (Add, types::I16) => I::I32AtomicRmw16AddU(m),
        (Add, types::I32) => I::I32AtomicRmwAdd(m),
        (Add, _) => I::I64AtomicRmwAdd(m),
        (Sub, types::I8) => I::I32AtomicRmw8SubU(m),
        (Sub, types::I16) => I::I32AtomicRmw16SubU(m),
        (Sub, types::I32) => I::I32AtomicRmwSub(m),
        (Sub, _) => I::I64AtomicRmwSub(m),
        (And, types::I8) => I::I32AtomicRmw8AndU(m),
        (And, types::I16) => I::I32AtomicRmw16AndU(m),
        (And, types::I32) => I::I32AtomicRmwAnd(m),
        (And, _) => I::I64AtomicRmwAnd(m),
        (Or, types::I8) => I::I32AtomicRmw8OrU(m),
        (Or, types::I16) => I::I32AtomicRmw16OrU(m),
        (Or, types::I32) => I::I32AtomicRmwOr(m),
        (Or, _) => I::I64AtomicRmwOr(m),
        (Xor, types::I8) => I::I32AtomicRmw8XorU(m),
        (Xor, types::I16) => I::I32AtomicRmw16XorU(m),
        (Xor, types::I32) => I::I32AtomicRmwXor(m),
        (Xor, _) => I::I64AtomicRmwXor(m),
        (Xchg, types::I8) => I::I32AtomicRmw8XchgU(m),
        (Xchg, types::I16) => I::I32AtomicRmw16XchgU(m),
        (Xchg, types::I32) => I::I32AtomicRmwXchg(m),
        (Xchg, _) => I::I64AtomicRmwXchg(m),
        _ => return None,
    })
}
