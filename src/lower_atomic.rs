//! Atomic memory operation lowering (wasm threads proposal).
//!
//! Wasm provides 8- to 64-bit atomic loads, stores, compare-exchange, and RMW
//! `add`, `sub`, `and`, `or`, `xor`, and `xchg`. Remaining RMW operations
//! (`nand`, `umin`/`umax`, `smin`/`smax`) and multi-byte big-endian RMWs lower
//! to a `cmpxchg` loop:
//!
//! ```text
//! loop
//!   old = atomic.load p
//!   seen = cmpxchg p, old, f(old, x)
//!   br_if 0 (seen != old)
//! end
//! ```
//!
//! Big-endian atomic loads, stores, and CAS byte-swap values around the native
//! little-endian wasm atomic instructions.

use crate::emit::{Emitter, is64, w};
use crate::locals::Local;
use crate::{Result, unsupported};
use cranelift_codegen::ir::{
    AtomicRmwOp, Inst, InstructionData as D, MemFlags, Opcode, Type, types,
};
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
            memory_index: self.config.module.memory_index.0,
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
        let [x, old] = self.scratches(vt);
        // The value as stored in memory, which differs from `old` when
        // big-endian.
        let raw = if big { self.scratch(vt) } else { old };
        let m = self.atomic_memarg(ty);
        self.address()?;
        self.local_set(p);
        self.operand()?;
        self.local_set(x);

        self.ins(I::Loop(BlockType::Empty));
        self.local_get(p);
        self.ins(atomic_load(ty, m));
        if big {
            self.local_tee(raw);
            self.bswap(ty)?;
        }
        self.local_set(old);
        self.local_get(p);
        self.local_get(raw);
        self.rmw_value(op, ty, old, x);
        if big {
            self.bswap(ty)?;
        }
        self.ins(cmpxchg(ty, m));
        self.local_get(raw);
        self.ins(w!(ty, I32Ne, I64Ne));
        self.ins(I::BrIf(0));
        self.ins(I::End);
        self.local_get(old);
        Ok(())
    }

    /// Push `op(old, x)` for the locals `old` and `x`, zero-extended for
    /// narrow types.
    fn rmw_value(&mut self, op: AtomicRmwOp, ty: Type, old: Local, x: Local) {
        if op == AtomicRmwOp::Xchg {
            self.local_get(x);
            return;
        }
        self.local_get(old);
        self.local_get(x);
        match op {
            AtomicRmwOp::Add => self.ins(w!(ty, I32Add, I64Add)),
            AtomicRmwOp::Sub => self.ins(w!(ty, I32Sub, I64Sub)),
            AtomicRmwOp::And => self.ins(w!(ty, I32And, I64And)),
            AtomicRmwOp::Or => self.ins(w!(ty, I32Or, I64Or)),
            AtomicRmwOp::Xor => self.ins(w!(ty, I32Xor, I64Xor)),
            AtomicRmwOp::Xchg => unreachable!("handled above"),
            AtomicRmwOp::Nand => {
                self.ins(w!(ty, I32And, I64And));
                self.iconst(ty, -1);
                self.ins(w!(ty, I32Xor, I64Xor));
            }
            AtomicRmwOp::Umin | AtomicRmwOp::Umax | AtomicRmwOp::Smin | AtomicRmwOp::Smax => {
                // select(old, x, old cmp x), comparing sign-extended values
                // for the signed forms.
                let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
                self.local_get(old);
                if signed {
                    self.sext(ty);
                }
                self.local_get(x);
                if signed {
                    self.sext(ty);
                }
                self.ins(match op {
                    AtomicRmwOp::Umin => w!(ty, I32LtU, I64LtU),
                    AtomicRmwOp::Umax => w!(ty, I32GtU, I64GtU),
                    AtomicRmwOp::Smin => w!(ty, I32LtS, I64LtS),
                    _ => w!(ty, I32GtS, I64GtS),
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
