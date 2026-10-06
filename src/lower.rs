//! Per-instruction lowering.
//!
//! `i8` and `i16` values are represented as zero-extended `i32`s. Operations
//! that can set bits above the type's width mask their result; operations
//! that depend on the sign bit sign-extend their inputs first.

use crate::analysis::{FuncInfo, is_remat};
use crate::locals::{Locals, NO_LOCAL};
use crate::stackify::Stackified;
use crate::{Config, Env, Name, Result, SymbolValue, unsupported, wasm_type};
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    Endianness, Function, Inst, InstructionData as D, Opcode, StackSlot, Type, Value, ValueDef,
    types,
};
use wasm_encoder::{BlockType, Encode, Instruction as I, MemArg, ValType};

/// An entry on the wasm label stack, used to compute branch depths.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Label {
    If,
    Loop(cranelift_codegen::ir::Block),
    BlockFollowedBy(cranelift_codegen::ir::Block),
    Dispatch,
    Other,
}

/// Pending structured-emission work; see `structure.rs`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Task {
    /// Emit a dominator subtree.
    Tree(cranelift_codegen::ir::Block),
    /// Move block arguments and transfer control along an edge.
    Branch {
        src: cranelift_codegen::ir::Block,
        call: cranelift_codegen::ir::BlockCall,
    },
    Else,
    /// Close the innermost label.
    End,
}

pub(crate) struct Emitter<'a> {
    pub func: &'a Function,
    pub config: &'a Config,
    pub env: &'a mut dyn Env,
    pub info: &'a FuncInfo,
    pub stack: &'a Stackified,
    pub locals: Locals,
    pub labels: Vec<Label>,
    pub tasks: Vec<Task>,
    /// Locals allocated during emission (scratch, frame pointer, label).
    pub extra: Vec<ValType>,
    /// Shadow-stack frame: (frame pointer local, frame size).
    pub frame: Option<(u32, u32)>,
    pub slot_offset: SecondaryMap<StackSlot, u32>,
    /// The label local, when emitting a dispatch loop.
    pub dispatch_label: Option<u32>,
    sink: Vec<u8>,
    pending_set: Option<u32>,
    scratch_free: Vec<(ValType, u32)>,
    #[cfg(debug_assertions)]
    expect: Vec<(Vec<Value>, usize)>,
}

fn is64(ty: Type) -> bool {
    ty == types::I64 || ty == types::F64
}

fn narrow_bits(ty: Type) -> Option<u32> {
    match ty {
        types::I8 => Some(8),
        types::I16 => Some(16),
        _ => None,
    }
}

macro_rules! w {
    ($ty:expr, $a:ident, $b:ident) => {
        if is64($ty) { I::$b } else { I::$a }
    };
}

impl<'a> Emitter<'a> {
    pub fn new(
        func: &'a Function,
        config: &'a Config,
        env: &'a mut dyn Env,
        info: &'a FuncInfo,
        stack: &'a Stackified,
        locals: Locals,
    ) -> Self {
        Emitter {
            func,
            config,
            env,
            info,
            stack,
            locals,
            labels: Vec::new(),
            tasks: Vec::new(),
            extra: Vec::new(),
            frame: None,
            slot_offset: SecondaryMap::new(),
            dispatch_label: None,
            sink: Vec::new(),
            pending_set: None,
            scratch_free: Vec::new(),
            #[cfg(debug_assertions)]
            expect: Vec::new(),
        }
    }

    // ---- Encoding -------------------------------------------------------

    /// Emit one instruction. `local.set x; local.get x` is fused into
    /// `local.tee x`.
    pub fn ins(&mut self, i: I<'_>) {
        if let Some(l) = self.pending_set.take() {
            if self.config.use_tee && matches!(i, I::LocalGet(g) if g == l) {
                I::LocalTee(l).encode(&mut self.sink);
                return;
            }
            I::LocalSet(l).encode(&mut self.sink);
        }
        if let I::LocalSet(l) = i {
            self.pending_set = Some(l);
            return;
        }
        i.encode(&mut self.sink);
    }

    pub fn finish(mut self) -> (Vec<u8>, Locals, Vec<ValType>) {
        if let Some(l) = self.pending_set.take() {
            I::LocalSet(l).encode(&mut self.sink);
        }
        (self.sink, self.locals, self.extra)
    }

    pub fn new_local(&mut self, ty: ValType) -> u32 {
        self.extra.push(ty);
        self.locals.next_index() + self.extra.len() as u32 - 1
    }

    fn scratch(&mut self, ty: ValType) -> u32 {
        if let Some(i) = self.scratch_free.iter().position(|&(t, _)| t == ty) {
            return self.scratch_free.swap_remove(i).1;
        }
        self.new_local(ty)
    }

    fn release(&mut self, ty: ValType, l: u32) {
        self.scratch_free.push((ty, l));
    }

    pub fn br_depth(&self, label: Label) -> u32 {
        let i = self
            .labels
            .iter()
            .rposition(|&l| l == label)
            .expect("branch target in scope");
        (self.labels.len() - 1 - i) as u32
    }

    // ---- Values ---------------------------------------------------------

    fn ty(&self, v: Value) -> Type {
        self.func.dfg.value_type(v)
    }

    pub fn local(&self, v: Value) -> u32 {
        let l = self.locals.local_of[v];
        debug_assert_ne!(l, NO_LOCAL, "{v} has no local");
        l
    }

    /// Push the value `v`.
    pub fn value(&mut self, v: Value) -> Result<()> {
        #[cfg(debug_assertions)]
        if let Some((exp, i)) = self.expect.last_mut() {
            assert_eq!(exp.get(*i), Some(&v), "operand order mismatch");
            *i += 1;
        }
        match self.func.dfg.value_def(v) {
            ValueDef::Result(d, _) if is_remat(self.func, d) => self.constant(d),
            ValueDef::Result(d, _) if self.stack.inlined[d] => self.inst(d),
            _ => {
                self.ins(I::LocalGet(self.local(v)));
                Ok(())
            }
        }
    }

    fn constant(&mut self, inst: Inst) -> Result<()> {
        let ty = self.ty(self.func.dfg.first_result(inst));
        match self.func.dfg.insts[inst] {
            D::UnaryImm { imm, .. } => self.iconst(ty, imm.bits()),
            D::UnaryIeee32 { imm, .. } => {
                self.ins(I::F32Const(wasm_encoder::Ieee32::new(imm.bits())))
            }
            D::UnaryIeee64 { imm, .. } => {
                self.ins(I::F64Const(wasm_encoder::Ieee64::new(imm.bits())))
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn iconst(&mut self, ty: Type, v: i64) {
        match ty {
            types::I64 => self.ins(I::I64Const(v)),
            types::I8 => self.ins(I::I32Const((v as u8) as i32)),
            types::I16 => self.ins(I::I32Const((v as u16) as i32)),
            _ => self.ins(I::I32Const(v as i32)),
        }
    }

    fn ptr_const(&mut self, v: i64) {
        if self.config.memory64 {
            self.ins(I::I64Const(v))
        } else {
            self.ins(I::I32Const(v as i32))
        }
    }

    fn ptr_add(&mut self) {
        self.ins(if self.config.memory64 {
            I::I64Add
        } else {
            I::I32Add
        })
    }

    /// Clear the bits above a narrow type's width.
    fn mask(&mut self, ty: Type) {
        if let Some(bits) = narrow_bits(ty) {
            self.ins(I::I32Const(((1u32 << bits) - 1) as i32));
            self.ins(I::I32And);
        }
    }

    /// Sign-extend a narrow value to its full `i32`.
    fn sext(&mut self, ty: Type) {
        match ty {
            types::I8 => self.ins(I::I32Extend8S),
            types::I16 => self.ins(I::I32Extend16S),
            _ => {}
        }
    }

    /// Convert an integer on the stack to an `i32` truth value.
    fn int_to_cond(&mut self, ty: Type) {
        if ty == types::I64 {
            self.ins(I::I64Const(0));
            self.ins(I::I64Ne);
        }
    }

    /// Push a value and convert it to the wasm address type.
    fn address(&mut self, v: Value) -> Result<()> {
        self.value(v)?;
        match (self.ty(v), self.config.memory64) {
            (types::I64, false) => self.ins(I::I32WrapI64),
            (types::I64, true) => {}
            (types::I32, true) => self.ins(I::I64ExtendI32U),
            (types::I32, false) => {}
            (ty, _) => return unsupported(format!("address of type {ty}")),
        }
        Ok(())
    }

    /// Push a shift amount, converted to the shifted type's width and
    /// masked for narrow types.
    fn shift_amount(&mut self, amt: Value, ty: Type) -> Result<()> {
        self.value(amt)?;
        match (is64(self.ty(amt)), is64(ty)) {
            (true, false) => self.ins(I::I32WrapI64),
            (false, true) => self.ins(I::I64ExtendI32U),
            _ => {}
        }
        if let Some(bits) = narrow_bits(ty) {
            self.ins(I::I32Const(bits as i32 - 1));
            self.ins(I::I32And);
        }
        Ok(())
    }

    fn memarg(&mut self, offset: u64, flags: cranelift_codegen::ir::MemFlags, size: u32) -> MemArg {
        let data = self.func.dfg.mem_flags[flags];
        MemArg {
            offset,
            align: if data.aligned() {
                size.trailing_zeros()
            } else {
                0
            },
            memory_index: self.config.memory_index,
        }
    }

    /// Push an address and return the static offset to use in the memarg.
    fn mem_address(
        &mut self,
        addr: Value,
        offset: i32,
        flags: cranelift_codegen::ir::MemFlags,
    ) -> Result<u64> {
        if self.func.dfg.mem_flags[flags].explicit_endianness() == Some(Endianness::Big) {
            return unsupported("big-endian memory access");
        }
        self.address(addr)?;
        if offset >= 0 {
            Ok(offset as u64)
        } else {
            self.ptr_const(offset as i64);
            self.ptr_add();
            Ok(0)
        }
    }

    pub fn epilogue(&mut self) {
        if let Some((fp, size)) = self.frame {
            self.ins(I::LocalGet(fp));
            self.ptr_const(size as i64);
            self.ptr_add();
            self.ins(I::GlobalSet(self.config.stack_pointer_global.unwrap()));
        }
    }

    // ---- Instructions -----------------------------------------------------

    /// Emit a root instruction and store or drop its results.
    pub fn root(&mut self, inst: Inst) -> Result<()> {
        self.inst(inst)?;
        for &r in self.func.dfg.inst_results(inst).iter().rev() {
            let l = self.locals.local_of[r];
            if l == NO_LOCAL {
                self.ins(I::Drop);
            } else {
                self.ins(I::LocalSet(l));
            }
        }
        Ok(())
    }

    /// Emit an instruction, leaving its results on the stack.
    pub fn inst(&mut self, inst: Inst) -> Result<()> {
        #[cfg(debug_assertions)]
        {
            let mut ops = Vec::new();
            crate::stackify::operands(self.func, self.info, inst, &mut ops);
            self.expect.push((ops.iter().map(|o| o.value).collect(), 0));
        }
        self.lower(inst)?;
        #[cfg(debug_assertions)]
        {
            let (exp, i) = self.expect.pop().unwrap();
            assert_eq!(exp.len(), i, "not all operands of {inst} were emitted");
        }
        Ok(())
    }

    fn lower(&mut self, inst: Inst) -> Result<()> {
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
                self.value(args[0])?;
                self.value(args[1])?;
                self.ins(match op {
                    Opcode::Iadd => w!(ty, I32Add, I64Add),
                    Opcode::Isub => w!(ty, I32Sub, I64Sub),
                    _ => w!(ty, I32Mul, I64Mul),
                });
                self.mask(ty);
            }
            (Opcode::Band | Opcode::Bor | Opcode::Bxor, _) if ty.is_int() => {
                self.value(args[0])?;
                self.value(args[1])?;
                self.ins(match op {
                    Opcode::Band => w!(ty, I32And, I64And),
                    Opcode::Bor => w!(ty, I32Or, I64Or),
                    _ => w!(ty, I32Xor, I64Xor),
                });
            }
            (Opcode::Bnot | Opcode::Ineg, _) if ty.is_int() => {
                self.value(args[0])?;
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
                self.value(args[0])?;
                if op == Opcode::Sshr {
                    self.sext(ty);
                }
                self.shift_amount(args[1], ty)?;
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
                self.value(args[0])?;
                if let Some(bits) = narrow_bits(ty) {
                    let x = self.scratch(ValType::I32);
                    let n = self.scratch(ValType::I32);
                    self.ins(I::LocalTee(x));
                    self.shift_amount(args[1], ty)?;
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
                    self.shift_amount(args[1], ty)?;
                    self.ins(if op == Opcode::Rotl {
                        w!(ty, I32Rotl, I64Rotl)
                    } else {
                        w!(ty, I32Rotr, I64Rotr)
                    });
                }
            }
            (Opcode::Udiv | Opcode::Urem, _) => {
                self.value(args[0])?;
                self.value(args[1])?;
                self.ins(if op == Opcode::Udiv {
                    w!(ty, I32DivU, I64DivU)
                } else {
                    w!(ty, I32RemU, I64RemU)
                });
            }
            (Opcode::Sdiv | Opcode::Srem, _) => {
                self.value(args[0])?;
                self.sext(ty);
                self.value(args[1])?;
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
                        self.value(args[0])?;
                        if signed {
                            self.sext(ty);
                        }
                        self.value(args[1])?;
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
                        self.value(args[0])?;
                        self.ins(ext.clone());
                        self.value(args[1])?;
                        self.ins(ext);
                        self.ins(I::I64Mul);
                        self.ins(I::I64Const(32));
                        self.ins(I::I64ShrU);
                        self.ins(I::I32WrapI64);
                    }
                    _ => return unsupported(format!("{op}.{ty}")),
                }
            }
            (Opcode::Iabs, _) => {
                self.value(args[0])?;
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
                self.value(args[0])?;
                if signed {
                    self.sext(ty);
                }
                self.ins(I::LocalTee(a));
                self.value(args[1])?;
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
                self.value(args[0])?;
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
                self.value(args[0])?;
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
                self.value(args[0])?;
                if signed {
                    self.sext(aty);
                }
                self.value(args[1])?;
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
                self.value(args[0])?;
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
                self.value(args[1])?;
                self.value(args[2])?;
                self.value(args[0])?;
                self.int_to_cond(aty);
                self.ins(I::Select);
            }
            (Opcode::Bitselect, _) if ty.is_int() => {
                let vt = wasm_type(ty)?;
                let c = self.scratch(vt);
                self.value(args[0])?;
                self.ins(I::LocalTee(c));
                self.value(args[1])?;
                self.ins(w!(ty, I32And, I64And));
                self.value(args[2])?;
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
                self.value(args[0])?;
                self.ins(I::LocalTee(a));
                self.value(args[1])?;
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

            // Conversions.
            (Opcode::Uextend, _) => {
                self.value(args[0])?;
                if is64(ty) && !is64(aty) {
                    self.ins(I::I64ExtendI32U);
                }
            }
            (Opcode::Sextend, _) => {
                self.value(args[0])?;
                self.sext(aty);
                if is64(ty) && !is64(aty) {
                    self.ins(I::I64ExtendI32S);
                }
                self.mask(ty);
            }
            (Opcode::Ireduce, _) => {
                self.value(args[0])?;
                if is64(aty) {
                    self.ins(I::I32WrapI64);
                }
                self.mask(ty);
            }
            (Opcode::Bitcast, _) if !ty.is_vector() => {
                self.value(args[0])?;
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
                self.value(args[0])?;
                self.ins(I::F64PromoteF32);
            }
            (Opcode::Fdemote, _) if ty == types::F32 && aty == types::F64 => {
                self.value(args[0])?;
                self.ins(I::F32DemoteF64);
            }
            (Opcode::FcvtFromSint | Opcode::FcvtFromUint, _) if !ty.is_vector() => {
                let signed = op == Opcode::FcvtFromSint;
                self.value(args[0])?;
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
                self.value(args[0])?;
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
                self.value(args[0])?;
                self.ins(match (is64(ty), aty == types::F64, signed) {
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
                self.value(args[0])?;
                self.value(args[1])?;
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
                self.value(args[0])?;
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
                self.fcmp(cond, aty, args[0], args[1])?;
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
                D::Load {
                    arg, flags, offset, ..
                },
            ) => {
                let off = self.mem_address(arg, offset.into(), flags)?;
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
                    args: [x, p],
                    flags,
                    offset,
                    ..
                },
            ) => {
                let off = self.mem_address(p, offset.into(), flags)?;
                self.value(x)?;
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

            (Opcode::GetStackPointer | Opcode::GetFramePointer, _) => {
                match (self.frame, self.config.stack_pointer_global) {
                    (Some((fp, _)), _) => self.ins(I::LocalGet(fp)),
                    (None, Some(sp)) => self.ins(I::GlobalGet(sp)),
                    (None, None) => return unsupported(format!("{op} without a stack pointer")),
                }
                self.ptr_to(ty)?;
            }

            // Calls.
            (Opcode::Call | Opcode::ReturnCall, D::Call { func_ref, .. }) => {
                for &a in args {
                    self.value(a)?;
                }
                let name = &dfg.ext_funcs[func_ref].name;
                let idx = self.env.func_index(Name::resolve(func, name))?;
                if op == Opcode::ReturnCall {
                    self.epilogue();
                    if self.config.tail_calls {
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
                for &a in &args[1..] {
                    self.value(a)?;
                }
                self.value(args[0])?;
                if is64(aty) {
                    self.ins(I::I32WrapI64);
                }
                let type_index = self.env.type_index(&dfg.signatures[sig_ref])?;
                let table_index = self.config.table_index;
                if op == Opcode::ReturnCallIndirect {
                    self.epilogue();
                    if self.config.tail_calls {
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
                for &a in args {
                    self.value(a)?;
                }
                self.epilogue();
                self.ins(I::Return);
            }

            // Traps.
            (Opcode::Trap, _) => self.ins(I::Unreachable),
            (Opcode::Trapz | Opcode::Trapnz, _) => {
                self.value(args[0])?;
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

    /// Convert a pointer-typed value on the stack to `ty`.
    fn ptr_to(&mut self, ty: Type) -> Result<()> {
        match (self.config.memory64, ty) {
            (false, types::I32) | (true, types::I64) => {}
            (false, types::I64) => self.ins(I::I64ExtendI32U),
            (true, types::I32) => self.ins(I::I32WrapI64),
            _ => return unsupported(format!("pointer of type {ty}")),
        }
        Ok(())
    }

    /// Trap if the `i32` on the stack is nonzero.
    fn trap_if(&mut self) {
        self.ins(I::If(BlockType::Empty));
        self.ins(I::Unreachable);
        self.ins(I::End);
    }

    fn fcmp(&mut self, cond: FloatCC, ty: Type, a: Value, b: Value) -> Result<()> {
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
            self.value(a)?;
            self.value(b)?;
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
                self.value(a)?;
                self.ins(I::LocalTee(sa));
                self.ins(I::LocalGet(sa));
                self.ins(cmp.clone());
                self.value(b)?;
                self.ins(I::LocalTee(sb));
                self.ins(I::LocalGet(sb));
                self.ins(cmp);
                self.ins(comb);
            }
            FloatCC::OrderedNotEqual | FloatCC::UnorderedOrEqual => {
                self.value(a)?;
                self.ins(I::LocalTee(sa));
                self.value(b)?;
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
