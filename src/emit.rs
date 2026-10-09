//! Emission state shared by structuring and lowering, and value emission.
//!
//! `i8` and `i16` values are represented as zero-extended `i32`s. Operations
//! that can set bits above the type's width mask their result; operations
//! that depend on the sign bit sign-extend their inputs first.

use crate::analysis::{FuncInfo, is_remat};
use crate::encode::{Body, ExtraLocals};
use crate::expr_trees::{ExprTrees, Operand, operands};
use crate::locals::{Locals, NO_LOCAL};
use crate::{Config, Env, Result, unsupported};
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{
    Endianness, Function, Inst, InstructionData as D, StackSlot, Type, Value, ValueDef, types,
};
use wasm_encoder::{BlockType, Instruction as I, MemArg, ValType};

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
    pub trees: &'a ExprTrees,
    pub locals: Locals,
    pub labels: Vec<Label>,
    pub tasks: Vec<Task>,
    /// Shadow-stack frame: (frame pointer local, frame size).
    pub frame: Option<(u32, u32)>,
    pub slot_offset: SecondaryMap<StackSlot, u32>,
    /// The label local, when emitting a dispatch loop.
    pub dispatch_label: Option<u32>,
    body: Body,
    /// Locals created during emission (scratch, frame pointer, label).
    extra: ExtraLocals,
    /// Operands of the instructions being lowered, outermost first, and a
    /// (next, end) cursor into it for each of those instructions.
    operand_stack: Vec<Value>,
    cursors: Vec<(usize, usize)>,
    ops_tmp: Vec<Operand>,
}

pub(crate) fn is64(ty: Type) -> bool {
    ty == types::I64 || ty == types::F64
}

pub(crate) fn narrow_bits(ty: Type) -> Option<u32> {
    match ty {
        types::I8 => Some(8),
        types::I16 => Some(16),
        _ => None,
    }
}

/// Pick the 32- or 64-bit form of an instruction for type `$ty`.
macro_rules! w {
    ($ty:expr, $a:ident, $b:ident) => {
        if $crate::emit::is64($ty) {
            I::$b
        } else {
            I::$a
        }
    };
}
pub(crate) use w;

impl<'a> Emitter<'a> {
    pub fn new(
        func: &'a Function,
        config: &'a Config,
        env: &'a mut dyn Env,
        info: &'a FuncInfo,
        trees: &'a ExprTrees,
        locals: Locals,
    ) -> Self {
        let base = locals.next_index();
        Emitter {
            func,
            config,
            env,
            info,
            trees,
            locals,
            labels: Vec::new(),
            tasks: Vec::new(),
            frame: None,
            slot_offset: SecondaryMap::new(),
            dispatch_label: None,
            body: Body::new(config.opts.use_tee),
            extra: ExtraLocals::new(base),
            operand_stack: Vec::new(),
            cursors: Vec::new(),
            ops_tmp: Vec::new(),
        }
    }

    pub fn ins(&mut self, i: I<'_>) {
        self.body.ins(i);
    }

    pub fn finish(self) -> (Vec<u8>, Locals, Vec<ValType>) {
        let extra = self.extra.types().to_vec();
        (self.body.finish(), self.locals, extra)
    }

    pub fn new_local(&mut self, ty: ValType) -> u32 {
        self.extra.fresh(ty)
    }

    pub(crate) fn scratch(&mut self, ty: ValType) -> u32 {
        self.extra.scratch(ty)
    }

    pub(crate) fn release(&mut self, ty: ValType, l: u32) {
        self.extra.release(ty, l);
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

    pub(crate) fn ty(&self, v: Value) -> Type {
        self.func.dfg.value_type(v)
    }

    pub fn local(&self, v: Value) -> u32 {
        let l = self.locals.local_of[v];
        debug_assert_ne!(l, NO_LOCAL, "{v} has no local");
        l
    }

    /// Push the value `v`.
    pub fn value(&mut self, v: Value) -> Result<()> {
        match self.func.dfg.value_def(v) {
            ValueDef::Result(d, _) if is_remat(self.func, d) => self.constant(d),
            ValueDef::Result(d, _) if self.trees.inlined[d] => self.inst(d),
            _ => {
                self.ins(I::LocalGet(self.local(v)));
                Ok(())
            }
        }
    }

    /// Push the next operand of the instruction being lowered, in the order
    /// given by [`operands`], and return it.
    pub(crate) fn operand(&mut self) -> Result<Value> {
        let cursor = self.cursors.last_mut().expect("operand() outside inst()");
        assert!(cursor.0 < cursor.1, "too many operands requested");
        let v = self.operand_stack[cursor.0];
        cursor.0 += 1;
        self.value(v)?;
        Ok(v)
    }

    pub(crate) fn constant(&mut self, inst: Inst) -> Result<()> {
        let ty = self.ty(self.func.dfg.first_result(inst));
        match self.func.dfg.insts[inst] {
            D::UnaryImm { imm, .. } => self.iconst(ty, imm.bits()),
            D::UnaryIeee32 { imm, .. } => {
                self.ins(I::F32Const(wasm_encoder::Ieee32::new(imm.bits())))
            }
            D::UnaryIeee64 { imm, .. } => {
                self.ins(I::F64Const(wasm_encoder::Ieee64::new(imm.bits())))
            }
            D::UnaryConst {
                constant_handle, ..
            } => {
                let bytes = self.func.dfg.constants.get(constant_handle).as_slice();
                let bytes: [u8; 16] = bytes.try_into().expect("128-bit vector constant");
                self.ins(I::V128Const(i128::from_le_bytes(bytes)))
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub(crate) fn iconst(&mut self, ty: Type, v: i64) {
        match ty {
            types::I64 => self.ins(I::I64Const(v)),
            types::I8 => self.ins(I::I32Const((v as u8) as i32)),
            types::I16 => self.ins(I::I32Const((v as u16) as i32)),
            _ => self.ins(I::I32Const(v as i32)),
        }
    }

    pub(crate) fn ptr_const(&mut self, v: i64) {
        if self.config.module.memory64 {
            self.ins(I::I64Const(v))
        } else {
            self.ins(I::I32Const(v as i32))
        }
    }

    pub(crate) fn ptr_add(&mut self) {
        self.ins(if self.config.module.memory64 {
            I::I64Add
        } else {
            I::I32Add
        })
    }

    /// Clear the bits above a narrow type's width.
    pub(crate) fn mask(&mut self, ty: Type) {
        if let Some(bits) = narrow_bits(ty) {
            self.ins(I::I32Const(((1u32 << bits) - 1) as i32));
            self.ins(I::I32And);
        }
    }

    /// Sign-extend a narrow value to its full `i32`.
    pub(crate) fn sext(&mut self, ty: Type) {
        match ty {
            types::I8 => self.ins(I::I32Extend8S),
            types::I16 => self.ins(I::I32Extend16S),
            _ => {}
        }
    }

    /// Convert an integer on the stack to an `i32` truth value.
    pub(crate) fn int_to_cond(&mut self, ty: Type) {
        if ty == types::I64 {
            self.ins(I::I64Const(0));
            self.ins(I::I64Ne);
        }
    }

    /// Push the next operand, an address, converted to the wasm address
    /// type.
    pub(crate) fn address(&mut self) -> Result<()> {
        let v = self.operand()?;
        match (self.ty(v), self.config.module.memory64) {
            (types::I64, false) => self.ins(I::I32WrapI64),
            (types::I64, true) => {}
            (types::I32, true) => self.ins(I::I64ExtendI32U),
            (types::I32, false) => {}
            (ty, _) => return unsupported(format!("address of type {ty}")),
        }
        Ok(())
    }

    /// Push the next operand, a shift amount, converted to the shifted
    /// type's width and masked for narrow types.
    pub(crate) fn shift_amount(&mut self, ty: Type) -> Result<()> {
        let amt = self.operand()?;
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

    pub(crate) fn memarg(
        &mut self,
        offset: u64,
        flags: cranelift_codegen::ir::MemFlags,
        size: u32,
    ) -> MemArg {
        let data = self.func.dfg.mem_flags[flags];
        MemArg {
            offset,
            align: if data.aligned() {
                size.trailing_zeros()
            } else {
                0
            },
            memory_index: self.config.module.memory_index,
        }
    }

    /// Push the next operand, an address, and return the static offset to
    /// use in the memarg.
    pub(crate) fn mem_address(
        &mut self,
        offset: i32,
        flags: cranelift_codegen::ir::MemFlags,
    ) -> Result<u64> {
        if self.func.dfg.mem_flags[flags].explicit_endianness() == Some(Endianness::Big) {
            return unsupported("big-endian memory access");
        }
        self.address()?;
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
            self.ins(I::GlobalSet(
                self.config.module.stack_pointer_global.unwrap(),
            ));
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
    ///
    /// Lowering takes the operands one by one with [`Self::operand`], so they
    /// are pushed in exactly the order tree building assumed.
    pub fn inst(&mut self, inst: Inst) -> Result<()> {
        let start = self.operand_stack.len();
        let mut ops = std::mem::take(&mut self.ops_tmp);
        operands(self.func, self.info, inst, &mut ops);
        self.operand_stack.extend(ops.iter().map(|o| o.value));
        self.ops_tmp = ops;
        self.cursors.push((start, self.operand_stack.len()));
        self.lower(inst)?;
        let (next, end) = self.cursors.pop().unwrap();
        assert_eq!(next, end, "lowering of {inst} skipped operands");
        self.operand_stack.truncate(start);
        Ok(())
    }

    /// Convert a pointer-typed value on the stack to `ty`.
    pub(crate) fn ptr_to(&mut self, ty: Type) -> Result<()> {
        match (self.config.module.memory64, ty) {
            (false, types::I32) | (true, types::I64) => {}
            (false, types::I64) => self.ins(I::I64ExtendI32U),
            (true, types::I32) => self.ins(I::I32WrapI64),
            _ => return unsupported(format!("pointer of type {ty}")),
        }
        Ok(())
    }

    /// Trap if the `i32` on the stack is nonzero.
    pub(crate) fn trap_if(&mut self) {
        self.ins(I::If(BlockType::Empty));
        self.ins(I::Unreachable);
        self.ins(I::End);
    }
}
