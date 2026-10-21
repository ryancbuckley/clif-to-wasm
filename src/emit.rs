//! Instruction, value, operand, local and stack-frame emission.
//!
//! [`Emitter`] encapsulates the low-level mechanics of producing a WebAssembly
//! function body:
//!
//! - **Function prologue and epilogue**: lays out stack slots in a shadow
//!   stack frame in linear memory (via `ModuleLayout::stack_pointer_global`)
//!   and masks narrow (`i8`/`i16`) entry parameters.
//! - **Operand stack cursor**: [`Emitter::inst`] loads the operands of an
//!   instruction in the exact order defined by [`crate::expr_trees::operands`],
//!   and lowering pulls them one by one with [`Emitter::operand`].
//! - **Scoped scratch locals**: temporary locals requested with
//!   [`Emitter::scratch`] or [`Emitter::scratches`] are tracked on a LIFO
//!   stack and automatically returned to a free pool when [`Emitter::inst`]
//!   (or an inner scratch scope) completes.
//! - **Narrow-integer representation**: `i8` and `i16` values are stored
//!   zero-extended in `i32`s. Operations that can set bits above the type's
//!   width call [`Emitter::mask`]; operations that depend on the sign bit call
//!   [`Emitter::sext`] on their inputs first.
//! - **Byte encoding**: encodes instructions into the function's byte stream,
//!   fusing adjacent `local.set x; local.get x` into `local.tee x` when
//!   `Optimizations::use_tee` is enabled.

use crate::analysis::{FuncInfo, is_remat};
use crate::expr_trees::{ExprTrees, Operand, operands};
use crate::locals::{Local, Locals};
use crate::{
    CompiledFunction, Config, Env, FuncIndex, GlobalIndex, Name, Reloc, RelocKind, Result,
    TableIndex, TypeIndex, unsupported, wasm_signature,
};
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, ArgumentPurpose, Endianness, FuncRef, Function, Inst,
    InstructionData as D, MemFlags, Opcode, SigRef, Signature, StackSlot, Type, Value, ValueDef,
    types,
};
use wasm_encoder::{BlockType, Encode, Instruction as I, MemArg, ValType};

/// Saved position on the emitter's active scratch-local stack, returned by
/// [`Emitter::scratch_mark`] and passed to [`Emitter::free_scratches`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScratchMark(usize);

pub(crate) struct Emitter<'a> {
    pub func: &'a Function,
    pub config: &'a Config,
    pub env: &'a mut dyn Env,
    pub info: &'a FuncInfo,
    pub trees: &'a ExprTrees,
    pub locals: Locals,
    /// Shadow-stack frame: `(frame_pointer_local, frame_size)`.
    frame: Option<(Local, u32)>,
    /// Frame offset of the area where calls copy by-value struct arguments.
    byval_base: u32,
    slot_offset: SecondaryMap<StackSlot, u32>,
    body: Body,
    /// Locals created during emission (scratch locals, frame pointer, etc.).
    extra: ExtraLocals,
    /// Operands of the instructions being lowered, outermost first, and a
    /// `(next, end)` cursor into it for each active instruction.
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
    /// Create an emitter for `func` and emit the function prologue (shadow
    /// stack allocation and narrow parameter masking).
    pub fn new(
        func: &'a Function,
        config: &'a Config,
        env: &'a mut dyn Env,
        info: &'a FuncInfo,
        trees: &'a ExprTrees,
        locals: Locals,
    ) -> Result<Self> {
        let base = locals.next_index();
        let mut e = Emitter {
            func,
            config,
            env,
            info,
            trees,
            locals,
            frame: None,
            byval_base: 0,
            slot_offset: SecondaryMap::new(),
            body: Body::new(config.opts.use_tee),
            extra: ExtraLocals::new(base),
            operand_stack: Vec::new(),
            cursors: Vec::new(),
            ops_tmp: Vec::new(),
        };
        e.emit_prologue()?;
        Ok(e)
    }

    fn emit_prologue(&mut self) -> Result<()> {
        let func = self.func;
        let config = self.config;
        if !func.dynamic_stack_slots.is_empty() {
            return unsupported("dynamic stack slots");
        }
        // Calls copy by-value struct arguments (`StructArgument`) into the
        // frame; one area serves all calls, sized for the largest.
        let mut byval_size: u32 = 0;
        for b in func.layout.blocks() {
            for inst in func.layout.block_insts(b) {
                if let Some(sig) = func.dfg.call_signature(inst) {
                    let size: u32 = func.dfg.signatures[sig]
                        .params
                        .iter()
                        .map(|p| match p.purpose {
                            ArgumentPurpose::StructArgument(n) => n.next_multiple_of(16),
                            _ => 0,
                        })
                        .sum();
                    byval_size = byval_size.max(size);
                }
            }
        }
        if !func.sized_stack_slots.is_empty() || byval_size > 0 {
            let Some(sp) = config.module.stack_pointer_global else {
                return unsupported("stack slots without ModuleLayout::stack_pointer_global");
            };
            let mut size: u32 = 0;
            for (slot, data) in func.sized_stack_slots.iter() {
                if data.align_shift > 4 {
                    return unsupported("stack slot alignment above 16");
                }
                let align = 1u32 << data.align_shift;
                let Some(aligned) = size.checked_next_multiple_of(align) else {
                    return unsupported("stack frame size overflow");
                };
                self.slot_offset[slot] = aligned;
                let Some(next) = aligned.checked_add(data.size) else {
                    return unsupported("stack frame size overflow");
                };
                size = next;
            }
            let Some(size) = size
                .checked_next_multiple_of(16)
                .and_then(|s| Some((s, s.checked_add(byval_size)?)))
            else {
                return unsupported("stack frame size overflow");
            };
            let (byval_base, size) = size;
            self.byval_base = byval_base;
            let fp = self.new_local(config.module.ptr_ty());
            self.frame = Some((fp, size));
            self.global_get(sp);
            self.ptr_const(size as i64);
            self.ins(if config.module.memory64 {
                I::I64Sub
            } else {
                I::I32Sub
            });
            self.local_tee(fp);
            self.global_set(sp);
        }

        // Callers may pass arbitrary bits in the upper bits of narrow parameters.
        let entry = self.info.rpo[0];
        for (i, &p) in func.dfg.block_params(entry).iter().enumerate() {
            let bits = match func.dfg.value_type(p) {
                types::I8 => 0xff,
                types::I16 => 0xffff,
                _ => continue,
            };
            if self.info.is_live(p) {
                let l = Local(i as u32);
                self.local_get(l);
                self.ins(I::I32Const(bits));
                self.ins(I::I32And);
                self.local_set(l);
            }
        }
        Ok(())
    }

    pub fn epilogue(&mut self) {
        if let Some((fp, size)) = self.frame {
            self.local_get(fp);
            self.ptr_const(size as i64);
            self.ptr_add();
            self.global_set(self.config.module.stack_pointer_global.unwrap());
        }
    }

    pub fn ins(&mut self, i: I<'_>) {
        self.body.ins(i);
    }

    pub fn local_get(&mut self, l: Local) {
        self.ins(I::LocalGet(l.0));
    }

    pub fn local_set(&mut self, l: Local) {
        self.ins(I::LocalSet(l.0));
    }

    pub fn local_tee(&mut self, l: Local) {
        self.ins(I::LocalTee(l.0));
    }

    pub fn global_get(&mut self, g: GlobalIndex) {
        if self.config.module.relocatable {
            self.body.op(&[0x23]);
            self.body.uleb(RelocKind::GlobalIndexLeb, g.0);
        } else {
            self.ins(I::GlobalGet(g.0));
        }
    }

    pub fn global_set(&mut self, g: GlobalIndex) {
        if self.config.module.relocatable {
            self.body.op(&[0x24]);
            self.body.uleb(RelocKind::GlobalIndexLeb, g.0);
        } else {
            self.ins(I::GlobalSet(g.0));
        }
    }

    /// `call`, or `return_call` if `tail`.
    pub(crate) fn call(&mut self, f: FuncIndex, tail: bool) {
        if self.config.module.relocatable {
            self.body.op(&[if tail { 0x12 } else { 0x10 }]);
            self.body.uleb(RelocKind::FunctionIndexLeb, f.0);
        } else if tail {
            self.ins(I::ReturnCall(f.0));
        } else {
            self.ins(I::Call(f.0));
        }
    }

    /// `call_indirect`, or `return_call_indirect` if `tail`.
    pub(crate) fn call_indirect(&mut self, ty: TypeIndex, table: TableIndex, tail: bool) {
        let (type_index, table_index) = (ty.0, table.0);
        if self.config.module.relocatable {
            self.body.op(&[if tail { 0x13 } else { 0x11 }]);
            self.body.uleb(RelocKind::TypeIndexLeb, type_index);
            self.body.uleb(RelocKind::TableNumberLeb, table_index);
        } else if tail {
            self.ins(I::ReturnCallIndirect {
                type_index,
                table_index,
            });
        } else {
            self.ins(I::CallIndirect {
                type_index,
                table_index,
            });
        }
    }

    /// A constant of type `ty` (`i32` or `i64`) that the linker fills in
    /// from symbol `index`, with a relocation of the 32-bit kind `k32` or
    /// the 64-bit kind `k64`.
    pub(crate) fn reloc_const(
        &mut self,
        ty: Type,
        k32: RelocKind,
        k64: RelocKind,
        index: u32,
        addend: i64,
    ) {
        if ty == types::I64 {
            self.body.op(&[0x42]);
            self.body.sleb64(k64, index, addend);
        } else {
            self.body.op(&[0x41]);
            self.body.sleb32(k32, index, addend);
        }
    }

    /// Complete the function body (`unreachable; end`), run-length encode all
    /// declared and extra locals, and return the [`CompiledFunction`].
    pub fn finish(mut self) -> Result<CompiledFunction> {
        let (params, results) = wasm_signature(&self.func.signature)?;
        self.ins(I::Unreachable);
        self.ins(I::End);
        let (body, relocs) = self.body.finish();
        let mut decls: Vec<(u32, ValType)> = Vec::new();
        for &ty in self.locals.decls.iter().chain(self.extra.types()) {
            match decls.last_mut() {
                Some((n, t)) if *t == ty => *n += 1,
                _ => decls.push((1, ty)),
            }
        }
        Ok(CompiledFunction {
            params,
            results,
            locals: decls,
            body,
            relocs,
        })
    }

    // ---- Locals & Scratch Pool ----------------------------------------------

    /// Allocate a fresh local that persists for the rest of the function.
    pub fn new_local(&mut self, ty: ValType) -> Local {
        self.extra.fresh(ty)
    }

    /// Allocate a temporary scratch local for the current instruction (or
    /// inner scratch scope). Automatically returned to the free pool when the
    /// instruction or scope finishes.
    pub(crate) fn scratch(&mut self, ty: ValType) -> Local {
        self.extra.scratch(ty)
    }

    /// Allocate `N` temporary scratch locals of type `ty`.
    pub(crate) fn scratches<const N: usize>(&mut self, ty: ValType) -> [Local; N] {
        std::array::from_fn(|_| self.extra.scratch(ty))
    }

    /// Record the current scratch stack depth for an inner helper that may be
    /// called multiple times within one instruction.
    pub(crate) fn scratch_mark(&self) -> ScratchMark {
        self.extra.mark()
    }

    /// Reclaim all scratch locals allocated since `mark`.
    pub(crate) fn free_scratches(&mut self, mark: ScratchMark) {
        self.extra.free_since(mark);
    }

    // ---- Values & Operands --------------------------------------------------

    pub(crate) fn ty(&self, v: Value) -> Type {
        self.func.dfg.value_type(v)
    }

    pub fn local(&self, v: Value) -> Local {
        let l = self.locals.local_of[v];
        debug_assert_ne!(l, Local::NONE, "{v} has no local");
        l
    }

    /// Push the value `v` onto the wasm operand stack.
    pub fn value(&mut self, v: Value) -> Result<()> {
        match self.func.dfg.value_def(v) {
            ValueDef::Result(d, _) if is_remat(self.func, d) => self.constant(d),
            ValueDef::Result(d, _) if self.trees.is_inlined(d) => self.inst(d),
            _ => {
                self.local_get(self.local(v));
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
            D::UnaryIeee16 { imm, .. } => self.ins(I::I32Const(imm.bits().into())),
            D::UnaryIeee32 { imm, .. } => {
                self.ins(I::F32Const(wasm_encoder::Ieee32::new(imm.bits())))
            }
            D::UnaryIeee64 { imm, .. } => {
                self.ins(I::F64Const(wasm_encoder::Ieee64::new(imm.bits())))
            }
            D::UnaryConst {
                constant_handle, ..
            } => {
                // A narrower vector fills the low bytes.
                let data = self.func.dfg.constants.get(constant_handle).as_slice();
                let mut bytes = [0u8; 16];
                bytes[..data.len()].copy_from_slice(data);
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

    /// Push the next operand, an address, converted to the wasm address type.
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

    pub(crate) fn memarg(&mut self, offset: u64, flags: MemFlags, size: u32) -> MemArg {
        let data = self.func.dfg.mem_flags[flags];
        MemArg {
            offset,
            align: if data.aligned() {
                size.trailing_zeros()
            } else {
                0
            },
            memory_index: self.config.module.memory_index.0,
        }
    }

    /// Does an access with these flags use big-endian byte order? Wasm
    /// memory is little-endian, so such accesses reverse their bytes.
    pub(crate) fn big_endian(&self, flags: MemFlags) -> bool {
        self.func.dfg.mem_flags[flags].explicit_endianness() == Some(Endianness::Big)
    }

    /// Push the next operand, an address, and return the static offset to
    /// use in the memarg.
    pub(crate) fn mem_address(&mut self, offset: i32) -> Result<u64> {
        self.address()?;
        if offset >= 0 {
            Ok(offset as u64)
        } else {
            self.ptr_const(offset as i64);
            self.ptr_add();
            Ok(0)
        }
    }

    /// Emit `stack_addr` for `stack_slot` + `offset` with result type `ty`.
    pub(crate) fn stack_addr(&mut self, stack_slot: StackSlot, offset: i32, ty: Type) -> Result<()> {
        let (fp, _) = self.frame.expect("frame for stack slot");
        self.local_get(fp);
        let off = self.slot_offset[stack_slot] as i64 + i64::from(offset);
        self.ptr_const(off);
        self.ptr_add();
        self.ptr_to(ty)
    }

    /// Emit `get_stack_pointer` or `get_frame_pointer` with result type `ty`.
    pub(crate) fn stack_pointer(&mut self, op: Opcode, ty: Type) -> Result<()> {
        match (self.frame, self.config.module.stack_pointer_global) {
            (Some((fp, _)), _) => self.local_get(fp),
            (None, Some(sp)) => self.global_get(sp),
            (None, None) => return unsupported(format!("{op} without a stack pointer")),
        }
        self.ptr_to(ty)
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

    /// Resolve a direct callee `FuncRef` to its wasm function index.
    pub(crate) fn callee_index(&mut self, func_ref: FuncRef) -> Result<FuncIndex> {
        let name = &self.func.dfg.ext_funcs[func_ref].name;
        self.env.func_index(Name::resolve(self.func, name))
    }

    /// Push the first `n` arguments of a call with signature `sig`, the way
    /// the callee expects them: narrow `sext` arguments sign-extended, and
    /// by-value struct arguments copied into this frame (passing the copy's
    /// address).
    pub(crate) fn call_args(&mut self, sig: &Signature, n: usize, tail: bool) -> Result<()> {
        let mut byval = self.byval_base;
        for i in 0..n {
            self.operand()?;
            let p = sig.params.get(i);
            if let Some(&AbiParam {
                purpose: ArgumentPurpose::StructArgument(size),
                value_type,
                ..
            }) = p
            {
                if tail {
                    return unsupported("tail call with a by-value struct argument");
                }
                let (fp, _) = self.frame.expect("frame for by-value arguments");
                // The address in the memory's address type.
                match (self.config.module.memory64, value_type) {
                    (false, types::I64) => self.ins(I::I32WrapI64),
                    (true, types::I32) => self.ins(I::I64ExtendI32U),
                    _ => {}
                }
                let src = self.scratch(self.config.module.ptr_ty());
                self.local_set(src);
                self.frame_addr(fp, byval);
                self.local_get(src);
                self.ptr_const(size as i64);
                let mem = self.config.module.memory_index.0;
                self.ins(I::MemoryCopy {
                    src_mem: mem,
                    dst_mem: mem,
                });
                self.frame_addr(fp, byval);
                self.ptr_to(value_type)?;
                byval += size.next_multiple_of(16);
            } else {
                self.extend_for_abi(p);
            }
        }
        Ok(())
    }

    /// Sign-extend the narrow value on the stack if `p` asks for it. Values
    /// are kept zero-extended, which already satisfies `uext`.
    pub(crate) fn extend_for_abi(&mut self, p: Option<&AbiParam>) {
        if let Some(p) = p
            && p.extension == ArgumentExtension::Sext
        {
            self.sext(p.value_type);
        }
    }

    /// Restore the narrow-value invariant on a call's results: callees, LLVM
    /// ones in particular, may leave garbage in the upper bits, and `sext`
    /// returns arrive sign-extended.
    pub(crate) fn call_result(&mut self, inst: Inst) -> Result<()> {
        let dfg = &self.func.dfg;
        let results = dfg.inst_results(inst);
        if let &[r] = results {
            self.mask(dfg.value_type(r));
            return Ok(());
        }
        if !results
            .iter()
            .any(|&r| narrow_bits(dfg.value_type(r)).is_some())
        {
            return Ok(());
        }
        // Only the last result is on top: set them all aside, then push them
        // back masked.
        let tys: Vec<Type> = results.iter().map(|&r| dfg.value_type(r)).collect();
        let mut held = Vec::with_capacity(tys.len());
        for &ty in tys.iter().rev() {
            let l = self.scratch(crate::wasm_type(ty)?);
            self.local_set(l);
            held.push(l);
        }
        for (&ty, &l) in tys.iter().zip(held.iter().rev()) {
            self.local_get(l);
            self.mask(ty);
        }
        Ok(())
    }

    fn frame_addr(&mut self, fp: Local, offset: u32) {
        self.local_get(fp);
        self.ptr_const(offset as i64);
        self.ptr_add();
    }

    /// The type and table indices for an indirect call with signature `sig`.
    pub(crate) fn indirect_call(&mut self, sig: SigRef) -> Result<(TypeIndex, TableIndex)> {
        let Some(table_index) = self.config.module.table_index else {
            return unsupported("call_indirect without ModuleLayout::table_index");
        };
        let type_index = self.env.type_index(&self.func.dfg.signatures[sig])?;
        Ok((type_index, table_index))
    }

    // ---- Instructions -------------------------------------------------------

    /// Emit a root instruction and store or drop its results.
    pub fn root(&mut self, inst: Inst) -> Result<()> {
        self.inst(inst)?;
        for &r in self.func.dfg.inst_results(inst).iter().rev() {
            let l = self.locals.local_of[r];
            if l == Local::NONE {
                self.ins(I::Drop);
            } else {
                self.local_set(l);
            }
        }
        Ok(())
    }

    /// Emit an instruction, leaving its results on the stack.
    ///
    /// Lowering takes the operands one by one with [`Self::operand`], so they
    /// are pushed in exactly the order tree building assumed. Scratch locals
    /// allocated during lowering are automatically reclaimed on exit.
    pub fn inst(&mut self, inst: Inst) -> Result<()> {
        let start = self.operand_stack.len();
        let mut ops = std::mem::take(&mut self.ops_tmp);
        operands(self.func, self.info, inst, &mut ops);
        self.operand_stack.extend(ops.iter().map(|o| o.value));
        self.ops_tmp = ops;
        self.cursors.push((start, self.operand_stack.len()));
        let scratch_mark = self.extra.mark();
        let res = self.lower(inst);
        self.extra.free_since(scratch_mark);
        res?;
        let (next, end) = self.cursors.pop().unwrap();
        assert_eq!(next, end, "lowering of {inst} skipped operands");
        self.operand_stack.truncate(start);
        Ok(())
    }
}

// ---- Byte encoding & Extra locals -------------------------------------------

/// The encoded instructions of one function body, with a peephole that fuses
/// `local.set x; local.get x` into `local.tee x`.
struct Body {
    bytes: Vec<u8>,
    pending_set: Option<u32>,
    use_tee: bool,
    relocs: Vec<Reloc>,
}

impl Body {
    fn new(use_tee: bool) -> Self {
        Body {
            bytes: Vec::new(),
            pending_set: None,
            use_tee,
            relocs: Vec::new(),
        }
    }

    fn flush(&mut self) {
        if let Some(l) = self.pending_set.take() {
            I::LocalSet(l).encode(&mut self.bytes);
        }
    }

    /// Raw opcode bytes, for instructions whose immediates are relocated.
    fn op(&mut self, opcode: &[u8]) {
        self.flush();
        self.bytes.extend_from_slice(opcode);
    }

    fn reloc(&mut self, kind: RelocKind, index: u32, addend: i64) {
        self.relocs.push(Reloc {
            offset: self.bytes.len() as u32,
            kind,
            index,
            addend,
        });
    }

    /// A relocated index: a 5-byte LEB holding `index` as a placeholder.
    fn uleb(&mut self, kind: RelocKind, index: u32) {
        self.reloc(kind, index, 0);
        write_padded_uleb(&mut self.bytes, index);
    }

    fn sleb32(&mut self, kind: RelocKind, index: u32, addend: i64) {
        self.reloc(kind, index, addend);
        write_padded_sleb(&mut self.bytes, addend as i32 as i64, 5);
    }

    fn sleb64(&mut self, kind: RelocKind, index: u32, addend: i64) {
        self.reloc(kind, index, addend);
        write_padded_sleb(&mut self.bytes, addend, 10);
    }

    fn ins(&mut self, i: I<'_>) {
        if let Some(l) = self.pending_set.take() {
            if self.use_tee && matches!(i, I::LocalGet(g) if g == l) {
                I::LocalTee(l).encode(&mut self.bytes);
                return;
            }
            I::LocalSet(l).encode(&mut self.bytes);
        }
        if let I::LocalSet(l) = i {
            self.pending_set = Some(l);
            return;
        }
        i.encode(&mut self.bytes);
    }

    fn finish(mut self) -> (Vec<u8>, Vec<Reloc>) {
        self.flush();
        (self.bytes, self.relocs)
    }
}

/// Write `v` as an unsigned LEB padded to 5 bytes.
pub(crate) fn write_padded_uleb(out: &mut Vec<u8>, v: u32) {
    let mut v = v;
    for i in 0..5 {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        out.push(if i < 4 { byte | 0x80 } else { byte });
    }
}

/// Write `v` as a signed LEB padded to `len` bytes (5 for 32-bit fields, 10
/// for 64-bit ones).
pub(crate) fn write_padded_sleb(out: &mut Vec<u8>, v: i64, len: usize) {
    let mut v = v;
    for i in 0..len {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        out.push(if i + 1 < len { byte | 0x80 } else { byte });
    }
}

/// Locals created during emission, after the ones assigned to SSA values:
/// scoped scratch locals used inside instruction lowerings, and function-wide
/// locals such as the frame pointer or dispatch label.
struct ExtraLocals {
    base: u32,
    types: Vec<ValType>,
    free_scratch: Vec<(ValType, Local)>,
    active_scratch: Vec<(ValType, Local)>,
}

impl ExtraLocals {
    fn new(base: u32) -> Self {
        ExtraLocals {
            base,
            types: Vec::new(),
            free_scratch: Vec::new(),
            active_scratch: Vec::new(),
        }
    }

    fn fresh(&mut self, ty: ValType) -> Local {
        self.types.push(ty);
        Local(self.base + self.types.len() as u32 - 1)
    }

    fn scratch(&mut self, ty: ValType) -> Local {
        let l = match self.free_scratch.iter().position(|&(t, _)| t == ty) {
            Some(i) => self.free_scratch.swap_remove(i).1,
            None => self.fresh(ty),
        };
        self.active_scratch.push((ty, l));
        l
    }

    fn mark(&self) -> ScratchMark {
        ScratchMark(self.active_scratch.len())
    }

    fn free_since(&mut self, mark: ScratchMark) {
        self.free_scratch
            .extend(self.active_scratch.drain(mark.0..));
    }

    fn types(&self) -> &[ValType] {
        &self.types
    }
}
