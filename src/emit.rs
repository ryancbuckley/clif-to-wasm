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

use crate::analysis::{self, FuncInfo};
use crate::expr_trees::{self, ExprTrees, Operand};
use crate::locals::{Local, Locals};
use crate::{
    AddressType, CompiledFunction, Config, Env, FuncIndex, GlobalIndex, Name, Reloc, RelocKind,
    Result, TableIndex, TypeIndex, error, wasm_types,
};
use alloc::format;
use alloc::vec::Vec;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, ArgumentPurpose, Endianness, FuncRef, Function, Inst,
    InstructionData as D, MemFlags, Opcode, SigRef, Signature, StackSlot, Type, Value, ValueDef,
    types,
};
use wasm_encoder::{BlockType, Encode, Instruction as I, MemArg, MemArgTarget, ValType};

/// Saved position on the emitter's active scratch-local stack, returned by
/// [`Emitter::scratch_mark`] and passed to [`Emitter::free_scratches`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScratchMark(usize);

/// Whether a call returns to its caller or replaces the caller's frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CallKind {
    /// `call` or `call_indirect`.
    Normal,
    /// `return_call` or `return_call_indirect`.
    Tail,
}

/// A by-value struct argument still to be copied into the frame, from
/// [`Emitter::call_args`].
pub(crate) struct ByvalCopy {
    /// Offset of the copy in the frame.
    dst: u32,
    /// Holds the struct's address.
    src: Local,
    size: u32,
}

/// A function's frame on the shadow stack.
#[derive(Clone, Copy)]
struct Frame {
    /// Holds the frame's address, the stack pointer after allocation.
    fp: Local,
    size: u32,
    /// The stack pointer global.
    sp: GlobalIndex,
}

pub(crate) struct Emitter<'ctx> {
    pub func: &'ctx Function,
    pub config: &'ctx Config,
    pub env: &'ctx mut dyn Env,
    pub info: &'ctx FuncInfo,
    pub trees: &'ctx ExprTrees,
    pub locals: Locals,
    /// Shadow-stack frame, if the function needs one.
    frame: Option<Frame>,
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

impl<'ctx> Emitter<'ctx> {
    /// Create an emitter for `func` and emit the function prologue (shadow
    /// stack allocation and narrow parameter masking).
    pub fn new(
        func: &'ctx Function,
        config: &'ctx Config,
        env: &'ctx mut dyn Env,
        info: &'ctx FuncInfo,
        trees: &'ctx ExprTrees,
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
            return error::unsupported("dynamic stack slots");
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
                return error::unsupported(
                    "stack slots without ModuleLayout::stack_pointer_global",
                );
            };
            let mut size: u32 = 0;
            for (slot, data) in func.sized_stack_slots.iter() {
                if data.align_shift > 4 {
                    return error::unsupported("stack slot alignment above 16");
                }
                let align = 1u32 << data.align_shift;
                let Some(aligned) = size.checked_next_multiple_of(align) else {
                    return error::unsupported("stack frame size overflow");
                };
                self.slot_offset[slot] = aligned;
                let Some(next) = aligned.checked_add(data.size) else {
                    return error::unsupported("stack frame size overflow");
                };
                size = next;
            }
            let Some(size) = size
                .checked_next_multiple_of(16)
                .and_then(|s| Some((s, s.checked_add(byval_size)?)))
            else {
                return error::unsupported("stack frame size overflow");
            };
            let (byval_base, size) = size;
            self.byval_base = byval_base;
            let fp = self.new_local(config.module.ptr_ty());
            self.frame = Some(Frame { fp, size, sp });
            self.global_get(sp);
            self.ptr_const(size as i64);
            self.ins(match config.module.address_type {
                AddressType::I32 => I::I32Sub,
                AddressType::I64 => I::I64Sub,
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
        if let Some(Frame { fp, size, sp }) = self.frame {
            self.local_get(fp);
            self.ptr_const(size as i64);
            self.ptr_add();
            self.global_set(sp);
        }
    }

    /// The frame pointer. The prologue allocates a frame whenever the
    /// function has stack slots or calls with by-value struct arguments.
    fn fp(&self) -> Local {
        self.frame.expect("function has a frame").fp
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

    /// `call`, or `return_call` for a tail call.
    pub(crate) fn call(&mut self, f: FuncIndex, kind: CallKind) {
        let tail = kind == CallKind::Tail;
        if self.config.module.relocatable {
            self.body.op(&[if tail { 0x12 } else { 0x10 }]);
            self.body.uleb(RelocKind::FunctionIndexLeb, f.0);
        } else if tail {
            self.ins(I::ReturnCall(f.0));
        } else {
            self.ins(I::Call(f.0));
        }
    }

    /// `call_indirect`, or `return_call_indirect` for a tail call.
    pub(crate) fn call_indirect(&mut self, ty: TypeIndex, table: TableIndex, kind: CallKind) {
        let tail = kind == CallKind::Tail;
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

    /// Push a `ty` (`i32` or `i64`) constant relocated against symbol `index`
    /// with relocation kind `k32` or `k64`.
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
        let (params, results) =
            wasm_types::wasm_signature(&self.func.signature, &self.config.module)?;
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
        core::array::from_fn(|_| self.extra.scratch(ty))
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
            ValueDef::Result(d, _) if analysis::is_remat(self.func, d) => self.constant(d),
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
        match self.config.module.address_type {
            AddressType::I32 => self.ins(I::I32Const(v as i32)),
            AddressType::I64 => self.ins(I::I64Const(v)),
        }
    }

    pub(crate) fn ptr_add(&mut self) {
        self.ins(match self.config.module.address_type {
            AddressType::I32 => I::I32Add,
            AddressType::I64 => I::I64Add,
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
        match (self.ty(v), self.config.module.address_type) {
            (types::I64, AddressType::I32) => self.ins(I::I32WrapI64),
            (types::I64, AddressType::I64) => {}
            (types::I32, AddressType::I64) => self.ins(I::I64ExtendI32U),
            (types::I32, AddressType::I32) => {}
            (ty, _) => return error::unsupported(format!("address of type {ty}")),
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
            target: MemArgTarget::Memory(self.config.module.memory_index.0),
            ordering: None,
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
    pub(crate) fn stack_addr(
        &mut self,
        stack_slot: StackSlot,
        offset: i32,
        ty: Type,
    ) -> Result<()> {
        self.local_get(self.fp());
        let off = self.slot_offset[stack_slot] as i64 + i64::from(offset);
        self.ptr_const(off);
        self.ptr_add();
        self.ptr_to(ty)
    }

    /// Emit `get_stack_pointer` or `get_frame_pointer` with result type `ty`.
    pub(crate) fn stack_pointer(&mut self, op: Opcode, ty: Type) -> Result<()> {
        match (self.frame, self.config.module.stack_pointer_global) {
            (Some(frame), _) => self.local_get(frame.fp),
            (None, Some(sp)) => self.global_get(sp),
            (None, None) => return error::unsupported(format!("{op} without a stack pointer")),
        }
        self.ptr_to(ty)
    }

    /// Convert a pointer-typed value on the stack to `ty`.
    pub(crate) fn ptr_to(&mut self, ty: Type) -> Result<()> {
        match (self.config.module.address_type, ty) {
            (AddressType::I32, types::I32) | (AddressType::I64, types::I64) => {}
            (AddressType::I32, types::I64) => self.ins(I::I64ExtendI32U),
            (AddressType::I64, types::I32) => self.ins(I::I32WrapI64),
            _ => return error::unsupported(format!("pointer of type {ty}")),
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

    /// Push the first `n` call arguments for `sig`: sign-extending narrow `sext`
    /// arguments and staging by-value struct arguments (`StructArgument`) for
    /// copying into the caller's frame.
    ///
    /// Actual `memory.copy` instructions are emitted by [`Self::byval_copies`]
    /// after all call operands are evaluated, in case a later operand mutates
    /// a passed struct or performs a nested call using the same frame area.
    pub(crate) fn call_args(
        &mut self,
        sig: &Signature,
        n: usize,
        tail: bool,
    ) -> Result<Vec<ByvalCopy>> {
        let mut byval = self.byval_base;
        let mut copies = Vec::new();
        for i in 0..n {
            let p = sig.params.get(i);
            if let Some(&AbiParam {
                purpose: ArgumentPurpose::StructArgument(size),
                value_type,
                ..
            }) = p
            {
                if tail {
                    return error::unsupported("tail call with a by-value struct argument");
                }
                // The struct's address, in the memory's address type.
                self.address()?;
                let src = self.scratch(self.config.module.ptr_ty());
                self.local_set(src);
                copies.push(ByvalCopy {
                    dst: byval,
                    src,
                    size,
                });
                self.frame_addr(byval);
                self.ptr_to(value_type)?;
                byval += size.next_multiple_of(16);
            } else {
                self.operand()?;
                self.extend_for_abi(p);
            }
        }
        Ok(copies)
    }

    /// Copy by-value struct arguments into the frame, after [`Self::call_args`]
    /// and any other operand of the call. `memory.copy` leaves the arguments
    /// on the stack alone.
    pub(crate) fn byval_copies(&mut self, copies: Vec<ByvalCopy>) {
        let mem = self.config.module.memory_index.0;
        for c in copies {
            self.frame_addr(c.dst);
            self.local_get(c.src);
            self.ptr_const(c.size as i64);
            self.ins(I::MemoryCopy {
                src_mem: mem,
                dst_mem: mem,
            });
        }
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

    /// Mask narrow call results to maintain the zero-extension invariant
    /// (callees may leave upper bits undefined or sign-extended).
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
            let l = self.scratch(wasm_types::wasm_type(ty)?);
            self.local_set(l);
            held.push(l);
        }
        let in_order = tys.iter().zip(held.iter().rev());
        for (&ty, &l) in in_order {
            self.local_get(l);
            self.mask(ty);
        }
        Ok(())
    }

    fn frame_addr(&mut self, offset: u32) {
        self.local_get(self.fp());
        self.ptr_const(offset as i64);
        self.ptr_add();
    }

    /// The type and table indices for an indirect call with signature `sig`.
    pub(crate) fn indirect_call(&mut self, sig: SigRef) -> Result<(TypeIndex, TableIndex)> {
        let Some(table_index) = self.config.module.table_index else {
            return error::unsupported("call_indirect without ModuleLayout::table_index");
        };
        let (params, results) =
            wasm_types::wasm_signature(&self.func.dfg.signatures[sig], &self.config.module)?;
        let type_index = self.env.type_index(&params, &results)?;
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
        let mut ops = core::mem::take(&mut self.ops_tmp);
        expr_trees::operands(self.func, self.info, inst, &mut ops);
        self.operand_stack.extend(ops.iter().map(|o| o.value));
        self.ops_tmp = ops;
        self.cursors.push((start, self.operand_stack.len()));
        let scratch_mark = self.extra.mark();
        let res = self.lower(inst);
        self.extra.free_since(scratch_mark);
        res?;
        let (next, end) = self.cursors.pop().expect("cursor pushed above");
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
        self.bytes.extend_from_slice(&padded_leb::<5>(index as i64));
    }

    fn sleb32(&mut self, kind: RelocKind, index: u32, addend: i64) {
        self.reloc(kind, index, addend);
        self.bytes
            .extend_from_slice(&padded_leb::<5>(addend as i32 as i64));
    }

    fn sleb64(&mut self, kind: RelocKind, index: u32, addend: i64) {
        self.reloc(kind, index, addend);
        self.bytes.extend_from_slice(&padded_leb::<10>(addend));
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

/// Returns `v` as a signed LEB padded to `N` bytes. With `N` = 5, a `u32`
/// encodes the same as an unsigned LEB: it is non-negative and fits in 35 bits.
pub(crate) fn padded_leb<const N: usize>(mut v: i64) -> [u8; N] {
    let mut out = [0; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = (v & 0x7f) as u8 | if i + 1 < N { 0x80 } else { 0 };
        v >>= 7;
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use hashbrown::HashSet;
    use proptest::prelude::*;
    use wasmparser::{BinaryReader, Operator, OperatorsReader};

    struct NoEnv;
    impl Env for NoEnv {
        fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex> {
            Err(crate::Error::Env(name.to_string()))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Op {
        LocalGet(u32),
        LocalSet(u32),
        LocalTee(u32),
        I32Const(i32),
        I32Add,
        Drop,
    }

    impl Op {
        fn to_insn(self) -> I<'static> {
            match self {
                Op::LocalGet(x) => I::LocalGet(x),
                Op::LocalSet(x) => I::LocalSet(x),
                Op::LocalTee(x) => I::LocalTee(x),
                Op::I32Const(c) => I::I32Const(c),
                Op::I32Add => I::I32Add,
                Op::Drop => I::Drop,
            }
        }
    }

    fn decode_ops(bytes: &[u8]) -> Vec<Op> {
        let reader = OperatorsReader::new(BinaryReader::new(bytes, 0));
        let mut out = Vec::new();
        for op in reader {
            let op = match op.unwrap() {
                Operator::LocalGet { local_index } => Op::LocalGet(local_index),
                Operator::LocalSet { local_index } => Op::LocalSet(local_index),
                Operator::LocalTee { local_index } => Op::LocalTee(local_index),
                Operator::I32Const { value } => Op::I32Const(value),
                Operator::I32Add => Op::I32Add,
                Operator::Drop => Op::Drop,
                other => panic!("unexpected operator: {other:?}"),
            };
            out.push(op);
        }
        out
    }

    /// Reference model for `Body` peephole fusion.
    fn model_peephole(ops: &[Op], use_tee: bool) -> Vec<Op> {
        if !use_tee {
            return ops.to_vec();
        }
        let mut out = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            if let (Op::LocalSet(s), Some(&Op::LocalGet(g))) = (ops[i], ops.get(i + 1))
                && s == g
            {
                out.push(Op::LocalTee(s));
                i += 2;
            } else {
                out.push(ops[i]);
                i += 1;
            }
        }
        out
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            4 => (0u32..4).prop_map(Op::LocalGet),
            4 => (0u32..4).prop_map(Op::LocalSet),
            2 => (0u32..4).prop_map(Op::LocalTee),
            2 => any::<i32>().prop_map(Op::I32Const),
            1 => Just(Op::I32Add),
            1 => Just(Op::Drop),
        ]
    }

    #[derive(Clone, Copy, Debug)]
    enum PoolAction {
        Fresh(ValType),
        Scratch(ValType),
        Mark,
        FreeLastMark,
    }

    fn valtype_strategy() -> impl Strategy<Value = ValType> {
        prop_oneof![
            Just(ValType::I32),
            Just(ValType::I64),
            Just(ValType::F32),
            Just(ValType::F64),
            Just(ValType::V128),
        ]
    }

    proptest! {
        #[test]
        fn body_peephole_matches_model(
            ops in prop::collection::vec(op_strategy(), 0..64),
            use_tee in any::<bool>(),
        ) {
            let mut body = Body::new(use_tee);
            for &op in &ops {
                body.ins(op.to_insn());
            }
            let (bytes, relocs) = body.finish();
            prop_assert!(relocs.is_empty(), "a body without relocatable operands has no relocations");
            let decoded = decode_ops(&bytes);
            let expected = model_peephole(&ops, use_tee);
            prop_assert_eq!(decoded, expected, "tee fusion must match the reference model");
        }

        #[test]
        fn extra_locals_invariants(
            base in 0u32..32,
            actions in prop::collection::vec(
                prop_oneof![
                    2 => valtype_strategy().prop_map(PoolAction::Fresh),
                    5 => valtype_strategy().prop_map(PoolAction::Scratch),
                    2 => Just(PoolAction::Mark),
                    2 => Just(PoolAction::FreeLastMark),
                ],
                0..80,
            ),
        ) {
            let mut extra = ExtraLocals::new(base);
            let mut marks: Vec<(ScratchMark, usize)> = Vec::new();
            let mut fresh_set: HashSet<Local> = HashSet::new();
            let mut active: Vec<(ValType, Local)> = Vec::new();

            for action in actions {
                match action {
                    PoolAction::Fresh(ty) => {
                        let l = extra.fresh(ty);
                        prop_assert!(l.0 >= base, "fresh local {l:?} must not alias a parameter");
                        prop_assert!(fresh_set.insert(l), "duplicate fresh local {l:?}");
                        for &(_, al) in &active {
                            prop_assert_ne!(l, al, "fresh local aliased active scratch");
                        }
                        prop_assert_eq!(extra.types()[(l.0 - base) as usize], ty, "fresh local {:?} must have the requested type", l);
                    }
                    PoolAction::Scratch(ty) => {
                        let l = extra.scratch(ty);
                        prop_assert!(l.0 >= base, "scratch {l:?} must not alias a parameter");
                        prop_assert!(!fresh_set.contains(&l), "scratch aliased fresh local");
                        for &(_, al) in &active {
                            prop_assert_ne!(l, al, "scratch aliased another active scratch");
                        }
                        prop_assert_eq!(extra.types()[(l.0 - base) as usize], ty, "scratch {:?} must have the requested type", l);
                        active.push((ty, l));
                    }
                    PoolAction::Mark => {
                        marks.push((extra.mark(), active.len()));
                    }
                    PoolAction::FreeLastMark => {
                        if let Some((m, len)) = marks.pop() {
                            extra.free_since(m);
                            active.truncate(len);
                        }
                    }
                }
                // Total allocated types equals fresh locals + active scratches + free scratches.
                prop_assert_eq!(
                    extra.types().len(),
                    fresh_set.len() + extra.active_scratch.len() + extra.free_scratch.len(), "every local must be fresh, an active scratch, or a free scratch"
                );
            }

            // Reclaiming everything returns all scratches to the free pool and reuses them without growing `types`.
            extra.free_since(ScratchMark(0));
            let count_before = extra.types().len();
            for &(ty, _) in &active {
                let l = extra.scratch(ty);
                prop_assert_eq!(extra.types()[(l.0 - base) as usize], ty, "reused scratch {:?} must have the requested type", l);
            }
            prop_assert_eq!(extra.types().len(), count_before, "reusing freed scratches must not declare new locals");
        }
    }

    #[test]
    fn stack_frame_overflow_is_unsupported() {
        use cranelift_codegen::ir::{StackSlotData, StackSlotKind, ValueList};
        let mut func = Function::new();
        let b0 = func.dfg.make_block();
        func.layout.append_block(b0);
        let ret = func.dfg.make_inst(D::MultiAry {
            opcode: Opcode::Return,
            args: ValueList::default(),
        });
        func.layout.append_inst(ret, b0);
        func.sized_stack_slots.push(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            u32::MAX - 4,
            0,
        ));
        let mut config = Config::default();
        config.module.stack_pointer_global = Some(GlobalIndex(0));
        let err = crate::compile_function(&func, &config, &mut NoEnv).unwrap_err();
        assert_eq!(
            err,
            crate::Error::Unsupported("stack frame size overflow".into()),
            "a frame larger than the address space must be rejected"
        );
    }
}
