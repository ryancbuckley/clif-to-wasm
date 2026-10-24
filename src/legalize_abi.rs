//! Multi-result return demotion ([`crate::ModuleLayout::demote_multi_results`]).
//!
//! Under LLVM's Basic C ABI for WebAssembly, functions returning more than one
//! wasm value (counting `i128`/`f128` as two `i64`s) instead take a hidden
//! first pointer parameter to caller-allocated space, store their results at
//! C-struct-aligned offsets there, and return nothing. This pass rewrites both
//! function definitions and call sites before `i128` legalization.

use crate::{Result, unsupported};
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{
    AbiParam, ArgumentPurpose, Function, Inst, InstBuilder, InstructionData, MemFlagsData, Opcode,
    Signature, StackSlotData, StackSlotKind, Type, Value, types,
};

/// Number of wasm scalar values needed to represent `ps`.
fn wasm_len(ps: &[AbiParam]) -> usize {
    ps.iter()
        .map(|p| match p.value_type {
            types::I128 | types::F128 => 2,
            _ => 1,
        })
        .sum()
}

pub(crate) fn needs_demotion(sig: &Signature) -> bool {
    wasm_len(&sig.returns) > 1
}

/// Whether `func` or any signature it references returns multiple wasm values.
pub(crate) fn scan(func: &Function) -> bool {
    needs_demotion(&func.signature) || func.dfg.signatures.values().any(needs_demotion)
}

/// Byte offset of each result in the return area and total byte size.
fn layout(types: impl Iterator<Item = Type>) -> (Vec<u32>, u32) {
    let mut size = 0u32;
    let offsets = types
        .map(|ty| {
            let align = ty.bytes().min(16).max(1);
            let off = size.next_multiple_of(align);
            size = off + ty.bytes();
            off
        })
        .collect();
    (offsets, size)
}

fn demoted(sig: &Signature, ptr: Type) -> Signature {
    let mut sig = sig.clone();
    sig.returns.clear();
    sig.params
        .insert(0, AbiParam::special(ptr, ArgumentPurpose::StructReturn));
    sig
}

/// Rewrite `func` so that no signature returns more than one wasm value.
pub(crate) fn demote(func: &mut Function, ptr: Type) -> Result<()> {
    let flags = MemFlagsData::trusted();

    // The function's own return area pointer, as a new first parameter.
    let own = func.signature.returns.clone();
    let mut ret_ptr: Option<Value> = None;
    if needs_demotion(&func.signature) {
        let entry = func
            .layout
            .entry_block()
            .expect("function has an entry block");
        let old = func.dfg.detach_block_params(entry);
        let old: Vec<Value> = old.as_slice(&func.dfg.value_lists).to_vec();
        ret_ptr = Some(func.dfg.append_block_param(entry, ptr));
        for v in old {
            func.dfg.attach_block_param(entry, v);
        }
        func.signature = demoted(&func.signature, ptr);
    }
    let (own_offsets, _) = layout(own.iter().map(|p| p.value_type));

    let mut demoted_sigs = cranelift_codegen::entity::SecondaryMap::new();
    for (sig_ref, sig) in func.dfg.signatures.iter_mut() {
        if needs_demotion(sig) {
            *sig = demoted(sig, ptr);
            demoted_sigs[sig_ref] = true;
        }
    }

    // One stack slot, sized for the largest, serves every call: results are
    // loaded right after it.
    let mut area_size = 0;
    for b in func.layout.blocks() {
        for inst in func.layout.block_insts(b) {
            if let Some(sig_ref) = func.dfg.call_signature(inst)
                && demoted_sigs[sig_ref]
            {
                let results = func.dfg.inst_results(inst);
                let (_, size) = layout(results.iter().map(|&r| func.dfg.value_type(r)));
                area_size = area_size.max(size);
            }
        }
    }
    let area = (area_size > 0).then(|| {
        func.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            area_size,
            4,
        ))
    });

    let mut pos = FuncCursor::new(func);
    while pos.next_block().is_some() {
        while let Some(inst) = pos.next_inst() {
            let op = pos.func.dfg.insts[inst].opcode();
            let Some(sig_ref) = pos.func.dfg.call_signature(inst) else {
                if op == Opcode::Return
                    && let Some(p) = ret_ptr
                {
                    let args = pos.func.dfg.inst_args(inst).to_vec();
                    for (&v, &off) in args.iter().zip(&own_offsets) {
                        pos.ins().store(flags, v, p, off as i32);
                    }
                    pos.ins().return_(&[]);
                    pos.remove_inst_and_step_back();
                }
                continue;
            };
            if !demoted_sigs[sig_ref] {
                continue;
            }
            match op {
                Opcode::Call | Opcode::CallIndirect => {
                    let results = pos.func.dfg.inst_results(inst).to_vec();
                    let (offsets, _) = layout(results.iter().map(|&r| pos.func.dfg.value_type(r)));
                    let addr = pos.ins().stack_addr(ptr, area.unwrap(), 0);
                    insert_first_arg(pos.func, inst, addr);
                    pos.func.dfg.detach_inst_results(inst);
                    pos.next_inst();
                    for (&r, &off) in results.iter().zip(&offsets) {
                        let ty = pos.func.dfg.value_type(r);
                        let v = pos.ins().load(ty, flags, addr, off as i32);
                        pos.func.dfg.change_to_alias(r, v);
                    }
                    pos.prev_inst();
                }
                Opcode::ReturnCall | Opcode::ReturnCallIndirect => {
                    let Some(p) = ret_ptr else {
                        return unsupported("tail call returning more values than the caller");
                    };
                    insert_first_arg(pos.func, inst, p);
                }
                _ => return unsupported(format!("{op} returning more than one value")),
            }
        }
    }
    Ok(())
}

/// Insert `v` as the first argument passed to the callee of call `inst`
/// (after the callee address of an indirect call).
fn insert_first_arg(func: &mut Function, inst: Inst, v: Value) {
    let dfg = &mut func.dfg;
    let pool = &mut dfg.value_lists;
    match &mut dfg.insts[inst] {
        InstructionData::Call { args, .. } => args.insert(0, v, pool),
        InstructionData::CallIndirect { args, .. } => args.insert(1, v, pool),
        _ => unreachable!("not a call"),
    }
}
