//! Expression trees: which values are passed on wasm's operand stack instead
//! of through locals.
//!
//! An expression tree is a root instruction plus the single-use definitions
//! folded into it, emitted together so that each folded result is pushed
//! right before the instruction that consumes it. This is what the
//! literature calls *stack scheduling*, and what LLVM's WebAssembly backend
//! calls *RegStackify* (`WebAssemblyRegStackify.cpp`).
//!
//! It is not the control-flow "stackifier" (LLVM's CFGStackify, or Yuri
//! Iozzelli's Stackifier algorithm): placing `block`/`loop` markers is a
//! different problem, solved here in `structure.rs` with Ramsey's
//! algorithm.
//!
//! # Prior art
//!
//! - The overall shape (walk each block bottom-up, visit operands in
//!   reverse, move single-use definitions next to their use) follows LLVM's
//!   RegStackify. LLVM checks that a move is safe by scanning the
//!   instructions in between.
//! - Side-effect *colors* come from Cranelift's own lowering, where a load
//!   may merge into its user only if no side effect lies between them (equal
//!   colors).
//! - The rule below, which lets one tree absorb several effectful
//!   definitions with an O(1) check each, and the rule for conditional
//!   operands, are specific to this crate. They are checked by differential
//!   fuzzing against Cranelift's interpreter, not by a published proof.
//!
//! # Algorithm
//!
//! Each block is walked backwards. Every live instruction that is not a
//! rematerialized constant starts as a *root* (`root_of[i] = Some(i)`). For
//! each root, its operands are visited in reverse emission order (the operand
//! stack is LIFO), depth-first, and an operand's defining instruction `d` is
//! folded into the tree (`root_of[d] = Some(root)`) when:
//!
//! - `d` is in the same block, has a single result, and that result has a
//!   single use;
//! - `d` is not a rematerialized constant (those are emitted at every use
//!   anyway); and
//! - if `d` has a side effect (including trapping and reading memory), the
//!   move does not reorder it with another side effect.
//!
//! The last check is O(1). `color[i]` is the number of effectful instructions
//! before `i` in its block. Folding effectful `d` into the tree rooted at `r`
//! moves `d` to just before the part of the tree built so far. It is safe iff
//! every effectful instruction between `d` and `r` has already been moved
//! into this tree (and therefore sits after `d`'s new position, in the same
//! relative order). Because trees are grown backwards, those are exactly the
//! effectful instructions counted so far, so the condition is
//! `color[r] - color[d] - 1 == moved`.
//!
//! Operands that are only evaluated on some paths (block arguments of a
//! conditional branch) only accept trees without side effects, since folding
//! would make an unconditional effect conditional.

use crate::analysis::{FuncInfo, has_lowering_side_effect, is_remat};
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{Function, Inst, Opcode, Value, ValueDef};

/// Emission recurses over trees, so bound their depth.
const MAX_TREE_DEPTH: u32 = 100;

pub(crate) struct ExprTrees {
    /// For every emitted instruction (roots and inlined nodes): the root of
    /// the expression tree containing it. A root maps to `Some(itself)`;
    /// dead instructions and rematerialized constants map to `None`.
    root_of: SecondaryMap<Inst, Option<Inst>>,
}

impl ExprTrees {
    /// Is `inst` the root of an emitted expression tree?
    pub fn is_root(&self, inst: Inst) -> bool {
        self.root_of[inst] == Some(inst)
    }

    /// Is `inst` folded into another instruction's expression tree?
    pub fn is_inlined(&self, inst: Inst) -> bool {
        self.root_of[inst].is_some_and(|r| r != inst)
    }

    /// The root of the tree containing `inst`, or `None` if `inst` is dead or
    /// rematerialized.
    pub fn root(&self, inst: Inst) -> Option<Inst> {
        self.root_of[inst]
    }
}

/// An operand in emission order. `conditional` operands are only evaluated on
/// some of the paths out of the instruction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Operand {
    pub value: Value,
    pub conditional: bool,
}

/// The operands of `inst` in the order they are pushed. This is the single
/// source of truth for evaluation order: tree building assumes it, and
/// lowering takes its operands from this list one by one
/// ([`crate::emit::Emitter::operand`]).
///
/// The exception is branches (`jump`, `brif`, `br_table`, `try_call`), which
/// `structure.rs` emits itself: the condition, index or call arguments
/// first, then each edge's arguments inside that edge's arm. Only those
/// first operands can hold side effects, because the arguments of a
/// multi-way branch or of a `try_call` (evaluated after the call) only
/// accept pure trees, so the order among the arguments does not matter.
pub(crate) fn operands(func: &Function, info: &FuncInfo, inst: Inst, out: &mut Vec<Operand>) {
    out.clear();
    let dfg = &func.dfg;
    let args = dfg.inst_args(inst);
    let plain = |out: &mut Vec<Operand>, vs: &[Value]| {
        out.extend(vs.iter().map(|&value| Operand {
            value,
            conditional: false,
        }))
    };
    let op = dfg.insts[inst].opcode();
    match op {
        Opcode::Select | Opcode::SelectSpectreGuard => {
            plain(out, &[args[1], args[2], args[0]]);
        }
        // `v128.bitselect` takes the mask last.
        Opcode::Bitselect if crate::fixed_type(dfg.value_type(args[0])).is_vector() => {
            plain(out, &[args[1], args[2], args[0]]);
        }
        Opcode::Store
        | Opcode::Istore8
        | Opcode::Istore16
        | Opcode::Istore32
        | Opcode::AtomicStore => {
            plain(out, &[args[1], args[0]]);
        }
        Opcode::CallIndirect | Opcode::ReturnCallIndirect => {
            plain(out, &args[1..]);
            plain(out, &args[..1]);
        }
        // A `try_call`'s edge arguments are evaluated after the call, so,
        // like those of a multi-way branch, they accept only pure trees.
        Opcode::TryCall | Opcode::TryCallIndirect => {
            if op == Opcode::TryCallIndirect {
                plain(out, &args[1..]);
                plain(out, &args[..1]);
            } else {
                plain(out, args);
            }
            let block = func.layout.inst_block(inst).unwrap();
            for &call in &info.dests[block] {
                out.extend(info.live_args(func, call).map(|value| Operand {
                    value,
                    conditional: true,
                }));
            }
        }
        Opcode::Jump | Opcode::Brif | Opcode::BrTable => {
            plain(out, args);
            let block = func.layout.inst_block(inst).unwrap();
            let dests = &info.dests[block];
            let conditional = dests.len() > 1;
            for &call in dests {
                out.extend(
                    info.live_args(func, call)
                        .map(|value| Operand { value, conditional }),
                );
            }
        }
        _ => plain(out, args),
    }
}

/// Build the expression trees of every reachable block. With `enabled` off,
/// every live non-constant instruction remains its own root.
pub(crate) fn build(func: &Function, info: &FuncInfo, enabled: bool) -> ExprTrees {
    let mut s = ExprTrees {
        root_of: SecondaryMap::new(),
    };
    let mut color: SecondaryMap<Inst, u32> = SecondaryMap::new();
    let mut effectful: SecondaryMap<Inst, bool> = SecondaryMap::new();
    let mut ops = Vec::new();
    // (value, allow side effects, depth)
    let mut worklist: Vec<(Value, bool, u32)> = Vec::new();

    for &block in &info.rpo {
        let mut c = 0;
        for inst in func.layout.block_insts(block) {
            if !info.live_inst[inst] || is_remat(func, inst) {
                continue;
            }
            s.root_of[inst] = Some(inst);
            color[inst] = c;
            if has_lowering_side_effect(func, inst) {
                effectful[inst] = true;
                c += 1;
            }
        }
        if !enabled {
            continue;
        }

        for root in func.layout.block_insts(block).rev() {
            if !s.is_root(root) {
                continue;
            }
            let mut moved = 0;
            operands(func, info, root, &mut ops);
            worklist.extend(ops.iter().map(|o| (o.value, !o.conditional, 1)));

            while let Some((v, allow_effects, depth)) = worklist.pop() {
                let ValueDef::Result(d, 0) = func.dfg.value_def(v) else {
                    continue;
                };
                if info.use_count[v] != 1
                    || !s.is_root(d)
                    || func.dfg.inst_results(d).len() != 1
                    || func.layout.inst_block(d) != Some(block)
                    || depth > MAX_TREE_DEPTH
                {
                    continue;
                }
                if effectful[d] {
                    if !allow_effects || color[root] - color[d] - 1 != moved {
                        continue;
                    }
                    moved += 1;
                }
                s.root_of[d] = Some(root);
                operands(func, info, d, &mut ops);
                worklist.extend(
                    ops.iter()
                        .map(|o| (o.value, allow_effects && !o.conditional, depth + 1)),
                );
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::cursor::{Cursor, FuncCursor};
    use cranelift_codegen::ir::{
        AbiParam, BlockArg, ExtFuncData, ExternalName, InstBuilder, MemFlagsData, Signature,
        UserExternalName, types,
    };
    use cranelift_codegen::isa::CallConv;
    use proptest::prelude::*;

    #[test]
    fn get_pinned_reg_is_not_inlined_across_set_pinned_reg() {
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        let mut func = Function::with_name_signature(Default::default(), sig);
        let b0 = func.dfg.make_block();
        func.layout.append_block(b0);
        let v0 = func.dfg.append_block_param(b0, types::I64);
        let mut pos = FuncCursor::new(&mut func).at_bottom(b0);
        let old = pos.ins().get_pinned_reg(types::I64);
        pos.ins().set_pinned_reg(v0);
        pos.ins().return_(&[old]);

        let info = FuncInfo::compute(&func).unwrap();
        let trees = build(&func, &info, true);
        let get_inst = func.layout.first_inst(b0).unwrap();
        assert!(
            trees.is_root(get_inst),
            "get_pinned_reg must remain a root when followed by set_pinned_reg"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum StepKind {
        Iconst,
        Iadd,
        Select,
        LoadNotrap,
        LoadTrap,
        Store,
        GetPinned,
        SetPinned,
        Udiv,
        Call1,
        UaddOverflow,
    }

    fn step_strategy() -> impl Strategy<Value = (StepKind, usize, usize, usize)> {
        (
            prop_oneof![
                Just(StepKind::Iconst),
                Just(StepKind::Iadd),
                Just(StepKind::Select),
                Just(StepKind::LoadNotrap),
                Just(StepKind::LoadTrap),
                Just(StepKind::Store),
                Just(StepKind::GetPinned),
                Just(StepKind::SetPinned),
                Just(StepKind::Udiv),
                Just(StepKind::Call1),
                Just(StepKind::UaddOverflow),
            ],
            0usize..32,
            0usize..32,
            0usize..32,
        )
    }

    /// Recursively simulate `Emitter::value` on `v`, appending any executed
    /// effectful instructions to `trace` in exact emission order.
    fn sim_value(
        func: &Function,
        info: &FuncInfo,
        trees: &ExprTrees,
        v: Value,
        trace: &mut Vec<Inst>,
    ) {
        if let ValueDef::Result(d, _) = func.dfg.value_def(v) {
            if is_remat(func, d) {
                return;
            }
            if trees.is_inlined(d) {
                sim_inst(func, info, trees, d, trace);
            }
        }
    }

    fn sim_inst(
        func: &Function,
        info: &FuncInfo,
        trees: &ExprTrees,
        inst: Inst,
        trace: &mut Vec<Inst>,
    ) {
        let mut ops = Vec::new();
        operands(func, info, inst, &mut ops);
        for op in ops {
            sim_value(func, info, trees, op.value, trace);
        }
        if has_lowering_side_effect(func, inst) {
            trace.push(inst);
        }
    }

    proptest! {
        #[test]
        fn expr_trees_preserve_effect_order_on_all_paths(
            steps in prop::collection::vec(step_strategy(), 1..20),
            term_kind in 0u8..3,
            arg_seeds in (0usize..32, 0usize..32, 0usize..32),
        ) {
            let mut sig = Signature::new(CallConv::SystemV);
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(types::I64));
            let mut func = Function::with_name_signature(Default::default(), sig);

            let mut callee_sig = Signature::new(CallConv::SystemV);
            callee_sig.params.push(AbiParam::new(types::I64));
            callee_sig.returns.push(AbiParam::new(types::I64));
            let sig_ref = func.dfg.signatures.push(callee_sig);
            let name_ref = func.declare_imported_user_function(UserExternalName {
                namespace: 0,
                index: 0,
            });
            let func_ref = func.dfg.ext_funcs.push(ExtFuncData {
                name: ExternalName::user(name_ref),
                signature: sig_ref,
                colocated: false,
                patchable: false,
            });

            let b0 = func.dfg.make_block();
            let b1 = func.dfg.make_block();
            let b2 = func.dfg.make_block();
            func.layout.append_block(b0);
            let p0 = func.dfg.append_block_param(b0, types::I64);
            let p1 = func.dfg.append_block_param(b0, types::I64);

            let mut vals = vec![p0, p1];
            let mut pos = FuncCursor::new(&mut func).at_bottom(b0);
            for (kind, ia, ib, ic) in steps {
                let va = vals[ia % vals.len()];
                let vb = vals[ib % vals.len()];
                let vc = vals[ic % vals.len()];
                match kind {
                    StepKind::Iconst => vals.push(pos.ins().iconst(types::I64, ia as i64)),
                    StepKind::Iadd => vals.push(pos.ins().iadd(va, vb)),
                    StepKind::Select => vals.push(pos.ins().select(va, vb, vc)),
                    StepKind::LoadNotrap => {
                        vals.push(pos.ins().load(types::I64, MemFlagsData::trusted(), va, 0))
                    }
                    StepKind::LoadTrap => {
                        vals.push(pos.ins().load(types::I64, MemFlagsData::new(), va, 0))
                    }
                    StepKind::Store => {
                        pos.ins().store(MemFlagsData::new(), va, vb, 0);
                    }
                    StepKind::GetPinned => vals.push(pos.ins().get_pinned_reg(types::I64)),
                    StepKind::SetPinned => {
                        pos.ins().set_pinned_reg(va);
                    }
                    StepKind::Udiv => vals.push(pos.ins().udiv(va, vb)),
                    StepKind::Call1 => {
                        let c = pos.ins().call(func_ref, &[va]);
                        vals.push(pos.func.dfg.first_result(c));
                    }
                    StepKind::UaddOverflow => {
                        let (s, _) = pos.ins().uadd_overflow(va, vb);
                        vals.push(s);
                    }
                }
            }

            let v_cond = vals[arg_seeds.0 % vals.len()];
            let v_x = vals[arg_seeds.1 % vals.len()];
            let v_y = vals[arg_seeds.2 % vals.len()];

            match term_kind {
                0 => {
                    pos.ins().return_(&[v_x]);
                }
                1 => {
                    pos.func.layout.append_block(b1);
                    let bp1 = pos.func.dfg.append_block_param(b1, types::I64);
                    let bp2 = pos.func.dfg.append_block_param(b1, types::I64);
                    pos.goto_bottom(b0);
                    pos.ins().jump(b1, &[BlockArg::Value(v_x), BlockArg::Value(v_y)]);
                    pos.goto_bottom(b1);
                    let sum = pos.ins().iadd(bp1, bp2);
                    pos.ins().return_(&[sum]);
                }
                _ => {
                    pos.func.layout.append_block(b1);
                    pos.func.layout.append_block(b2);
                    let bp1 = pos.func.dfg.append_block_param(b1, types::I64);
                    let bp2 = pos.func.dfg.append_block_param(b2, types::I64);
                    pos.goto_bottom(b0);
                    pos.ins().brif(
                        v_cond,
                        b1,
                        &[BlockArg::Value(v_x)],
                        b2,
                        &[BlockArg::Value(v_y)],
                    );
                    pos.goto_bottom(b1);
                    pos.ins().return_(&[bp1]);
                    pos.goto_bottom(b2);
                    pos.ins().return_(&[bp2]);
                }
            }

            let info = FuncInfo::compute(&func).unwrap();
            let trees = build(&func, &info, true);

            // Structural invariants on `trees`.
            for inst in func.layout.block_insts(b0) {
                if !info.live_inst[inst] || is_remat(&func, inst) {
                    prop_assert_eq!(trees.root(inst), None);
                } else {
                    let r = trees.root(inst).unwrap();
                    prop_assert!(trees.is_root(r));
                    if trees.is_inlined(inst) {
                        prop_assert_eq!(func.dfg.inst_results(inst).len(), 1);
                        let res = func.dfg.first_result(inst);
                        prop_assert_eq!(info.use_count[res], 1);
                    }
                }
            }

            // Expected linear sequence of live effectful instructions in `b0`.
            let expected_effects: Vec<Inst> = func
                .layout
                .block_insts(b0)
                .filter(|&i| info.live_inst[i] && has_lowering_side_effect(&func, i))
                .collect();

            // Simulate `Structurer` emission of `b0` on each outgoing path and verify
            // that the emitted sequence of effectful instructions equals `expected_effects`.
            let term = func.layout.last_inst(b0).unwrap();
            let mut body_effects = Vec::new();
            for inst in func.layout.block_insts(b0) {
                if inst != term && trees.is_root(inst) {
                    sim_inst(&func, &info, &trees, inst, &mut body_effects);
                }
            }

            match func.dfg.insts[term].opcode() {
                Opcode::Return => {
                    let mut trace = body_effects;
                    sim_inst(&func, &info, &trees, term, &mut trace);
                    prop_assert_eq!(trace, expected_effects);
                }
                Opcode::Jump => {
                    let mut trace = body_effects;
                    for a in info.live_args(&func, info.dests[b0][0]) {
                        sim_value(&func, &info, &trees, a, &mut trace);
                    }
                    trace.push(term);
                    prop_assert_eq!(trace, expected_effects);
                }
                Opcode::Brif => {
                    let mut prefix = body_effects;
                    let cond = func.dfg.inst_args(term)[0];
                    sim_value(&func, &info, &trees, cond, &mut prefix);
                    for &call in &info.dests[b0] {
                        let mut arm_trace = prefix.clone();
                        for a in info.live_args(&func, call) {
                            sim_value(&func, &info, &trees, a, &mut arm_trace);
                        }
                        arm_trace.push(term);
                        prop_assert_eq!(arm_trace, expected_effects.clone());
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

