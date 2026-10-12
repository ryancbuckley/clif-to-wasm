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
//! Each block is walked backwards. Every live instruction that has not been
//! folded into a later one is a *root*. For each root, its operands are
//! visited in reverse emission order (the operand stack is LIFO), depth-first,
//! and an operand's defining instruction is folded into the tree when:
//!
//! - it is in the same block, has exactly one result and that result has
//!   exactly one use;
//! - it is not a rematerialized constant (those are emitted at every use
//!   anyway); and
//! - if it has a side effect (including trapping and reading memory), the
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
    /// The result of this instruction is consumed directly from the stack by
    /// its only user, so the instruction is emitted as part of that user's
    /// tree.
    pub inlined: SecondaryMap<Inst, bool>,
    /// For inlined instructions: the root of the tree containing it.
    pub root_of: SecondaryMap<Inst, Option<Inst>>,
}

impl ExprTrees {
    pub fn is_root(&self, info: &FuncInfo, func: &Function, inst: Inst) -> bool {
        info.live_inst[inst] && !self.inlined[inst] && !is_remat(func, inst)
    }

    pub fn root(&self, inst: Inst) -> Inst {
        self.root_of[inst].unwrap_or(inst)
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
/// every live instruction is its own root.
pub(crate) fn build(func: &Function, info: &FuncInfo, enabled: bool) -> ExprTrees {
    let mut s = ExprTrees {
        inlined: SecondaryMap::new(),
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
            color[inst] = c;
            if has_lowering_side_effect(func, inst) {
                effectful[inst] = true;
                c += 1;
            }
        }

        for root in func.layout.block_insts(block).rev() {
            if !enabled || !s.is_root(info, func, root) {
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
                    || func.dfg.inst_results(d).len() != 1
                    || func.layout.inst_block(d) != Some(block)
                    || is_remat(func, d)
                    || depth > MAX_TREE_DEPTH
                {
                    continue;
                }
                debug_assert!(!s.inlined[d]);
                if effectful[d] {
                    if !allow_effects || color[root] - color[d] - 1 != moved {
                        continue;
                    }
                    moved += 1;
                }
                s.inlined[d] = true;
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
