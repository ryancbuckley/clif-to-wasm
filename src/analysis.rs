//! Function-level analyses shared by the later passes.

use crate::{Result, unsupported};
use cranelift_codegen::dominator_tree::DominatorTree;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{
    Block, BlockArg, BlockCall, Function, Inst, InstructionData, Opcode, Value, ValueDef,
};

pub(crate) const UNREACHABLE: u32 = u32::MAX;

pub(crate) struct FuncInfo {
    /// Reachable blocks in reverse postorder.
    pub rpo: Vec<Block>,
    /// Position of each block in `rpo`, or `UNREACHABLE`.
    pub rpo_num: SecondaryMap<Block, u32>,
    /// The distinct outgoing edges of each block's terminator. Two
    /// `BlockCall`s are the same edge when they have the same target and the
    /// same arguments; e.g. `brif v, b1(x), b1(x)` has one edge.
    pub dests: SecondaryMap<Block, Vec<BlockCall>>,
    /// Distinct reachable predecessor blocks.
    pub preds: SecondaryMap<Block, Vec<Block>>,
    /// Number of forward edges (from a lower RPO number) into each block.
    pub forward_in: SecondaryMap<Block, u32>,
    /// Does any back edge target this block?
    pub loop_header: SecondaryMap<Block, bool>,
    pub domtree: DominatorTree,
    /// Is every retreating edge a back edge to a dominator?
    pub reducible: bool,
    /// Instructions that must be emitted.
    pub live_inst: SecondaryMap<Inst, bool>,
    /// Values whose content is needed.
    pub live_val: SecondaryMap<Value, bool>,
    /// Number of uses of each value by live instructions, counting block
    /// arguments only when the receiving parameter is live.
    pub use_count: SecondaryMap<Value, u32>,
}

impl FuncInfo {
    pub fn is_merge(&self, b: Block) -> bool {
        self.forward_in[b] >= 2
    }

    pub fn is_backward(&self, from: Block, to: Block) -> bool {
        self.rpo_num[to] <= self.rpo_num[from]
    }

    pub fn compute(func: &Function) -> Result<FuncInfo> {
        let cfg = ControlFlowGraph::with_function(func);
        let domtree = DominatorTree::with_function(func, &cfg);
        let rpo: Vec<Block> = domtree.cfg_rpo().copied().collect();
        let mut rpo_num = SecondaryMap::with_default(UNREACHABLE);
        for (i, &b) in rpo.iter().enumerate() {
            rpo_num[b] = i as u32;
        }

        let pool = &func.dfg.value_lists;
        let mut dests: SecondaryMap<Block, Vec<BlockCall>> = SecondaryMap::new();
        let mut preds: SecondaryMap<Block, Vec<Block>> = SecondaryMap::new();
        let mut forward_in = SecondaryMap::new();
        let mut loop_header = SecondaryMap::new();
        let mut reducible = true;

        for &b in &rpo {
            let Some(term) = func.layout.last_inst(b) else {
                return unsupported(format!("empty block {b}"));
            };
            let data = &func.dfg.insts[term];
            match data.opcode() {
                Opcode::Jump
                | Opcode::Brif
                | Opcode::BrTable
                | Opcode::Return
                | Opcode::ReturnCall
                | Opcode::ReturnCallIndirect
                | Opcode::Trap => {}
                op => return unsupported(format!("terminator {op}")),
            }
            let mut ds: Vec<BlockCall> = Vec::new();
            for call in data.branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables) {
                if call.args(pool).any(|a| !matches!(a, BlockArg::Value(_))) {
                    return unsupported("non-value block argument");
                }
                if !ds.iter().any(|d| same_call(func, d, call)) {
                    ds.push(*call);
                }
            }
            for call in &ds {
                let t = call.block(pool);
                if !preds[t].contains(&b) {
                    preds[t].push(b);
                }
                if rpo_num[t] <= rpo_num[b] {
                    loop_header[t] = true;
                    if !domtree.block_dominates(t, b) {
                        reducible = false;
                    }
                } else {
                    forward_in[t] += 1;
                }
            }
            dests[b] = ds;
        }

        let mut info = FuncInfo {
            rpo,
            rpo_num,
            dests,
            preds,
            forward_in,
            loop_header,
            domtree,
            reducible,
            live_inst: SecondaryMap::new(),
            live_val: SecondaryMap::new(),
            use_count: SecondaryMap::new(),
        };
        info.compute_liveness(func);
        Ok(info)
    }

    /// Value-level dead code elimination, followed by use counting.
    fn compute_liveness(&mut self, func: &Function) {
        let dfg = &func.dfg;
        let pool = &dfg.value_lists;
        let mut worklist: Vec<Value> = Vec::new();

        for &b in &self.rpo {
            for inst in func.layout.block_insts(b) {
                if must_lower_even_if_unused(func, inst) || dfg.insts[inst].opcode().is_terminator()
                {
                    self.live_inst[inst] = true;
                    worklist.extend_from_slice(dfg.inst_args(inst));
                }
            }
        }

        while let Some(v) = worklist.pop() {
            if self.live_val[v] {
                continue;
            }
            self.live_val[v] = true;
            match dfg.value_def(v) {
                ValueDef::Result(inst, _) => {
                    if !self.live_inst[inst] {
                        self.live_inst[inst] = true;
                        worklist.extend_from_slice(dfg.inst_args(inst));
                    }
                }
                ValueDef::Param(block, idx) => {
                    for &p in &self.preds[block] {
                        for call in &self.dests[p] {
                            if call.block(pool) == block
                                && let BlockArg::Value(a) = call.args(pool).nth(idx).unwrap()
                            {
                                worklist.push(a);
                            }
                        }
                    }
                }
                ValueDef::Union(..) => unreachable!("unions only exist inside the egraph pass"),
            }
        }

        for &b in &self.rpo {
            for inst in func.layout.block_insts(b) {
                if self.live_inst[inst] {
                    for &v in dfg.inst_args(inst) {
                        self.use_count[v] += 1;
                    }
                }
            }
            for i in 0..self.dests[b].len() {
                let call = self.dests[b][i];
                let args: Vec<Value> = self.live_args(func, call).collect();
                for a in args {
                    self.use_count[a] += 1;
                }
            }
        }
    }

    /// The arguments of `call` whose receiving block parameter is live, paired
    /// with that parameter.
    pub fn live_arg_pairs<'a>(
        &'a self,
        func: &'a Function,
        call: BlockCall,
    ) -> impl Iterator<Item = (Value, Value)> + 'a {
        let pool = &func.dfg.value_lists;
        let params = func.dfg.block_params(call.block(pool));
        call.args(pool)
            .zip(params.iter())
            .filter_map(move |(a, &p)| match a {
                BlockArg::Value(a) if self.live_val[p] => Some((a, p)),
                _ => None,
            })
    }

    pub fn live_args<'a>(
        &'a self,
        func: &'a Function,
        call: BlockCall,
    ) -> impl Iterator<Item = Value> + 'a {
        self.live_arg_pairs(func, call).map(|(a, _)| a)
    }
}

pub(crate) fn same_call(func: &Function, a: &BlockCall, b: &BlockCall) -> bool {
    let pool = &func.dfg.value_lists;
    a.block(pool) == b.block(pool) && a.args(pool).eq(b.args(pool))
}

/// Is this instruction a constant that is rematerialized at each use instead
/// of being stored in a local?
pub(crate) fn is_remat(func: &Function, inst: Inst) -> bool {
    matches!(
        func.dfg.insts[inst],
        InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            ..
        } | InstructionData::UnaryIeee32 {
            opcode: Opcode::F32const,
            ..
        } | InstructionData::UnaryIeee64 {
            opcode: Opcode::F64const,
            ..
        }
    )
}

/// Mirrors `cranelift_codegen::inst_predicates` (which is private).
fn trivially_has_side_effects(op: Opcode) -> bool {
    op.is_call()
        || op.is_branch()
        || op.is_terminator()
        || op.is_return()
        || op.can_trap()
        || op.other_side_effects()
        || op.can_store()
}

/// Must the instruction be emitted even when its results are unused?
pub(crate) fn must_lower_even_if_unused(func: &Function, inst: Inst) -> bool {
    let data = &func.dfg.insts[inst];
    let op = data.opcode();
    if trivially_has_side_effects(op) {
        return true;
    }
    if !op.can_load() {
        return false;
    }
    match *data {
        InstructionData::Load { flags, .. } => !func.dfg.mem_flags[flags].notrap(),
        _ => true,
    }
}

/// Does the instruction have an effect that must not be reordered with
/// another effect? This includes trapping and reading memory, and is what
/// Cranelift's lowering uses to compute side-effect colors.
pub(crate) fn has_lowering_side_effect(func: &Function, inst: Inst) -> bool {
    let op = func.dfg.insts[inst].opcode();
    op != Opcode::GetPinnedReg && (must_lower_even_if_unused(func, inst) || op.can_load())
}
