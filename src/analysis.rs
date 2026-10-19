//! Function-level control-flow and data-flow analyses shared across passes.
//!
//! `FuncInfo::compute` performs a single-pass CFG and dominator analysis over
//! reachable blocks, followed by backwards value- and instruction-level
//! liveness (dead-code elimination and exact use counting).

use crate::{Result, unsupported};
use cranelift_codegen::dominator_tree::DominatorTree;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{
    Block, BlockArg, BlockCall, Function, Inst, InstructionData, Opcode, Value, ValueDef,
};

/// A block's position in reverse postorder (`0..info.rpo.len()`), or
/// [`RpoIndex::UNREACHABLE`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct RpoIndex(pub u32);

impl RpoIndex {
    pub const UNREACHABLE: Self = Self(u32::MAX);
}

impl Default for RpoIndex {
    fn default() -> Self {
        Self::UNREACHABLE
    }
}

pub(crate) struct FuncInfo {
    /// Reachable blocks in reverse postorder.
    pub rpo: Vec<Block>,
    /// Position of each block in `rpo`, or [`RpoIndex::UNREACHABLE`].
    pub rpo_num: SecondaryMap<Block, RpoIndex>,
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
    /// Number of uses of each value by live instructions and live block
    /// parameters. A value is live iff `use_count[v] > 0`.
    pub use_count: SecondaryMap<Value, u32>,
}

impl FuncInfo {
    /// Does this block have two or more incoming forward edges?
    pub fn is_merge(&self, b: Block) -> bool {
        self.forward_in[b] >= 2
    }

    /// Is the edge `from -> to` a retreating (backward) edge in RPO?
    pub fn is_backward(&self, from: Block, to: Block) -> bool {
        self.rpo_num[to] <= self.rpo_num[from]
    }

    /// Is `v` used by at least one live instruction or live block parameter?
    pub fn is_live(&self, v: Value) -> bool {
        self.use_count[v] > 0
    }

    pub fn compute(func: &Function) -> Result<FuncInfo> {
        let cfg = ControlFlowGraph::with_function(func);
        let domtree = DominatorTree::with_function(func, &cfg);
        let rpo: Vec<Block> = domtree.cfg_rpo().copied().collect();
        let mut rpo_num = SecondaryMap::with_default(RpoIndex::UNREACHABLE);
        for (i, &b) in rpo.iter().enumerate() {
            rpo_num[b] = RpoIndex(i as u32);
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
                | Opcode::TryCall
                | Opcode::TryCallIndirect
                | Opcode::Trap => {}
                op => return unsupported(format!("terminator {op}")),
            }
            // A `try_call` keeps every edge, in exception-table order with
            // the normal return last, because each becomes its own wasm
            // label; its edges may pass call results (`ret0`) and exception
            // payloads (`exn0`) instead of values.
            let is_try = matches!(data.opcode(), Opcode::TryCall | Opcode::TryCallIndirect);
            let mut ds: Vec<BlockCall> = Vec::new();
            for call in data.branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables) {
                if !is_try && call.args(pool).any(|a| !matches!(a, BlockArg::Value(_))) {
                    return unsupported("non-value block argument");
                }
                if is_try || !ds.iter().any(|d| same_call(func, d, call)) {
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
            use_count: SecondaryMap::new(),
        };
        info.compute_liveness(func);
        Ok(info)
    }

    /// Backwards value-level dead code elimination and exact use counting.
    ///
    /// Starting from effectful instructions and terminators, every operand of
    /// a newly live instruction and every incoming block argument of a newly
    /// live block parameter is pushed to the worklist once per use. Each pop
    /// therefore increments `use_count[v]`, and the first pop (`use_count == 1`)
    /// marks `v`'s definition live.
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
            self.use_count[v] += 1;
            if self.use_count[v] > 1 {
                continue;
            }
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
                BlockArg::Value(a) if self.is_live(p) => Some((a, p)),
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
    match func.dfg.insts[inst] {
        InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            ..
        }
        | InstructionData::UnaryIeee32 {
            opcode: Opcode::F32const,
            ..
        }
        | InstructionData::UnaryIeee64 {
            opcode: Opcode::F64const,
            ..
        }
        | InstructionData::UnaryIeee16 {
            opcode: Opcode::F16const,
            ..
        } => true,
        // Vectors of up to 128 bits fit a `v128.const`; others stay
        // ordinary (and unsupported) instructions.
        InstructionData::UnaryConst {
            opcode: Opcode::Vconst,
            ..
        } => func.dfg.value_type(func.dfg.first_result(inst)).bits() <= 128,
        _ => false,
    }
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
    // Unlike Cranelift's version, `get_pinned_reg` counts: here it reads a
    // global that `set_pinned_reg` writes, so the two must not be reordered.
    must_lower_even_if_unused(func, inst) || op.can_load()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::cursor::{Cursor, FuncCursor};
    use cranelift_codegen::entity::EntityRef;
    use cranelift_codegen::ir::{
        AbiParam, InstBuilder, JumpTableData, MemFlagsData, Signature, TrapCode, types,
    };
    use cranelift_codegen::isa::CallConv;
    use proptest::prelude::*;
    use std::collections::{HashMap, HashSet, VecDeque};

    #[test]
    fn get_pinned_reg_has_lowering_side_effect() {
        let mut func = Function::new();
        let b0 = func.dfg.make_block();
        func.layout.append_block(b0);
        let mut pos = FuncCursor::new(&mut func).at_bottom(b0);
        let v = pos.ins().get_pinned_reg(types::I64);
        pos.ins().return_(&[v]);
        let get_inst = func.layout.first_inst(b0).unwrap();
        // `get_pinned_reg` has `other_side_effects`, so it is kept and ordered
        // with other effects (Cranelift's lowering excludes it from both).
        assert!(must_lower_even_if_unused(&func, get_inst));
        assert!(has_lowering_side_effect(&func, get_inst));
    }

    /// Independent BFS reachability from `start`, optionally forbidding `avoid`.
    fn reachable_without(
        entry: Block,
        succs: &HashMap<Block, Vec<Block>>,
        avoid: Option<Block>,
    ) -> HashSet<Block> {
        let mut visited = HashSet::new();
        if Some(entry) == avoid {
            return visited;
        }
        let mut q = VecDeque::new();
        visited.insert(entry);
        q.push_back(entry);
        while let Some(b) = q.pop_front() {
            if let Some(nexts) = succs.get(&b) {
                for &n in nexts {
                    if Some(n) != avoid && visited.insert(n) {
                        q.push_back(n);
                    }
                }
            }
        }
        visited
    }

    fn analysis_func_strategy() -> impl Strategy<Value = Function> {
        (
            1usize..=7,
            prop::collection::vec(0usize..=3, 1..=7),
            prop::collection::vec(
                prop::collection::vec((0u8..8, 0usize..16, 0usize..16), 0..=5),
                1..=7,
            ),
            prop::collection::vec((0u8..5, 0usize..7, 0usize..7, 0usize..16), 1..=7),
        )
            .prop_map(|(num_blocks, param_counts, block_insts, terms)| {
                let mut sig = Signature::new(CallConv::SystemV);
                sig.params.push(AbiParam::new(types::I32));
                sig.returns.push(AbiParam::new(types::I32));
                let mut func = Function::with_name_signature(Default::default(), sig);

                let blocks: Vec<Block> = (0..num_blocks).map(|_| func.dfg.make_block()).collect();
                for (bi, &b) in blocks.iter().enumerate() {
                    func.layout.append_block(b);
                    if bi == 0 {
                        func.dfg.append_block_param(b, types::I32);
                    } else {
                        let np = param_counts.get(bi).copied().unwrap_or(0);
                        for _ in 0..np {
                            func.dfg.append_block_param(b, types::I32);
                        }
                    }
                }

                let entry_p = func.dfg.block_params(blocks[0])[0];
                let mut vals_per_block = Vec::with_capacity(num_blocks);

                for bi in 0..num_blocks {
                    let b = blocks[bi];
                    let mut vals = vec![entry_p];
                    if bi > 0 {
                        vals.extend_from_slice(func.dfg.block_params(b));
                    }
                    let mut pos = FuncCursor::new(&mut func).at_bottom(b);
                    if let Some(insts) = block_insts.get(bi) {
                        for &(kind, ia, ib) in insts {
                            let va = vals[ia % vals.len()];
                            let vb = vals[ib % vals.len()];
                            let r = match kind {
                                0 => pos.ins().iadd(va, vb),
                                1 => pos.ins().isub(va, vb),
                                2 => pos.ins().imul(va, vb),
                                3 => pos.ins().iconst(types::I32, ia as i64),
                                4 => pos.ins().load(types::I32, MemFlagsData::trusted(), va, 0),
                                5 => pos.ins().load(types::I32, MemFlagsData::new(), va, 0),
                                6 => {
                                    pos.ins().store(MemFlagsData::new(), va, vb, 0);
                                    continue;
                                }
                                _ => pos.ins().udiv(va, vb),
                            };
                            vals.push(r);
                        }
                    }
                    vals_per_block.push(vals);
                }

                let make_call = |func: &mut Function,
                                 target: Block,
                                 vals: &[Value],
                                 seed: usize|
                 -> BlockCall {
                    let np = func.dfg.block_params(target).len();
                    let args: Vec<BlockArg> = (0..np)
                        .map(|i| BlockArg::Value(vals[(seed + i) % vals.len()]))
                        .collect();
                    func.dfg.block_call(target, &args)
                };

                for bi in 0..num_blocks {
                    let b = blocks[bi];
                    let vals = &vals_per_block[bi];
                    let &(tkind, t1, t2, seed) = terms.get(bi).unwrap_or(&(0, 0, 0, 0));
                    let cond = vals[seed % vals.len()];
                    let ret_v = vals[(seed + 1) % vals.len()];
                    if num_blocks == 1 || tkind == 0 {
                        let mut pos = FuncCursor::new(&mut func).at_bottom(b);
                        pos.ins().return_(&[ret_v]);
                    } else if tkind == 4 {
                        let mut pos = FuncCursor::new(&mut func).at_bottom(b);
                        pos.ins().trap(TrapCode::user(1).unwrap());
                    } else {
                        let target1 = blocks[1 + (t1 % (num_blocks - 1))];
                        let target2 = blocks[1 + (t2 % (num_blocks - 1))];
                        match tkind {
                            1 => {
                                let c = make_call(&mut func, target1, vals, seed);
                                let mut pos = FuncCursor::new(&mut func).at_bottom(b);
                                let args: Vec<BlockArg> =
                                    c.args(&pos.func.dfg.value_lists).collect();
                                pos.ins().jump(target1, &args);
                            }
                            2 => {
                                let c1 = make_call(&mut func, target1, vals, seed);
                                let c2 = make_call(&mut func, target2, vals, seed + (t1 & 1));
                                let mut pos = FuncCursor::new(&mut func).at_bottom(b);
                                let a1: Vec<BlockArg> =
                                    c1.args(&pos.func.dfg.value_lists).collect();
                                let a2: Vec<BlockArg> =
                                    c2.args(&pos.func.dfg.value_lists).collect();
                                pos.ins().brif(cond, target1, &a1, target2, &a2);
                            }
                            _ => {
                                let c1 = make_call(&mut func, target1, vals, seed);
                                let c2 = make_call(&mut func, target2, vals, seed + 1);
                                let jt = func.dfg.jump_tables.push(JumpTableData::new(
                                    c1,
                                    &[c2, c1],
                                ));
                                let mut pos = FuncCursor::new(&mut func).at_bottom(b);
                                pos.ins().br_table(cond, jt);
                            }
                        }
                    }
                }

                func
            })
    }

    proptest! {
        #[test]
        fn func_info_matches_independent_oracle(func in analysis_func_strategy()) {
            let info = FuncInfo::compute(&func).unwrap();
            let entry = func.layout.entry_block().unwrap();
            let pool = &func.dfg.value_lists;

            // 1. Build independent adjacency list over all blocks.
            let mut raw_succs: HashMap<Block, Vec<Block>> = HashMap::new();
            for b in func.layout.blocks() {
                let term = func.layout.last_inst(b).unwrap();
                let mut s = Vec::new();
                for call in func.dfg.insts[term]
                    .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables)
                {
                    s.push(call.block(pool));
                }
                raw_succs.insert(b, s);
            }

            // 2. Reachable blocks in `info.rpo` must match independent BFS reachability.
            let reachable = reachable_without(entry, &raw_succs, None);
            let rpo_set: HashSet<Block> = info.rpo.iter().copied().collect();
            prop_assert_eq!(info.rpo.len(), rpo_set.len(), "duplicate blocks in rpo");
            prop_assert_eq!(&rpo_set, &reachable);
            prop_assert_eq!(info.rpo[0], entry);

            for b in func.layout.blocks() {
                if reachable.contains(&b) {
                    let idx = info.rpo.iter().position(|&x| x == b).unwrap();
                    prop_assert_eq!(info.rpo_num[b], RpoIndex(idx as u32));
                } else {
                    prop_assert_eq!(info.rpo_num[b], RpoIndex::UNREACHABLE);
                }
            }

            // 3. Verify `preds`, `forward_in`, `loop_header`, and `reducible` against definition of dominance.
            // Block `d` dominates reachable block `x` iff `x` is not reachable from `entry` when `d` is removed.
            let dominates = |d: Block, x: Block| -> bool {
                d == x || !reachable_without(entry, &raw_succs, Some(d)).contains(&x)
            };

            let mut expected_reducible = true;
            for &b in &info.rpo {
                let mut exp_preds = Vec::new();
                let mut exp_forward = 0u32;
                let mut exp_loop = false;
                for &p in &info.rpo {
                    for call in &info.dests[p] {
                        if call.block(pool) == b {
                            if !exp_preds.contains(&p) {
                                exp_preds.push(p);
                            }
                            if info.rpo_num[b] <= info.rpo_num[p] {
                                exp_loop = true;
                                if !dominates(b, p) {
                                    expected_reducible = false;
                                }
                            } else {
                                exp_forward += 1;
                            }
                        }
                    }
                }
                prop_assert_eq!(&info.preds[b], &exp_preds);
                prop_assert_eq!(info.forward_in[b], exp_forward);
                prop_assert_eq!(info.loop_header[b], exp_loop);
            }
            prop_assert_eq!(info.reducible, expected_reducible);

            // 4. Independent iterative fixpoint oracle for `live_inst` and `use_count`.
            let mut exp_live_inst: HashSet<Inst> = HashSet::new();
            let mut exp_live_val: HashSet<Value> = HashSet::new();
            for &b in &info.rpo {
                for inst in func.layout.block_insts(b) {
                    if must_lower_even_if_unused(&func, inst)
                        || func.dfg.insts[inst].opcode().is_terminator()
                    {
                        exp_live_inst.insert(inst);
                    }
                }
            }
            loop {
                let mut changed = false;
                for &b in &info.rpo {
                    for inst in func.layout.block_insts(b) {
                        if exp_live_inst.contains(&inst) {
                            for &arg in func.dfg.inst_args(inst) {
                                if exp_live_val.insert(arg) {
                                    changed = true;
                                }
                            }
                        } else if func
                            .dfg
                            .inst_results(inst)
                            .iter()
                            .any(|r| exp_live_val.contains(r))
                        {
                            exp_live_inst.insert(inst);
                            changed = true;
                        }
                    }
                    for (idx, &param) in func.dfg.block_params(b).iter().enumerate() {
                        if exp_live_val.contains(&param) {
                            for &p in &info.preds[b] {
                                for call in &info.dests[p] {
                                    if call.block(pool) == b
                                        && let BlockArg::Value(a) = call.args(pool).nth(idx).unwrap()
                                        && exp_live_val.insert(a)
                                    {
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }

            // Count uses across all live instructions and live block parameters in reachable blocks.
            let mut exp_use_count: HashMap<Value, u32> = HashMap::new();
            for &b in &info.rpo {
                for inst in func.layout.block_insts(b) {
                    prop_assert_eq!(info.live_inst[inst], exp_live_inst.contains(&inst));
                    if exp_live_inst.contains(&inst) {
                        for &arg in func.dfg.inst_args(inst) {
                            *exp_use_count.entry(arg).or_default() += 1;
                        }
                    }
                }
                for (idx, &param) in func.dfg.block_params(b).iter().enumerate() {
                    if exp_live_val.contains(&param) {
                        for &p in &info.preds[b] {
                            for call in &info.dests[p] {
                                if call.block(pool) == b
                                    && let BlockArg::Value(a) = call.args(pool).nth(idx).unwrap()
                                {
                                    *exp_use_count.entry(a).or_default() += 1;
                                }
                            }
                        }
                    }
                }
            }

            for vi in 0..func.dfg.num_values() {
                let v = Value::new(vi);
                let exp = exp_use_count.get(&v).copied().unwrap_or(0);
                prop_assert_eq!(info.use_count[v], exp, "use_count mismatch for {}", v);
                prop_assert_eq!(info.is_live(v), exp_live_val.contains(&v));
            }
        }
    }
}

