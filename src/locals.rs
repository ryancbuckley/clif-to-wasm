//! Live-range analysis and local allocation for non-inlined SSA values.
//!
//! # Live ranges
//!
//! Each root instruction receives a position in RPO, with extra positions for
//! block starts and ends. A value's live range is a sorted list of
//! non-overlapping half-open [`Segment`]s (at most two per block), built by
//! walking predecessors backward from each use to the definition.
//!
//! Two conventions make block-argument moves cheap to coalesce (relied on by
//! [`Locals::move_needed`]):
//!
//! - A block parameter is defined at each predecessor's terminator, so in a
//!   predecessor it is live only from the terminator to the block end.
//! - Segments are half-open (`[lo, hi)`): a value whose last use is at `p`
//!   does not overlap a value defined at `p`, because instructions and parallel
//!   block-argument moves read all inputs before writing any outputs.
//!
//! Thus a branch argument at its last use can share a local with the target
//! block parameter.
//!
//! # Coloring
//!
//! Values are colored greedily by start position per wasm [`ValType`], trying
//! in order:
//!
//! 1. a *move partner*'s local (an argument or parameter paired with this value
//!    in a block-argument move), eliminating the move;
//! 2. an expired local whose segments all end at or before this value's start
//!    (tracked via a min-heap);
//! 3. up to [`SCAN_LIMIT`] recently allocated locals whose segments do not
//!    overlap this value's segments (reusing live-range holes);
//! 4. a fresh local.

use crate::analysis::FuncInfo;
use crate::expr_trees::{ExprTrees, Operand, operands};
use crate::{Result, wasm_type};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::{Block, Function, Inst, Value, ValueDef};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use wasm_encoder::ValType;

/// Index of a WebAssembly local variable within a function (parameters
/// `0..num_params` followed by declared locals).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct Local(pub u32);

impl Local {
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for Local {
    fn default() -> Self {
        Self::NONE
    }
}

/// How many existing locals a value may be checked against for a
/// non-overlapping slot, beyond its move partners and the free list.
const SCAN_LIMIT: usize = 32;

/// A program point in the linearized RPO instruction sequence (block starts,
/// root instructions, and block ends).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
struct Pos(u32);

/// Per-value visitation stamp in [`Scratch`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Stamp(u32);

/// A half-open interval `[lo, hi)` of program positions.
type Segment = (Pos, Pos);

pub(crate) struct Locals {
    /// The local holding each value, or [`Local::NONE`].
    pub local_of: SecondaryMap<Value, Local>,
    /// Types of the declared locals, starting after the parameters.
    pub decls: Vec<ValType>,
    pub num_params: u32,
}

impl Locals {
    pub fn next_index(&self) -> u32 {
        self.num_params + self.decls.len() as u32
    }

    /// Whether the block-argument move `param := arg` must be emitted.
    ///
    /// Returns false when both share a local. This is safe even in parallel
    /// moves because all needed arguments are pushed before any parameter is
    /// written, and non-overlapping live ranges guarantee that no other
    /// argument or parameter of the same branch shares that local.
    pub fn move_needed(&self, arg: Value, param: Value) -> bool {
        let la = self.local_of[arg];
        la == Local::NONE || la != self.local_of[param]
    }
}

pub(crate) fn allocate(
    func: &Function,
    info: &FuncInfo,
    trees: &ExprTrees,
    coalesce: bool,
) -> Result<Locals> {
    let ranges = compute_live_ranges(func, info, trees);
    let entry_params = func.dfg.block_params(info.rpo[0]);
    let num_params = entry_params.len() as u32;
    let mut local_of = SecondaryMap::with_default(Local::NONE);
    let moves = if coalesce { &ranges.moves[..] } else { &[] };
    let partners = Buckets::new(
        func.dfg.num_values(),
        moves.iter().flat_map(|&(a, p)| [(a, p), (p, a)]),
    );

    let mut slots: Vec<Slot> = Vec::new();
    let mut classes: Vec<Class> = Vec::new();
    for &param in entry_params {
        let ty = wasm_type(func.dfg.value_type(param))?;
        local_of[param] = new_slot(&mut slots, &mut classes, ty);
    }

    for (v, segs) in ranges.iter() {
        let ty = wasm_type(func.dfg.value_type(v))?;
        let (lo, hi) = (segs[0].0, segs.last().unwrap().1);
        let fits = |slots: &[Slot], l: Local| {
            let s = &slots[l.0 as usize];
            s.ty == ty && !overlaps(&s.segs, segs)
        };

        let chosen = if local_of[v] != Local::NONE {
            local_of[v]
        } else if let Some(l) = partners
            .get(v)
            .iter()
            .map(|&w| local_of[w])
            .find(|&l| l != Local::NONE && fits(&slots, l))
        {
            l
        } else {
            let class = class_of(&mut classes, ty);
            class
                .take_free(&slots, lo)
                .or_else(|| {
                    let recent = class.all.iter().rev().take(SCAN_LIMIT);
                    recent.copied().find(|&l| fits(&slots, l))
                })
                .unwrap_or_else(|| new_slot(&mut slots, &mut classes, ty))
        };

        let slot = &mut slots[chosen.0 as usize];
        merge_into(&mut slot.segs, segs);
        slot.end = slot.end.max(hi);
        let end = slot.end;
        local_of[v] = chosen;
        class_of(&mut classes, ty).ends.push(Reverse((end, chosen)));
    }

    let decls = slots[num_params as usize..].iter().map(|s| s.ty).collect();
    Ok(Locals {
        local_of,
        decls,
        num_params,
    })
}

// ---- Live-range construction ------------------------------------------------

struct LiveRanges {
    /// `(value, first_segment_index, segment_count)`, sorted by the start of
    /// the value's first segment.
    values: Vec<(Value, u32, u32)>,
    segments: Vec<Segment>,
    /// Block-argument moves `(argument, parameter)` where both need a local.
    moves: Vec<(Value, Value)>,
}

impl LiveRanges {
    fn iter(&self) -> impl Iterator<Item = (Value, &[Segment])> {
        self.values
            .iter()
            .map(|&(v, first, len)| (v, &self.segments[first as usize..(first + len) as usize]))
    }
}

/// Does `v` need a local (as opposed to being dead, rematerialized, or
/// passed on the operand stack)?
fn needs_local(func: &Function, info: &FuncInfo, trees: &ExprTrees, v: Value) -> bool {
    info.is_live(v)
        && match func.dfg.value_def(v) {
            ValueDef::Result(d, _) => trees.is_root(d),
            ValueDef::Param(..) => true,
            ValueDef::Union(..) => unreachable!(),
        }
}

/// Per-block scratch state for building one value's segments.
struct Scratch {
    stamp: SecondaryMap<Block, Stamp>,
    live_in: SecondaryMap<Block, Stamp>,
    lo: SecondaryMap<Block, Pos>,
    hi: SecondaryMap<Block, Pos>,
    term_def: SecondaryMap<Block, bool>,
    touched: Vec<Block>,
}

impl Scratch {
    fn touch(&mut self, b: Block, cur: Stamp) {
        if self.stamp[b] != cur {
            self.stamp[b] = cur;
            self.lo[b] = Pos(u32::MAX);
            self.hi[b] = Pos(0);
            self.term_def[b] = false;
            self.touched.push(b);
        }
    }

    fn add(&mut self, b: Block, cur: Stamp, p: Pos) {
        self.touch(b, cur);
        self.lo[b] = self.lo[b].min(p);
        self.hi[b] = self.hi[b].max(p);
    }
}

fn compute_live_ranges(func: &Function, info: &FuncInfo, trees: &ExprTrees) -> LiveRanges {
    let entry = info.rpo[0];
    let mut pos: SecondaryMap<Inst, Pos> = SecondaryMap::new();
    let mut start: SecondaryMap<Block, Pos> = SecondaryMap::new();
    let mut end: SecondaryMap<Block, Pos> = SecondaryMap::new();
    let mut p = 0u32;
    for &b in &info.rpo {
        start[b] = Pos(p);
        p += 1;
        for inst in func.layout.block_insts(b) {
            if trees.is_root(inst) {
                pos[inst] = Pos(p);
                p += 1;
            }
        }
        end[b] = Pos(p);
        p += 1;
    }
    let term_pos = |b: Block| pos[func.layout.last_inst(b).unwrap()];

    // Collect uses of each value that needs a local, and block-argument moves.
    let is_local = |v: Value| needs_local(func, info, trees, v);
    let mut use_list: Vec<(Value, Block, Pos)> = Vec::new();
    let mut moves: Vec<(Value, Value)> = Vec::new();
    let mut ops: Vec<Operand> = Vec::new();
    for &b in &info.rpo {
        for inst in func.layout.block_insts(b) {
            let Some(root) = trees.root(inst) else {
                continue;
            };
            let at_block = func.layout.inst_block(root).unwrap();
            operands(func, info, inst, &mut ops);
            for o in &ops {
                if is_local(o.value) {
                    use_list.push((o.value, at_block, pos[root]));
                }
            }
        }
        for &call in &info.dests[b] {
            for (a, param) in info.live_arg_pairs(func, call) {
                if is_local(a) {
                    moves.push((a, param));
                }
            }
        }
    }
    let num_values = func.dfg.num_values();
    let uses = Buckets::new(num_values, use_list.iter().map(|&(v, b, p)| (v, (b, p))));

    let mut scratch = Scratch {
        stamp: SecondaryMap::new(),
        live_in: SecondaryMap::new(),
        lo: SecondaryMap::new(),
        hi: SecondaryMap::new(),
        term_def: SecondaryMap::new(),
        touched: Vec::new(),
    };
    let mut walk: Vec<Block> = Vec::new();
    let mut seg_buf: Vec<Segment> = Vec::new();
    let mut values: Vec<(Value, u32, u32)> = Vec::new();
    for vi in 0..num_values {
        let v = Value::new(vi);
        let v_uses = uses.get(v);
        if v_uses.is_empty() {
            continue;
        }
        let cur = Stamp(vi as u32 + 1);
        scratch.touched.clear();
        let def_block = match func.dfg.value_def(v) {
            ValueDef::Result(d, _) => {
                let b = func.layout.inst_block(d).unwrap();
                scratch.add(b, cur, pos[d]);
                b
            }
            ValueDef::Param(b, _) => {
                scratch.add(b, cur, start[b]);
                if b != entry {
                    for &pred in &info.preds[b] {
                        scratch.touch(pred, cur);
                        scratch.term_def[pred] = true;
                    }
                }
                b
            }
            ValueDef::Union(..) => unreachable!(),
        };
        for &(ub, upos) in v_uses {
            scratch.add(ub, cur, upos);
            if ub == def_block || scratch.live_in[ub] == cur {
                continue;
            }
            scratch.live_in[ub] = cur;
            scratch.add(ub, cur, start[ub]);
            walk.push(ub);
            while let Some(x) = walk.pop() {
                for &pred in &info.preds[x] {
                    scratch.add(pred, cur, end[pred]);
                    if pred != def_block && scratch.live_in[pred] != cur {
                        scratch.live_in[pred] = cur;
                        scratch.add(pred, cur, start[pred]);
                        walk.push(pred);
                    }
                }
            }
        }
        let first = seg_buf.len();
        for &b in &scratch.touched {
            let (lo, hi) = (scratch.lo[b], scratch.hi[b]);
            let has_points = lo <= hi;
            if scratch.term_def[b] {
                let t = term_pos(b);
                if has_points && hi >= t {
                    seg_buf.push((lo, hi.max(end[b])));
                } else {
                    if has_points {
                        seg_buf.push((lo, hi));
                    }
                    seg_buf.push((t, end[b]));
                }
            } else if has_points {
                seg_buf.push((lo, hi));
            }
        }
        seg_buf[first..].sort_unstable();
        values.push((v, first as u32, (seg_buf.len() - first) as u32));
    }
    values.sort_unstable_by_key(|&(v, first, _)| (seg_buf[first as usize].0, v.index()));

    LiveRanges {
        values,
        segments: seg_buf,
        moves,
    }
}

// ---- Coloring helpers -------------------------------------------------------

/// Items grouped by value index (compressed sparse rows).
struct Buckets<T> {
    first: Vec<u32>,
    items: Vec<T>,
}

impl<T: Copy> Buckets<T> {
    fn new(n: usize, entries: impl Iterator<Item = (Value, T)> + Clone) -> Self {
        let mut first = vec![0u32; n + 1];
        for (v, _) in entries.clone() {
            first[v.index() + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        let mut fill = first.clone();
        // Every slot is overwritten below; any entry serves as filler.
        let Some((_, filler)) = entries.clone().next() else {
            return Buckets {
                first,
                items: Vec::new(),
            };
        };
        let mut items = vec![filler; first[n] as usize];
        for (v, t) in entries {
            items[fill[v.index()] as usize] = t;
            fill[v.index()] += 1;
        }
        Buckets { first, items }
    }

    fn get(&self, v: Value) -> &[T] {
        let i = v.index();
        &self.items[self.first[i] as usize..self.first[i + 1] as usize]
    }
}

/// Does `v` overlap any segment of `slot`? Both are sorted and
/// non-overlapping; `v` is short, so binary-search `slot` for each segment.
fn overlaps(slot: &[Segment], v: &[Segment]) -> bool {
    v.iter().any(|&(lo, hi)| {
        // The first slot segment that ends after `lo`.
        let i = slot.partition_point(|s| s.1 <= lo);
        i < slot.len() && slot[i].0 < hi
    })
}

/// Merge `b` into the sorted, non-overlapping list `a`.
fn merge_into(a: &mut Vec<Segment>, b: &[Segment]) {
    for &seg in b {
        if a.last().is_none_or(|l| l.0 <= seg.0) {
            a.push(seg);
        } else {
            let i = a.partition_point(|s| s.0 <= seg.0);
            a.insert(i, seg);
        }
    }
}

/// A local and the segments of the values assigned to it.
struct Slot {
    ty: ValType,
    segs: Vec<Segment>,
    /// The end of the last segment.
    end: Pos,
}

/// The locals of one wasm type.
struct Class {
    ty: ValType,
    all: Vec<Local>,
    /// `(end, local)` min-heap. Entries whose end is stale are skipped.
    ends: BinaryHeap<Reverse<(Pos, Local)>>,
    /// Locals that the heap showed to end at or before some earlier start.
    free: Vec<Local>,
}

impl Class {
    /// A local whose segments all end at or before `lo`.
    fn take_free(&mut self, slots: &[Slot], lo: Pos) -> Option<Local> {
        while let Some(&Reverse((end, l))) = self.ends.peek() {
            if end > lo {
                break;
            }
            self.ends.pop();
            if slots[l.0 as usize].end == end {
                self.free.push(l);
            }
        }
        while let Some(l) = self.free.pop() {
            if slots[l.0 as usize].end <= lo {
                return Some(l);
            }
        }
        None
    }
}

fn class_of(classes: &mut Vec<Class>, ty: ValType) -> &mut Class {
    let i = match classes.iter().position(|c| c.ty == ty) {
        Some(i) => i,
        None => {
            classes.push(Class {
                ty,
                all: Vec::new(),
                ends: BinaryHeap::new(),
                free: Vec::new(),
            });
            classes.len() - 1
        }
    };
    &mut classes[i]
}

fn new_slot(slots: &mut Vec<Slot>, classes: &mut Vec<Class>, ty: ValType) -> Local {
    let l = Local(slots.len() as u32);
    slots.push(Slot {
        ty,
        segs: Vec::new(),
        end: Pos(0),
    });
    class_of(classes, ty).all.push(l);
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::tests::arb_function;
    use proptest::prelude::*;
    use std::collections::{BTreeSet, HashMap, HashSet};

    fn point_set(segs: &[Segment]) -> BTreeSet<u32> {
        let mut pts = BTreeSet::new();
        for &(lo, hi) in segs {
            assert!(lo.0 < hi.0, "invalid segment ({lo:?}, {hi:?})");
            for p in lo.0..hi.0 {
                assert!(
                    pts.insert(p),
                    "overlapping segments in single list: {segs:?}"
                );
            }
        }
        pts
    }

    /// Strategy generating a valid sorted list of non-overlapping half-open
    /// segments, including adjacent segments that touch at an endpoint.
    fn segments_strategy() -> impl Strategy<Value = Vec<Segment>> {
        prop::collection::vec((0u32..4, 1u32..6), 0..12).prop_map(|steps| {
            let mut out = Vec::new();
            let mut cur = 0u32;
            for (gap, len) in steps {
                let lo = cur + gap;
                let hi = lo + len;
                out.push((Pos(lo), Pos(hi)));
                cur = hi;
            }
            out
        })
    }

    #[derive(Clone, Debug)]
    enum ClassOp {
        NewSlot,
        TakeFree { delta_lo: u32, duration: u32 },
        ExtendExisting { slot_seed: usize, extra: u32 },
    }

    proptest! {
        #[test]
        fn buckets_matches_reference_model(
            n in 0usize..48,
            raw_entries in prop::collection::vec((0usize..48, any::<u32>()), 0..120),
        ) {
            let entries: Vec<(Value, u32)> = raw_entries
                .into_iter()
                .filter(|&(k, _)| k < n)
                .map(|(k, v)| (Value::new(k), v))
                .collect();

            let buckets = Buckets::new(n, entries.iter().copied());
            let mut expected = vec![Vec::new(); n];
            for &(v, item) in &entries {
                expected[v.index()].push(item);
            }
            for (i, exp) in expected.iter().enumerate() {
                prop_assert_eq!(buckets.get(Value::new(i)), exp.as_slice());
            }
        }

        #[test]
        fn overlaps_matches_point_set_oracle(
            a in segments_strategy(),
            b in segments_strategy(),
        ) {
            let pts_a = point_set(&a);
            let pts_b = point_set(&b);
            let expected = !pts_a.is_disjoint(&pts_b);
            prop_assert_eq!(overlaps(&a, &b), expected);
            prop_assert_eq!(overlaps(&b, &a), expected);
        }

        #[test]
        fn merge_into_preserves_sorted_non_overlapping_union(
            combined in segments_strategy(),
            choices in prop::collection::vec(any::<bool>(), 0..16),
        ) {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for (i, &seg) in combined.iter().enumerate() {
                if choices.get(i).copied().unwrap_or(false) {
                    a.push(seg);
                } else {
                    b.push(seg);
                }
            }
            prop_assert!(!overlaps(&a, &b));
            let expected_pts: BTreeSet<u32> =
                point_set(&a).union(&point_set(&b)).copied().collect();

            let mut merged = a.clone();
            merge_into(&mut merged, &b);
            prop_assert_eq!(point_set(&merged), expected_pts);
            prop_assert_eq!(merged, combined);
        }

        #[test]
        fn class_take_free_handles_stale_heap_entries(
            ops in prop::collection::vec(
                prop_oneof![
                    2 => Just(ClassOp::NewSlot),
                    5 => (0u32..5, 1u32..10).prop_map(|(delta_lo, duration)| ClassOp::TakeFree { delta_lo, duration }),
                    4 => (0usize..32, 1u32..12).prop_map(|(slot_seed, extra)| ClassOp::ExtendExisting { slot_seed, extra }),
                ],
                1..60,
            ),
        ) {
            let mut slots = Vec::new();
            let mut classes = Vec::new();
            let mut cur_lo = 0u32;

            for op in ops {
                match op {
                    ClassOp::NewSlot => {
                        let l = new_slot(&mut slots, &mut classes, ValType::I32);
                        slots[l.0 as usize].end = Pos(cur_lo + 1);
                        class_of(&mut classes, ValType::I32)
                            .ends
                            .push(Reverse((Pos(cur_lo + 1), l)));
                    }
                    ClassOp::TakeFree { delta_lo, duration } => {
                        cur_lo += delta_lo;
                        let lo = Pos(cur_lo);
                        let got = class_of(&mut classes, ValType::I32).take_free(&slots, lo);
                        if let Some(l) = got {
                            prop_assert!(
                                slots[l.0 as usize].end <= lo,
                                "take_free returned active slot {l:?} with end {:?} > {lo:?}",
                                slots[l.0 as usize].end
                            );
                            let new_end = Pos(cur_lo + duration);
                            slots[l.0 as usize].end = new_end;
                            class_of(&mut classes, ValType::I32)
                                .ends
                                .push(Reverse((new_end, l)));
                        } else {
                            // Verify no slot in `free` or `ends` actually has `end <= lo`.
                            let class = class_of(&mut classes, ValType::I32);
                            for &l in &class.free {
                                prop_assert!(slots[l.0 as usize].end > lo);
                            }
                            for &Reverse((_, l)) in class.ends.iter() {
                                prop_assert!(slots[l.0 as usize].end > lo);
                            }
                        }
                    }
                    ClassOp::ExtendExisting { slot_seed, extra } => {
                        if !slots.is_empty() {
                            let idx = slot_seed % slots.len();
                            let l = Local(idx as u32);
                            let new_end = slots[idx].end.max(Pos(cur_lo + extra));
                            slots[idx].end = new_end;
                            class_of(&mut classes, ValType::I32)
                                .ends
                                .push(Reverse((new_end, l)));
                        }
                    }
                }
            }
        }

        #[test]
        fn live_ranges_and_allocation_soundness(
            func in arb_function(),
            stack_values in any::<bool>(),
        ) {
            let info = FuncInfo::compute(&func).unwrap();
            let trees = crate::expr_trees::build(&func, &info, stack_values);
            let ranges = compute_live_ranges(&func, &info, &trees);

            // 1. Every value's segment list is non-empty, sorted, and non-overlapping.
            let mut seg_map: HashMap<Value, Vec<Segment>> = HashMap::new();
            for (v, segs) in ranges.iter() {
                prop_assert!(!segs.is_empty());
                for (i, &(lo, hi)) in segs.iter().enumerate() {
                    prop_assert!(lo < hi, "empty or inverted segment for {v}: ({lo:?}, {hi:?})");
                    if i > 0 {
                        prop_assert!(
                            segs[i - 1].1 <= lo,
                            "unsorted or overlapping segments for {v}: {segs:?}"
                        );
                    }
                }
                seg_map.insert(v, segs.to_vec());
            }

            // 2. Independent textbook iterative backwards dataflow liveness oracle.
            // Collect per-block local-value uses at each root position, and definitions.
            let is_local = |v: Value| needs_local(&func, &info, &trees, v);
            let mut live_in: HashMap<Block, HashSet<Value>> = HashMap::new();
            let mut live_out: HashMap<Block, HashSet<Value>> = HashMap::new();
            for &b in &info.rpo {
                live_in.insert(b, HashSet::new());
                live_out.insert(b, HashSet::new());
            }

            let mut changed = true;
            let mut ops = Vec::new();
            while changed {
                changed = false;
                for &b in info.rpo.iter().rev() {
                    let mut out_set = HashSet::new();
                    for &call in &info.dests[b] {
                        let succ = call.block(&func.dfg.value_lists);
                        let mut succ_in = live_in[&succ].clone();
                        // Block parameters of `succ` are defined on the incoming edge, not before.
                        for &p in func.dfg.block_params(succ) {
                            succ_in.remove(&p);
                        }
                        for (a, _) in info.live_arg_pairs(&func, call) {
                            if is_local(a) {
                                succ_in.insert(a);
                            }
                        }
                        out_set.extend(succ_in);
                    }
                    if out_set != live_out[&b] {
                        live_out.insert(b, out_set.clone());
                        changed = true;
                    }

                    // Walk instructions in `b` backwards.
                    let mut cur = out_set;
                    let term = func.layout.last_inst(b).unwrap();
                    // At terminator, remove block-arg uses that were added in `out_set`
                    // and re-apply terminator operands accurately.
                    for inst in func.layout.block_insts(b).rev() {
                        if trees.is_root(inst) {
                            for &r in func.dfg.inst_results(inst) {
                                cur.remove(&r);
                            }
                            // Collect all local operands in the tree rooted at `inst`.
                            for cand in func.layout.block_insts(b) {
                                if trees.root(cand) == Some(inst) {
                                    operands(&func, &info, cand, &mut ops);
                                    for o in &ops {
                                        if is_local(o.value) {
                                            cur.insert(o.value);
                                        }
                                    }
                                }
                            }
                        }
                        let _ = term;
                    }
                    if b == info.rpo[0] {
                        for &p in func.dfg.block_params(b) {
                            cur.remove(&p);
                        }
                    }
                    if cur != live_in[&b] {
                        live_in.insert(b, cur);
                        changed = true;
                    }
                }
            }

            // 3. Verify `allocate` with both `coalesce = false` and `coalesce = true`.
            let alloc_no_coal = allocate(&func, &info, &trees, false).unwrap();
            let alloc_coal = allocate(&func, &info, &trees, true).unwrap();

            for locals in [&alloc_no_coal, &alloc_coal] {
                // Every entry parameter i gets Local(i).
                for (i, &p) in func.dfg.block_params(info.rpo[0]).iter().enumerate() {
                    prop_assert_eq!(locals.local_of[p], Local(i as u32));
                }

                // Every pair of distinct values assigned the same Local must have non-overlapping segments
                // and identical ValType.
                let mut by_local: HashMap<Local, Vec<Value>> = HashMap::new();
                for (v, _) in ranges.iter() {
                    let l = locals.local_of[v];
                    prop_assert_ne!(l, Local::NONE);
                    let vt = wasm_type(func.dfg.value_type(v)).unwrap();
                    if l.0 >= locals.num_params {
                        prop_assert_eq!(locals.decls[(l.0 - locals.num_params) as usize], vt);
                    }
                    for &other in by_local.get(&l).unwrap_or(&Vec::new()) {
                        prop_assert!(
                            !overlaps(&seg_map[&v], &seg_map[&other]),
                            "values {v} and {other} share {l:?} but segments overlap: {:?} vs {:?}",
                            seg_map[&v],
                            seg_map[&other]
                        );
                    }
                    by_local.entry(l).or_default().push(v);
                }

                // Check against the textbook dataflow oracle at every instruction boundary:
                // no two simultaneously live values may share a Local.
                for &b in &info.rpo {
                    let mut cur = live_out[&b].clone();
                    for inst in func.layout.block_insts(b).rev() {
                        if !trees.is_root(inst) {
                            continue;
                        }
                        // All values in `cur` are live right after `inst` (except `inst` results if unused,
                        // which are not in `cur`).
                        let mut seen = HashMap::new();
                        for &v in &cur {
                            if let Some(&prev) = seen.get(&locals.local_of[v]) {
                                prop_assert_eq!(
                                    prev,
                                    v,
                                    "dataflow oracle found two values ({} and {}) live after {} in {} sharing {:?}",
                                    prev,
                                    v,
                                    inst,
                                    b,
                                    locals.local_of[v]
                                );
                            }
                            seen.insert(locals.local_of[v], v);
                        }
                        for &r in func.dfg.inst_results(inst) {
                            cur.remove(&r);
                        }
                        for cand in func.layout.block_insts(b) {
                            if trees.root(cand) == Some(inst) {
                                operands(&func, &info, cand, &mut ops);
                                for o in &ops {
                                    if is_local(o.value) {
                                        cur.insert(o.value);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 4. Coalescing never increases the number of required block-argument moves.
            let count_moves = |locals: &Locals| -> usize {
                info.rpo
                    .iter()
                    .flat_map(|&b| &info.dests[b])
                    .flat_map(|&call| info.live_arg_pairs(&func, call))
                    .filter(|&(a, p)| locals.move_needed(a, p))
                    .count()
            };
            prop_assert!(count_moves(&alloc_coal) <= count_moves(&alloc_no_coal));
        }
    }
}
