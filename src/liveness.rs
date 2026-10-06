//! Live ranges of the values that need a wasm local.
//!
//! Every root instruction gets a position, in reverse postorder, with extra
//! positions for each block's start and end. A value's live range is a list
//! of segments, at most two per block, found by walking predecessors from
//! each use up to the def.
//!
//! Two conventions make block-argument moves cheap to coalesce, and
//! [`crate::locals::Locals::move_needed`] relies on them:
//!
//! - A block parameter is defined by the terminator of each predecessor:
//!   there it is live from the terminator to the block end, not before.
//! - Segments are half-open with respect to a single instruction: a value
//!   whose last use is at position `p` does not conflict with a value
//!   defined at `p`. An instruction reads all its operands before writing
//!   its results, and block-argument moves push every argument before
//!   setting any parameter.
//!
//! So a value whose last use is a branch argument does not conflict with
//! the parameter it is passed to.

use crate::analysis::{FuncInfo, is_remat};
use crate::stackify::{Operand, Stackified, operands};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::{Block, Function, Inst, Value, ValueDef};

/// A half-open range of positions.
pub(crate) type Segment = (u32, u32);

pub(crate) struct LiveRanges {
    /// (value, first segment, number of segments), sorted by the start of
    /// the value's first segment.
    values: Vec<(Value, u32, u32)>,
    segments: Vec<Segment>,
    /// Block-argument moves `(argument, parameter)` where both need a local.
    pub moves: Vec<(Value, Value)>,
}

impl LiveRanges {
    /// The values that need a local, with their sorted, non-overlapping
    /// segments, in order of their first position.
    pub fn iter(&self) -> impl Iterator<Item = (Value, &[Segment])> {
        self.values
            .iter()
            .map(|&(v, first, len)| (v, &self.segments[first as usize..(first + len) as usize]))
    }
}

/// Does `v` need a local (as opposed to being dead, rematerialized, or
/// passed on the stack)?
pub(crate) fn needs_local(func: &Function, info: &FuncInfo, stack: &Stackified, v: Value) -> bool {
    if !info.live_val[v] || info.use_count[v] == 0 {
        return false;
    }
    match func.dfg.value_def(v) {
        ValueDef::Result(d, _) => !stack.inlined[d] && !is_remat(func, d),
        ValueDef::Param(..) => true,
        ValueDef::Union(..) => unreachable!(),
    }
}

/// Items grouped by value index (compressed sparse rows).
pub(crate) struct Buckets<T> {
    first: Vec<u32>,
    items: Vec<T>,
}

impl<T: Copy> Buckets<T> {
    pub fn new(n: usize, entries: impl Iterator<Item = (Value, T)> + Clone) -> Self {
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

    pub fn get(&self, i: usize) -> &[T] {
        &self.items[self.first[i] as usize..self.first[i + 1] as usize]
    }
}

/// Per-block scratch state for building one value's segments.
struct Scratch {
    stamp: SecondaryMap<Block, u32>,
    live_in: SecondaryMap<Block, u32>,
    lo: SecondaryMap<Block, u32>,
    hi: SecondaryMap<Block, u32>,
    term_def: SecondaryMap<Block, bool>,
    touched: Vec<Block>,
}

impl Scratch {
    fn touch(&mut self, b: Block, cur: u32) {
        if self.stamp[b] != cur {
            self.stamp[b] = cur;
            self.lo[b] = u32::MAX;
            self.hi[b] = 0;
            self.term_def[b] = false;
            self.touched.push(b);
        }
    }

    fn add(&mut self, b: Block, cur: u32, p: u32) {
        self.touch(b, cur);
        self.lo[b] = self.lo[b].min(p);
        self.hi[b] = self.hi[b].max(p);
    }
}

/// Compute the live ranges of every value that [needs a local](needs_local).
pub(crate) fn compute(func: &Function, info: &FuncInfo, stack: &Stackified) -> LiveRanges {
    let entry = info.rpo[0];
    // Positions.
    let mut pos: SecondaryMap<Inst, u32> = SecondaryMap::new();
    let mut start: SecondaryMap<Block, u32> = SecondaryMap::new();
    let mut end: SecondaryMap<Block, u32> = SecondaryMap::new();
    let mut p = 0;
    for &b in &info.rpo {
        start[b] = p;
        p += 1;
        for inst in func.layout.block_insts(b) {
            if stack.is_root(info, func, inst) {
                pos[inst] = p;
                p += 1;
            }
        }
        end[b] = p;
        p += 1;
    }
    let term_pos = |b: Block| pos[func.layout.last_inst(b).unwrap()];

    // Uses of each value that needs a local, bucketed by value with a
    // counting sort, and block-argument moves.
    let is_local = |v: Value| needs_local(func, info, stack, v);
    let mut use_list: Vec<(Value, Block, u32)> = Vec::new();
    let mut moves: Vec<(Value, Value)> = Vec::new();
    let mut ops: Vec<Operand> = Vec::new();
    for &b in &info.rpo {
        for inst in func.layout.block_insts(b) {
            if !info.live_inst[inst] {
                continue;
            }
            let root = stack.root(inst);
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

    // Segments.
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
        let v_uses = uses.get(vi);
        if v_uses.is_empty() {
            continue;
        }
        let v = Value::new(vi);
        let cur = vi as u32 + 1;
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
