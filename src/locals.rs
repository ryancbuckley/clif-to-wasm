//! Assignment of wasm locals to the values that are not consumed directly
//! from the operand stack.
//!
//! Every root instruction gets a position, in reverse postorder, with extra
//! positions for each block's start and end. A value's live range is a list
//! of segments, at most two per block, found by walking predecessors from
//! each use up to the def. A block parameter is defined by the terminator of
//! each predecessor, so there it is live from that terminator to the block
//! end; this lets a value whose last use is a branch argument share a local
//! with the parameter it is passed to.
//!
//! Segments are half-open with respect to a single instruction: a value
//! whose last use is at position `p` may share a local with a value defined
//! at `p`, because an instruction reads all its operands before writing its
//! results. Block-argument moves push every argument before setting any
//! parameter, so the same holds for them.
//!
//! Values are colored greedily in order of their first position. A value
//! first tries the locals of its *move partners* (branch arguments and the
//! parameters they are passed to), so that the move disappears, then a local
//! that is free from its start on, then (in small functions) any local whose
//! segments do not overlap its own.

use crate::analysis::{FuncInfo, is_remat};
use crate::stackify::{Operand, Stackified, operands};
use crate::{Config, Result, wasm_type};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::{Block, Function, Inst, Value, ValueDef};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use wasm_encoder::ValType;

pub(crate) const NO_LOCAL: u32 = u32::MAX;

/// How many existing locals a value may be checked against for a
/// non-overlapping slot, beyond its move partners and the free list.
const SCAN_LIMIT: usize = 32;

pub(crate) struct Locals {
    /// The local holding each value, or `NO_LOCAL`.
    pub local_of: SecondaryMap<Value, u32>,
    /// Types of the declared locals, starting after the parameters.
    pub decls: Vec<ValType>,
    pub num_params: u32,
}

impl Locals {
    pub fn next_index(&self) -> u32 {
        self.num_params + self.decls.len() as u32
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

type Segment = (u32, u32);

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

    fn get(&self, i: usize) -> &[T] {
        &self.items[self.first[i] as usize..self.first[i + 1] as usize]
    }
}

/// A local and the segments of the values assigned to it.
struct Slot {
    ty: ValType,
    segs: Vec<Segment>,
    /// The end of the last segment.
    end: u32,
}

/// The locals of one wasm type.
struct Class {
    ty: ValType,
    all: Vec<u32>,
    /// (end, local) min-heap. Entries whose end is stale are skipped.
    ends: BinaryHeap<Reverse<(u32, u32)>>,
    /// Locals that the heap showed to end at or before some earlier start.
    free: Vec<u32>,
}

impl Class {
    /// A local whose segments all end at or before `lo`.
    fn take_free(&mut self, slots: &[Slot], lo: u32) -> Option<u32> {
        while let Some(&Reverse((end, l))) = self.ends.peek() {
            if end > lo {
                break;
            }
            self.ends.pop();
            if slots[l as usize].end == end {
                self.free.push(l);
            }
        }
        while let Some(l) = self.free.pop() {
            if slots[l as usize].end <= lo {
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

fn new_slot(slots: &mut Vec<Slot>, classes: &mut Vec<Class>, ty: ValType) -> u32 {
    let l = slots.len() as u32;
    slots.push(Slot {
        ty,
        segs: Vec::new(),
        end: 0,
    });
    class_of(classes, ty).all.push(l);
    l
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

pub(crate) fn allocate(
    func: &Function,
    info: &FuncInfo,
    stack: &Stackified,
    _config: &Config,
) -> Result<Locals> {
    let entry = info.rpo[0];
    let entry_params = func.dfg.block_params(entry);
    let num_params = entry_params.len() as u32;
    let mut local_of = SecondaryMap::with_default(NO_LOCAL);

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

    // Uses of each value that needs a local, and move partners, bucketed by
    // value with a counting sort.
    let is_local = |v: Value| needs_local(func, info, stack, v);
    let mut use_list: Vec<(Value, Block, u32)> = Vec::new();
    let mut pair_list: Vec<(Value, Value)> = Vec::new();
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
                    pair_list.push((a, param));
                    pair_list.push((param, a));
                }
            }
        }
    }
    let num_values = func.dfg.num_values();
    let uses = Buckets::new(num_values, use_list.iter().map(|&(v, b, p)| (v, (b, p))));
    let partners = Buckets::new(num_values, pair_list.iter().copied());

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
    // (value, first segment index, number of segments)
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

    // Coloring.
    let mut slots: Vec<Slot> = Vec::new();
    let mut classes: Vec<Class> = Vec::new();
    for &param in entry_params {
        let ty = wasm_type(func.dfg.value_type(param))?;
        new_slot(&mut slots, &mut classes, ty);
    }

    for &(v, first, len) in &values {
        let segs = &seg_buf[first as usize..(first + len) as usize];
        let ty = wasm_type(func.dfg.value_type(v))?;
        let (lo, hi) = (segs[0].0, segs.last().unwrap().1);
        let fits = |slots: &[Slot], l: u32| {
            let s = &slots[l as usize];
            s.ty == ty && !overlaps(&s.segs, segs)
        };

        let chosen = if let Some(i) = entry_params.iter().position(|&p| p == v) {
            i as u32
        } else if let Some(l) = partners
            .get(v.index())
            .iter()
            .map(|w| local_of[*w])
            .find(|&l| l != NO_LOCAL && fits(&slots, l))
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

        let slot = &mut slots[chosen as usize];
        merge_into(&mut slot.segs, segs);
        slot.end = slot.end.max(hi);
        let end = slot.end;
        local_of[v] = chosen;
        class_of(&mut classes, ty).ends.push(Reverse((end, chosen)));
    }

    // Unused entry parameters keep their (pinned) local.
    for (i, &param) in entry_params.iter().enumerate() {
        local_of[param] = i as u32;
    }
    let decls = slots[num_params as usize..].iter().map(|s| s.ty).collect();
    Ok(Locals {
        local_of,
        decls,
        num_params,
    })
}
