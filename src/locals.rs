//! Assignment of wasm locals to the values that are not consumed directly
//! from the operand stack, given their [live ranges](crate::liveness).
//!
//! Values are colored greedily in order of their first position, separately
//! for each wasm type. A value first tries the locals of its *move partners*
//! (the arguments and parameters of block-argument moves it takes part in),
//! so that the move disappears, then a local that is free from its start
//! on, then a bounded number of other locals whose segments do not overlap
//! its own.

use crate::analysis::FuncInfo;
use crate::liveness::{Buckets, LiveRanges, Segment};
use crate::{Result, wasm_type};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::{Function, Value};
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

    /// Must the block-argument move `param := arg` be emitted?
    ///
    /// Not when both are in the same local. That is safe because a branch
    /// pushes all its (needed) arguments before setting any parameter, and
    /// because two values in one local never overlap: no other argument of
    /// the same branch can be in that local, and no other parameter of the
    /// target either (they are all defined at the same terminator).
    pub fn move_needed(&self, arg: Value, param: Value) -> bool {
        let la = self.local_of[arg];
        la == NO_LOCAL || la != self.local_of[param]
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

pub(crate) fn allocate(
    func: &Function,
    info: &FuncInfo,
    ranges: &LiveRanges,
    coalesce: bool,
) -> Result<Locals> {
    let entry_params = func.dfg.block_params(info.rpo[0]);
    let num_params = entry_params.len() as u32;
    let mut local_of = SecondaryMap::with_default(NO_LOCAL);
    let moves = if coalesce { &ranges.moves[..] } else { &[] };
    let partners = Buckets::new(
        func.dfg.num_values(),
        moves.iter().flat_map(|&(a, p)| [(a, p), (p, a)]),
    );

    // Coloring.
    let mut slots: Vec<Slot> = Vec::new();
    let mut classes: Vec<Class> = Vec::new();
    for &param in entry_params {
        let ty = wasm_type(func.dfg.value_type(param))?;
        new_slot(&mut slots, &mut classes, ty);
    }

    for (v, segs) in ranges.iter() {
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
