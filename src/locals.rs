//! Assignment of wasm locals to the values that are not consumed directly
//! from the operand stack.
//!
//! Every root instruction gets a position, in reverse postorder, with extra
//! positions for each block's start and end. A value's interval is the hull
//! of its def, its uses, and the start/end of every block where it is
//! live-in/live-out. Liveness is found by walking predecessors from each
//! use up to the def block. A hull is conservative for any block order:
//! two values live at the same point both contain that point.
//!
//! Intervals are half-open with respect to a single instruction: a value
//! whose last use is at position `p` may share a local with a value defined
//! at `p`, because an instruction reads all its operands before writing its
//! results. This also covers block-argument moves, which push every argument
//! before setting any parameter.
//!
//! Intervals are colored greedily in order of start position, separately
//! for each wasm value type.

use crate::analysis::{FuncInfo, is_remat};
use crate::stackify::{Operand, Stackified, operands};
use crate::{Config, Result, wasm_type};
use cranelift_codegen::entity::{EntityRef, SecondaryMap};
use cranelift_codegen::ir::{Block, Function, Inst, Value, ValueDef};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use wasm_encoder::ValType;

pub(crate) const NO_LOCAL: u32 = u32::MAX;

/// Coloring state for one wasm type: the active intervals as a min-heap of
/// (end, local), and the free locals.
type Class = (ValType, BinaryHeap<Reverse<(u32, u32)>>, Vec<u32>);

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

#[derive(Clone, Copy)]
struct Interval {
    lo: u32,
    hi: u32,
}

impl Interval {
    fn extend(&mut self, p: u32) {
        self.lo = self.lo.min(p);
        self.hi = self.hi.max(p);
    }
}

pub(crate) fn allocate(
    func: &Function,
    info: &FuncInfo,
    stack: &Stackified,
    _config: &Config,
) -> Result<Locals> {
    let entry = info.rpo[0];
    let num_params = func.dfg.block_params(entry).len() as u32;
    let mut local_of = SecondaryMap::with_default(NO_LOCAL);
    for (i, &p) in func.dfg.block_params(entry).iter().enumerate() {
        local_of[p] = i as u32;
    }

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

    let empty = Interval {
        lo: u32::MAX,
        hi: 0,
    };
    let mut iv: SecondaryMap<Value, Interval> = SecondaryMap::with_default(empty);
    let is_local = |v: Value| local_of[v] == NO_LOCAL && needs_local(func, info, stack, v);

    // Defs.
    for &b in &info.rpo {
        if b != entry {
            for &param in func.dfg.block_params(b) {
                if is_local(param) {
                    iv[param].extend(start[b]);
                    for &pred in &info.preds[b] {
                        let term = func.layout.last_inst(pred).unwrap();
                        iv[param].extend(pos[term]);
                        iv[param].extend(end[pred]);
                    }
                }
            }
        }
        for inst in func.layout.block_insts(b) {
            if stack.is_root(info, func, inst) {
                for &r in func.dfg.inst_results(inst) {
                    if is_local(r) {
                        iv[r].extend(pos[inst]);
                    }
                }
            }
        }
    }

    // Uses, and liveness walks.
    let mut ops: Vec<Operand> = Vec::new();
    let mut visited: SecondaryMap<Block, Option<Value>> = SecondaryMap::new();
    let mut walk: Vec<Block> = Vec::new();
    for &b in &info.rpo {
        for inst in func.layout.block_insts(b) {
            if !info.live_inst[inst] {
                continue;
            }
            let at = pos[stack.root(inst)];
            operands(func, info, inst, &mut ops);
            for o in &ops {
                let v = o.value;
                if !is_local(v) {
                    continue;
                }
                iv[v].extend(at);
                let def_block = match func.dfg.value_def(v) {
                    ValueDef::Result(d, _) => func.layout.inst_block(d).unwrap(),
                    ValueDef::Param(blk, _) => blk,
                    ValueDef::Union(..) => unreachable!(),
                };
                if def_block == b || visited[b] == Some(v) {
                    continue;
                }
                visited[b] = Some(v);
                iv[v].extend(start[b]);
                walk.push(b);
                while let Some(x) = walk.pop() {
                    for &pred in &info.preds[x] {
                        iv[v].extend(end[pred]);
                        if pred != def_block && visited[pred] != Some(v) {
                            visited[pred] = Some(v);
                            iv[v].extend(start[pred]);
                            walk.push(pred);
                        }
                    }
                }
            }
        }
    }

    // Coloring.
    let mut values: Vec<(Value, Interval, ValType)> = Vec::new();
    for (v, i) in iv.iter() {
        if i.lo <= i.hi {
            values.push((v, *i, wasm_type(func.dfg.value_type(v))?));
        }
    }
    values.sort_unstable_by_key(|(v, i, _)| (i.lo, v.index()));

    let mut decls: Vec<ValType> = Vec::new();
    // Per type: active (end, local) min-heap and free list.
    let mut classes: Vec<Class> = Vec::new();
    for (v, i, ty) in values {
        let class = match classes.iter().position(|c| c.0 == ty) {
            Some(c) => c,
            None => {
                classes.push((ty, BinaryHeap::new(), Vec::new()));
                classes.len() - 1
            }
        };
        let (_, active, free) = &mut classes[class];
        while let Some(&Reverse((hi, l))) = active.peek() {
            if hi > i.lo {
                break;
            }
            active.pop();
            free.push(l);
        }
        let l = free.pop().unwrap_or_else(|| {
            decls.push(ty);
            num_params + decls.len() as u32 - 1
        });
        active.push(Reverse((i.hi, l)));
        local_of[v] = l;
    }

    Ok(Locals {
        local_of,
        decls,
        num_params,
    })
}
