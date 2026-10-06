//! Structured control flow, following Norman Ramsey, "Beyond Relooper:
//! recursive translation of unstructured control flow to structured control
//! flow" (ICFP 2022).
//!
//! The translation is driven by the dominator tree. A block `x` is emitted
//! as:
//!
//! - wrapped in a `loop` if it is the target of a back edge;
//! - with one nested `block` for each dominator-tree child of `x` that is a
//!   *merge node* (has two or more forward in-edges). The children are
//!   placed after the `end` of their block in reverse postorder, so a forward
//!   branch to a merge node is a `br` out of the enclosing block;
//! - other children, which have a single in-edge, are emitted inline at
//!   their unique branch.
//!
//! All values crossing block boundaries are in locals, so every wasm `block`,
//! `loop` and `if` has type `[] -> []`.
//!
//! The recursion in the paper is turned into an explicit task stack
//! ([`Task`]) so that long chains of blocks cannot overflow the native stack.
//!
//! If the CFG is irreducible, the whole function is instead emitted as a
//! loop around a `br_table` that dispatches on a label local.

use crate::analysis::FuncInfo;
use crate::emit::{Emitter, Label, Task};
use crate::locals::Locals;
use crate::stackify::Stackified;
use crate::{CompiledFunction, Config, Env, Result, unsupported, wasm_signature};
use cranelift_codegen::ir::{Block, BlockCall, Function, Opcode, types};
use std::borrow::Cow;
use wasm_encoder::{BlockType, Instruction as I, ValType};

pub(crate) fn emit_function(
    func: &Function,
    config: &Config,
    env: &mut dyn Env,
    info: &FuncInfo,
    stack: &Stackified,
    locals: Locals,
) -> Result<CompiledFunction> {
    let (params, results) = wasm_signature(&func.signature)?;
    let mut e = Emitter::new(func, config, env, info, stack, locals);

    if !func.dynamic_stack_slots.is_empty() {
        return unsupported("dynamic stack slots");
    }
    if !func.sized_stack_slots.is_empty() {
        let Some(sp) = config.stack_pointer_global else {
            return unsupported("stack slots without Config::stack_pointer_global");
        };
        let mut size: u32 = 0;
        for (slot, data) in func.sized_stack_slots.iter() {
            if data.align_shift > 4 {
                return unsupported("stack slot alignment above 16");
            }
            let align = 1u32 << data.align_shift;
            size = size.next_multiple_of(align);
            e.slot_offset[slot] = size;
            size += data.size;
        }
        let size = size.next_multiple_of(16);
        let fp = e.new_local(config.ptr_ty());
        e.frame = Some((fp, size));
        e.ins(I::GlobalGet(sp));
        if config.memory64 {
            e.ins(I::I64Const(size as i64));
            e.ins(I::I64Sub);
        } else {
            e.ins(I::I32Const(size as i32));
            e.ins(I::I32Sub);
        }
        e.ins(I::LocalTee(fp));
        e.ins(I::GlobalSet(sp));
    }

    // Callers may pass garbage in the upper bits of narrow parameters.
    let entry = info.rpo[0];
    for (i, &p) in func.dfg.block_params(entry).iter().enumerate() {
        let bits = match func.dfg.value_type(p) {
            types::I8 => 0xff,
            types::I16 => 0xffff,
            _ => continue,
        };
        if info.live_val[p] {
            e.ins(I::LocalGet(i as u32));
            e.ins(I::I32Const(bits));
            e.ins(I::I32And);
            e.ins(I::LocalSet(i as u32));
        }
    }

    if info.reducible {
        e.tasks.push(Task::Tree(entry));
        e.run_tasks()?;
    } else {
        e.dispatch()?;
    }
    e.ins(I::Unreachable);
    e.ins(I::End);

    let (body, locals, extra) = e.finish();
    let mut decls: Vec<(u32, ValType)> = Vec::new();
    for ty in locals.decls.iter().chain(extra.iter()) {
        match decls.last_mut() {
            Some((n, t)) if t == ty => *n += 1,
            _ => decls.push((1, *ty)),
        }
    }
    Ok(CompiledFunction {
        params,
        results,
        locals: decls,
        body,
    })
}

impl Emitter<'_> {
    /// Run tasks until the stack is empty.
    fn run_tasks(&mut self) -> Result<()> {
        while let Some(task) = self.tasks.pop() {
            match task {
                Task::Tree(x) => self.tree(x)?,
                Task::Branch { src, call } => self.branch(src, call)?,
                Task::Else => self.ins(I::Else),
                Task::End => {
                    self.labels.pop();
                    self.ins(I::End);
                }
            }
        }
        Ok(())
    }

    /// Emit the dominator subtree rooted at `x`:
    ///
    /// ```text
    /// loop                 ;; if x is a loop header
    ///   block              ;; merge child y1 (highest RPO number)
    ///     ...
    ///       block          ;; merge child yk (lowest RPO number)
    ///         <x>
    ///       end
    ///       <tree yk>
    ///     ...
    ///   end
    ///   <tree y1>
    /// end
    /// ```
    fn tree(&mut self, x: Block) -> Result<()> {
        if self.info.loop_header[x] {
            self.ins(I::Loop(BlockType::Empty));
            self.labels.push(Label::Loop(x));
            self.tasks.push(Task::End);
        }
        let mut merges: Vec<Block> = self
            .info
            .domtree
            .children(x)
            .filter(|&c| self.info.is_merge(c))
            .collect();
        merges.sort_unstable_by_key(|&b| std::cmp::Reverse(self.info.rpo_num[b]));
        for &y in &merges {
            self.ins(I::Block(BlockType::Empty));
            self.labels.push(Label::BlockFollowedBy(y));
            self.tasks.push(Task::Tree(y));
            self.tasks.push(Task::End);
        }
        self.block_body(x)?;
        self.terminator(x)
    }

    fn dispatch(&mut self) -> Result<()> {
        let label = self.new_local(ValType::I32);
        self.dispatch_label = Some(label);
        let n = self.info.rpo.len();
        self.ins(I::Loop(BlockType::Empty));
        self.labels.push(Label::Dispatch);
        for _ in 0..n {
            self.ins(I::Block(BlockType::Empty));
            self.labels.push(Label::Other);
        }
        self.ins(I::LocalGet(label));
        let targets: Vec<u32> = (0..n as u32).collect();
        self.ins(I::BrTable(Cow::Owned(targets), 0));
        for i in 0..n {
            self.ins(I::End);
            self.labels.pop();
            let b = self.info.rpo[i];
            self.block_body(b)?;
            self.terminator(b)?;
            self.run_tasks()?;
        }
        self.labels.pop();
        self.ins(I::End);
        Ok(())
    }

    fn block_body(&mut self, b: Block) -> Result<()> {
        let func = self.func;
        let term = func.layout.last_inst(b).unwrap();
        for inst in func.layout.block_insts(b) {
            if inst != term && self.stack.is_root(self.info, func, inst) {
                self.root(inst)?;
            }
        }
        Ok(())
    }

    /// Emit the start of a terminator and push the tasks for its successors.
    ///
    /// Branch operands are pushed here directly rather than through
    /// `Emitter::inst`; see `stackify::operands` for why the order is safe.
    fn terminator(&mut self, b: Block) -> Result<()> {
        let func = self.func;
        let info = self.info;
        let term = func.layout.last_inst(b).unwrap();
        let dests = &info.dests[b];
        match func.dfg.insts[term].opcode() {
            Opcode::Jump => self.tasks.push(Task::Branch {
                src: b,
                call: dests[0],
            }),
            Opcode::Brif => {
                let cond = func.dfg.inst_args(term)[0];
                self.value(cond)?;
                if dests.len() == 1 {
                    self.ins(I::Drop);
                    self.tasks.push(Task::Branch {
                        src: b,
                        call: dests[0],
                    });
                    return Ok(());
                }
                let (mut then, mut els) = (dests[0], dests[1]);
                if func.dfg.value_type(cond) == types::I64 {
                    self.ins(I::I64Eqz);
                    std::mem::swap(&mut then, &mut els);
                }
                self.ins(I::If(BlockType::Empty));
                self.labels.push(Label::If);
                self.tasks.push(Task::End);
                self.tasks.push(Task::Branch { src: b, call: els });
                self.tasks.push(Task::Else);
                self.tasks.push(Task::Branch { src: b, call: then });
            }
            Opcode::BrTable => self.br_table(b, term)?,
            _ => self.root(term)?,
        }
        Ok(())
    }

    fn br_table(&mut self, b: Block, term: cranelift_codegen::ir::Inst) -> Result<()> {
        let func = self.func;
        let dfg = &func.dfg;
        let dests = &self.info.dests[b];
        let index = dfg.inst_args(term)[0];
        let ity = dfg.value_type(index);
        if dests.len() == 1 {
            self.value(index)?;
            self.ins(I::Drop);
            self.tasks.push(Task::Branch {
                src: b,
                call: dests[0],
            });
            return Ok(());
        }
        let cranelift_codegen::ir::InstructionData::BranchTable { table, .. } = dfg.insts[term]
        else {
            unreachable!()
        };
        let jt = &dfg.jump_tables[table];
        let pos = |call: &BlockCall| {
            dests
                .iter()
                .position(|d| crate::analysis::same_call(func, d, call))
                .unwrap() as u32
        };
        let targets: Vec<u32> = jt.as_slice().iter().map(pos).collect();
        let default = pos(&jt.default_block());

        // The blocks are `[] -> []`, so the index is computed inside them.
        let n = dests.len();
        for _ in 0..n {
            self.ins(I::Block(BlockType::Empty));
            self.labels.push(Label::Other);
        }
        self.value(index)?;
        if ity == types::I64 {
            // Clamp to the table length so that wrapping cannot alias an
            // in-range entry.
            let s = self.new_local(ValType::I64);
            let len = targets.len() as i64;
            self.ins(I::LocalTee(s));
            self.ins(I::I64Const(len));
            self.ins(I::LocalGet(s));
            self.ins(I::I64Const(len));
            self.ins(I::I64LtU);
            self.ins(I::Select);
            self.ins(I::I32WrapI64);
        }
        self.ins(I::BrTable(Cow::Owned(targets), default));
        // After the i-th `end` comes the code for the i-th destination.
        for &call in dests.iter().rev() {
            self.tasks.push(Task::Branch { src: b, call });
            self.tasks.push(Task::End);
        }
        Ok(())
    }

    /// Move the block arguments into the target's parameters and transfer
    /// control.
    fn branch(&mut self, src: Block, call: BlockCall) -> Result<()> {
        let func = self.func;
        let info = self.info;
        let moves: Vec<_> = info
            .live_arg_pairs(func, call)
            .filter(|&(a, p)| self.locals.move_needed(a, p))
            .collect();
        for &(a, _) in &moves {
            self.value(a)?;
        }
        for &(_, p) in moves.iter().rev() {
            self.ins(I::LocalSet(self.local(p)));
        }

        let t = call.block(&func.dfg.value_lists);
        if let Some(label) = self.dispatch_label {
            self.ins(I::I32Const(info.rpo_num[t] as i32));
            self.ins(I::LocalSet(label));
            self.ins(I::Br(self.br_depth(Label::Dispatch)));
        } else if info.is_backward(src, t) {
            self.ins(I::Br(self.br_depth(Label::Loop(t))));
        } else if info.is_merge(t) {
            self.ins(I::Br(self.br_depth(Label::BlockFollowedBy(t))));
        } else {
            self.tasks.push(Task::Tree(t));
        }
        Ok(())
    }
}
