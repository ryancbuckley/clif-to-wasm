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

use crate::analysis::{FuncInfo, same_call};
use crate::emit::Emitter;
use crate::expr_trees::ExprTrees;
use crate::locals::{Local, Locals};
use crate::{CompiledFunction, Config, Env, Result, unsupported, wasm_signature, wasm_type};
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, BlockCall, ExceptionTableItem, ExceptionTag, Function, Inst,
    InstructionData, Opcode, Signature, types,
};
use std::borrow::Cow;
use wasm_encoder::{BlockType, Catch, Instruction as I, ValType};

/// An entry on the wasm label stack, used to compute branch depths.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Label {
    If,
    Loop(Block),
    BlockFollowedBy(Block),
    Dispatch,
    Other,
}

/// Pending structured-emission work.
#[derive(Clone, Copy, Debug)]
enum Task {
    /// Emit a dominator subtree.
    Tree(Block),
    /// Move block arguments and transfer control along an edge.
    Branch { src: Block, call: BlockCall },
    Else,
    /// Close the innermost label.
    End,
    /// Like `Branch`, along an exception edge of a `try_call` whose wasm
    /// label left the payload on the stack (if `payload`).
    Catch {
        src: Block,
        call: BlockCall,
        payload: bool,
    },
}

struct Structurer<'a> {
    e: Emitter<'a>,
    labels: Vec<Label>,
    tasks: Vec<Task>,
    /// The label local, when emitting a dispatch loop for an irreducible CFG.
    dispatch_label: Option<Local>,
    /// Locals holding the call results and the exception payload of the
    /// `try_call` ending each block, which its edges pass as `ret0`/`exn0`.
    try_results: SecondaryMap<Block, Vec<Local>>,
    try_payload: SecondaryMap<Block, Vec<Local>>,
}

pub(crate) fn emit_function(
    func: &Function,
    config: &Config,
    env: &mut dyn Env,
    info: &FuncInfo,
    trees: &ExprTrees,
    locals: Locals,
) -> Result<CompiledFunction> {
    let e = Emitter::new(func, config, env, info, trees, locals)?;
    let mut s = Structurer {
        e,
        labels: Vec::new(),
        tasks: Vec::new(),
        dispatch_label: None,
        try_results: SecondaryMap::new(),
        try_payload: SecondaryMap::new(),
    };
    let entry = info.rpo[0];
    if info.reducible {
        s.tasks.push(Task::Tree(entry));
        s.run_tasks()?;
    } else {
        s.dispatch()?;
    }
    s.e.finish()
}

impl Structurer<'_> {
    fn br_depth(&self, label: Label) -> u32 {
        let i = self
            .labels
            .iter()
            .rposition(|&l| l == label)
            .expect("branch target in scope");
        (self.labels.len() - 1 - i) as u32
    }

    /// Run tasks until the stack is empty.
    fn run_tasks(&mut self) -> Result<()> {
        while let Some(task) = self.tasks.pop() {
            match task {
                Task::Tree(x) => self.tree(x)?,
                Task::Branch { src, call } => self.branch(src, call)?,
                Task::Catch { src, call, payload } => {
                    if payload {
                        for i in (0..self.try_payload[src].len()).rev() {
                            let l = self.try_payload[src][i];
                            self.e.local_set(l);
                        }
                    }
                    self.branch(src, call)?;
                }
                Task::Else => self.e.ins(I::Else),
                Task::End => {
                    self.labels.pop();
                    self.e.ins(I::End);
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
        let info = self.e.info;
        if info.loop_header[x] {
            self.e.ins(I::Loop(BlockType::Empty));
            self.labels.push(Label::Loop(x));
            self.tasks.push(Task::End);
        }
        let mut merges: Vec<Block> = info
            .domtree
            .children(x)
            .filter(|&c| info.is_merge(c))
            .collect();
        merges.sort_unstable_by_key(|&b| std::cmp::Reverse(info.rpo_num[b]));
        for &y in &merges {
            self.e.ins(I::Block(BlockType::Empty));
            self.labels.push(Label::BlockFollowedBy(y));
            self.tasks.push(Task::Tree(y));
            self.tasks.push(Task::End);
        }
        self.block_body(x)?;
        self.terminator(x)
    }

    fn dispatch(&mut self) -> Result<()> {
        let label = self.e.new_local(ValType::I32);
        self.dispatch_label = Some(label);
        let n = self.e.info.rpo.len();
        self.e.ins(I::Loop(BlockType::Empty));
        self.labels.push(Label::Dispatch);
        for _ in 0..n {
            self.e.ins(I::Block(BlockType::Empty));
            self.labels.push(Label::Other);
        }
        self.e.local_get(label);
        let targets: Vec<u32> = (0..n as u32).collect();
        self.e.ins(I::BrTable(Cow::Owned(targets), 0));
        for i in 0..n {
            self.e.ins(I::End);
            self.labels.pop();
            let b = self.e.info.rpo[i];
            self.block_body(b)?;
            self.terminator(b)?;
            self.run_tasks()?;
        }
        self.labels.pop();
        self.e.ins(I::End);
        Ok(())
    }

    fn block_body(&mut self, b: Block) -> Result<()> {
        let func = self.e.func;
        let term = func.layout.last_inst(b).unwrap();
        for inst in func.layout.block_insts(b) {
            if inst != term && self.e.trees.is_root(inst) {
                self.e.root(inst)?;
            }
        }
        Ok(())
    }

    /// Emit the start of a terminator and push the tasks for its successors.
    ///
    /// Branch operands are pushed here directly rather than through
    /// `Emitter::inst`; see `expr_trees::operands` for why the order is safe.
    fn terminator(&mut self, b: Block) -> Result<()> {
        let func = self.e.func;
        let info = self.e.info;
        let term = func.layout.last_inst(b).unwrap();
        let dests = &info.dests[b];
        match func.dfg.insts[term].opcode() {
            Opcode::Jump => self.tasks.push(Task::Branch {
                src: b,
                call: dests[0],
            }),
            Opcode::Brif => {
                let cond = func.dfg.inst_args(term)[0];
                self.e.value(cond)?;
                if dests.len() == 1 {
                    self.e.ins(I::Drop);
                    self.tasks.push(Task::Branch {
                        src: b,
                        call: dests[0],
                    });
                    return Ok(());
                }
                let (mut then, mut els) = (dests[0], dests[1]);
                if func.dfg.value_type(cond) == types::I64 {
                    self.e.ins(I::I64Eqz);
                    std::mem::swap(&mut then, &mut els);
                }
                self.e.ins(I::If(BlockType::Empty));
                self.labels.push(Label::If);
                self.tasks.push(Task::End);
                self.tasks.push(Task::Branch { src: b, call: els });
                self.tasks.push(Task::Else);
                self.tasks.push(Task::Branch { src: b, call: then });
            }
            Opcode::BrTable => self.br_table(b, term)?,
            Opcode::TryCall | Opcode::TryCallIndirect => self.try_call(b, term)?,
            _ => self.e.root(term)?,
        }
        Ok(())
    }

    fn br_table(&mut self, b: Block, term: Inst) -> Result<()> {
        let func = self.e.func;
        let dfg = &func.dfg;
        let dests = &self.e.info.dests[b];
        let index = dfg.inst_args(term)[0];
        let ity = dfg.value_type(index);
        if dests.len() == 1 {
            self.e.value(index)?;
            self.e.ins(I::Drop);
            self.tasks.push(Task::Branch {
                src: b,
                call: dests[0],
            });
            return Ok(());
        }
        let InstructionData::BranchTable { table, .. } = dfg.insts[term] else {
            unreachable!()
        };
        let jt = &dfg.jump_tables[table];
        let pos = |call: &BlockCall| {
            dests
                .iter()
                .position(|d| same_call(func, d, call))
                .unwrap() as u32
        };
        let targets: Vec<u32> = jt.as_slice().iter().map(pos).collect();
        let default = pos(&jt.default_block());

        // The blocks are `[] -> []`, so the index is computed inside them.
        let n = dests.len();
        for _ in 0..n {
            self.e.ins(I::Block(BlockType::Empty));
            self.labels.push(Label::Other);
        }
        self.e.value(index)?;
        if ity == types::I64 {
            // Clamp to the table length so that wrapping cannot alias an
            // in-range entry.
            let mark = self.e.scratch_mark();
            let s = self.e.scratch(ValType::I64);
            let len = targets.len() as i64;
            self.e.local_tee(s);
            self.e.ins(I::I64Const(len));
            self.e.local_get(s);
            self.e.ins(I::I64Const(len));
            self.e.ins(I::I64LtU);
            self.e.ins(I::Select);
            self.e.ins(I::I32WrapI64);
            self.e.free_scratches(mark);
        }
        self.e.ins(I::BrTable(Cow::Owned(targets), default));
        // After the i-th `end` comes the code for the i-th destination.
        for &call in dests.iter().rev() {
            self.tasks.push(Task::Branch { src: b, call });
            self.tasks.push(Task::End);
        }
        Ok(())
    }

    /// A `try_call`. The call runs inside a `try_table` whose catch clauses
    /// branch out to one wasm block per exception-table entry; the call's
    /// results and the exception payload go to locals, from which the edges
    /// pass them on as `ret`/`exn` arguments:
    ///
    /// ```text
    /// block                        ;; entry n-1
    ///   ...
    ///     block (result payload)   ;; entry 0, a tag
    ///       try_table (catch tag 0) ... (catch_all n-1)
    ///         <args> call f
    ///         local.set results
    ///       end
    ///       <normal edge>
    ///     end
    ///     local.set payload
    ///     <edge of entry 0>
    ///   ...
    /// end
    /// <edge of entry n-1>
    /// ```
    ///
    /// Context items of the exception table are ignored: they tell a native
    /// unwinder how to interpret tags, which `Env::exception_tag` resolves
    /// statically here.
    fn try_call(&mut self, b: Block, term: Inst) -> Result<()> {
        let func = self.e.func;
        let dfg = &func.dfg;
        let pool = &dfg.value_lists;
        let (func_ref, et) = match dfg.insts[term] {
            InstructionData::TryCall {
                func_ref,
                exception,
                ..
            } => (Some(func_ref), exception),
            InstructionData::TryCallIndirect { exception, .. } => (None, exception),
            _ => unreachable!(),
        };
        let table = &dfg.exception_tables[et];
        let sig_ref = table.signature();
        let sig = &dfg.signatures[sig_ref];
        let (_, results) = wasm_signature(sig)?;
        let normal = *table.normal_return();
        let entries: Vec<(Option<ExceptionTag>, BlockCall)> = table
            .items()
            .filter_map(|item| match item {
                ExceptionTableItem::Tag(tag, call) => Some((Some(tag), call)),
                ExceptionTableItem::Default(call) => Some((None, call)),
                ExceptionTableItem::Context(_) => None,
            })
            .collect();

        // The payload: the calling convention's, with pointer-sized values
        // as wide as the handlers say (i64 if none uses them).
        let exn_ty = |call: &BlockCall| {
            let params = dfg.block_params(call.block(pool));
            call.args(pool)
                .zip(params)
                .find_map(|(a, &p)| matches!(a, BlockArg::TryCallExn(_)).then(|| dfg.value_type(p)))
        };
        let ptr = entries
            .iter()
            .find_map(|(_, call)| exn_ty(call))
            .unwrap_or(types::I64);
        let payload_ir = sig.call_conv.exception_payload_types(ptr).to_vec();
        if !entries.is_empty() && payload_ir.is_empty() {
            return unsupported(format!(
                "exceptions with calling convention {}",
                sig.call_conv
            ));
        }
        for (tag, call) in &entries {
            if tag.is_none() && exn_ty(call).is_some() {
                return unsupported("catch-all handler using the exception payload");
            }
        }
        let payload: Vec<ValType> = payload_ir
            .iter()
            .map(|&t| wasm_type(t))
            .collect::<Result<_>>()?;
        let payload_block = match payload[..] {
            [] => BlockType::Empty,
            [t] => BlockType::Result(t),
            _ => {
                let mut s = Signature::new(sig.call_conv);
                s.returns.extend(payload_ir.iter().map(|&t| AbiParam::new(t)));
                BlockType::FunctionType(self.e.env.type_index(&s)?.0)
            }
        };

        // One label per entry, entry 0 innermost.
        let mut catches = Vec::with_capacity(entries.len());
        for (i, (tag, _)) in entries.iter().enumerate() {
            let label = i as u32;
            catches.push(match tag {
                Some(tag) => Catch::One {
                    tag: self.e.env.exception_tag(*tag)?.0,
                    label,
                },
                None => Catch::All { label },
            });
        }
        for (tag, _) in entries.iter().rev() {
            self.e.ins(I::Block(if tag.is_some() {
                payload_block
            } else {
                BlockType::Empty
            }));
            self.labels.push(Label::Other);
        }

        let result_locals: Vec<Local> = results.iter().map(|&t| self.e.new_local(t)).collect();
        let payload_locals: Vec<Local> = payload.iter().map(|&t| self.e.new_local(t)).collect();
        self.e.ins(I::TryTable(BlockType::Empty, Cow::Owned(catches)));
        self.labels.push(Label::Other);
        let args = dfg.inst_args(term);
        match func_ref {
            Some(func_ref) => {
                for &a in args {
                    self.e.value(a)?;
                }
                let idx = self.e.callee_index(func_ref)?;
                self.e.ins(I::Call(idx.0));
            }
            None => {
                for &a in &args[1..] {
                    self.e.value(a)?;
                }
                self.e.value(args[0])?;
                if dfg.value_type(args[0]) == types::I64 {
                    self.e.ins(I::I32WrapI64);
                }
                let (type_index, table_index) = self.e.indirect_call(sig_ref)?;
                self.e.ins(I::CallIndirect {
                    type_index: type_index.0,
                    table_index: table_index.0,
                });
            }
        }
        for &l in result_locals.iter().rev() {
            self.e.local_set(l);
        }
        self.labels.pop();
        self.e.ins(I::End);
        self.try_results[b] = result_locals;
        self.try_payload[b] = payload_locals;

        // After the i-th `end` comes the edge of the i-th entry.
        for &(tag, call) in entries.iter().rev() {
            self.tasks.push(Task::Catch {
                src: b,
                call,
                payload: tag.is_some(),
            });
            self.tasks.push(Task::End);
        }
        self.tasks.push(Task::Branch { src: b, call: normal });
        Ok(())
    }

    /// Move the block arguments into the target's parameters and transfer
    /// control.
    fn branch(&mut self, src: Block, call: BlockCall) -> Result<()> {
        let func = self.e.func;
        let info = self.e.info;
        let pool = &func.dfg.value_lists;
        let params = func.dfg.block_params(call.block(pool));
        // Push every argument, then set the parameters in reverse. The
        // `ret`/`exn` arguments of a `try_call` edge come from locals.
        let mut sets = Vec::new();
        for (a, &p) in call.args(pool).zip(params) {
            if !info.is_live(p) {
                continue;
            }
            match a {
                BlockArg::Value(a) if self.e.locals.move_needed(a, p) => self.e.value(a)?,
                BlockArg::Value(_) => continue,
                BlockArg::TryCallRet(i) => {
                    self.e.local_get(self.try_results[src][i as usize]);
                }
                BlockArg::TryCallExn(i) => {
                    self.e.local_get(self.try_payload[src][i as usize]);
                }
            }
            sets.push(p);
        }
        for &p in sets.iter().rev() {
            self.e.local_set(self.e.local(p));
        }

        let t = call.block(&func.dfg.value_lists);
        if let Some(label) = self.dispatch_label {
            self.e.ins(I::I32Const(info.rpo_num[t].0 as i32));
            self.e.local_set(label);
            self.e.ins(I::Br(self.br_depth(Label::Dispatch)));
        } else if info.is_backward(src, t) {
            self.e.ins(I::Br(self.br_depth(Label::Loop(t))));
        } else if info.is_merge(t) {
            self.e.ins(I::Br(self.br_depth(Label::BlockFollowedBy(t))));
        } else {
            self.tasks.push(Task::Tree(t));
        }
        Ok(())
    }
}
