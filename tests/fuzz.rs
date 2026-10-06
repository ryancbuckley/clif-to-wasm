//! Differential testing: random CLIF functions are run in Cranelift's
//! interpreter and, compiled by this backend, in wasmtime. Results (or the fact
//! that both trap) must agree.
//!
//! `FUZZ_ITERS` sets the number of functions (default 300); `FUZZ_SEED`
//! picks the starting seed.

mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::{Ieee32, Ieee64};
use cranelift_codegen::ir::{
    AbiParam, Function, InstBuilder, MemFlagsData, Signature, StackSlotData, StackSlotKind, Type,
    UserFuncName, Value, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_interpreter::environment::FunctionStore;
use cranelift_interpreter::interpreter::{Interpreter, InterpreterState};
use cranelift_interpreter::step::ControlFlow;
use cranelift_wasm_backend::Config;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
    fn int(&mut self) -> i64 {
        match self.below(4) {
            0 => self.below(8) as i64 - 2,
            1 => self.pick(&[
                i64::MIN,
                i64::MAX,
                0x7f,
                0x80,
                0xff,
                0x7fff,
                0x8000,
                0xffff_ffff,
            ]),
            _ => self.next() as i64,
        }
    }
}

const TYPES: [Type; 6] = [
    types::I8,
    types::I16,
    types::I32,
    types::I64,
    types::F32,
    types::F64,
];
const SLOT_SIZE: u32 = 64;

struct Gen<'a, 'b> {
    rng: &'a mut Rng,
    b: FunctionBuilder<'b>,
    vars: Vec<(Variable, Type)>,
    slot: cranelift_codegen::ir::StackSlot,
    budget: u32,
}

impl Gen<'_, '_> {
    fn var_of(&mut self, ty: Type) -> Option<Value> {
        let cands: Vec<_> = self
            .vars
            .iter()
            .filter(|(_, t)| *t == ty)
            .map(|(v, _)| *v)
            .collect();
        if cands.is_empty() {
            return None;
        }
        let v = self.rng.pick(&cands);
        Some(self.b.use_var(v))
    }

    fn constant(&mut self, ty: Type) -> Value {
        match ty {
            types::F32 => {
                let f = self
                    .rng
                    .pick(&[0.0f32, -0.0, 1.5, -3.25, 1e30, f32::INFINITY]);
                self.b.ins().f32const(Ieee32::with_float(f))
            }
            types::F64 => {
                let f = self
                    .rng
                    .pick(&[0.0f64, -0.0, 2.5, -7.75, 1e300, f64::NEG_INFINITY]);
                self.b.ins().f64const(Ieee64::with_float(f))
            }
            _ => {
                let i = self.rng.int();
                self.b.ins().iconst(ty, i)
            }
        }
    }

    fn leaf(&mut self, ty: Type) -> Value {
        if self.rng.chance(75)
            && let Some(v) = self.var_of(ty)
        {
            return v;
        }
        self.constant(ty)
    }

    fn addr(&mut self, size: u32) -> (Value, i32) {
        let slots = SLOT_SIZE / 8;
        let off = (self.rng.below(slots as u64) as u32 * 8).min(SLOT_SIZE - size) as i32;
        // Split the offset between the address and the immediate.
        let split = if self.rng.chance(50) { off } else { 0 };
        let a = self.b.ins().stack_addr(types::I64, self.slot, split);
        (a, off - split)
    }

    fn expr(&mut self, ty: Type, depth: u32) -> Value {
        if depth == 0 || self.rng.chance(20) {
            return self.leaf(ty);
        }
        let d = depth - 1;
        if ty.is_float() {
            return match self.rng.below(9) {
                0 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fadd(a, b)
                }
                1 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fmul(a, b)
                }
                2 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fsub(a, b)
                }
                3 => {
                    let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().fmin(a, b)
                }
                4 => {
                    let a = self.expr(ty, d);
                    self.b.ins().fneg(a)
                }
                5 => {
                    let src = self.rng.pick(&[types::I8, types::I32, types::I64]);
                    let a = self.expr(src, d);
                    if self.rng.chance(50) {
                        self.b.ins().fcvt_from_sint(ty, a)
                    } else {
                        self.b.ins().fcvt_from_uint(ty, a)
                    }
                }
                6 => {
                    let other = if ty == types::F32 {
                        types::F64
                    } else {
                        types::F32
                    };
                    let a = self.expr(other, d);
                    if ty == types::F32 {
                        self.b.ins().fdemote(ty, a)
                    } else {
                        self.b.ins().fpromote(ty, a)
                    }
                }
                7 => self.load(ty),
                _ => {
                    let c = self.cond(d);
                    let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                    self.b.ins().select(c, x, y)
                }
            };
        }
        match self.rng.below(24) {
            0..=6 => {
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.iadd(a, b),
                    1 => ins.isub(a, b),
                    2 => ins.imul(a, b),
                    3 => ins.band(a, b),
                    4 => ins.bor(a, b),
                    5 => ins.bxor(a, b),
                    _ => ins.iadd(a, b),
                }
            }
            7 => {
                let a = self.expr(ty, d);
                let amt_ty = self.rng.pick(&[types::I8, types::I32, types::I64]);
                let b = self.expr(amt_ty, d);
                let ins = self.b.ins();
                match self.rng.below(5) {
                    0 => ins.ishl(a, b),
                    1 => ins.ushr(a, b),
                    2 => ins.sshr(a, b),
                    3 => ins.rotl(a, b),
                    _ => ins.rotr(a, b),
                }
            }
            8 => {
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let ins = self.b.ins();
                match self.rng.below(4) {
                    0 => ins.smin(a, b),
                    1 => ins.smax(a, b),
                    2 => ins.umin(a, b),
                    _ => ins.umax(a, b),
                }
            }
            9 => {
                // Division may trap; both sides must agree.
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                let b = if self.rng.chance(80) {
                    let one = self.b.ins().iconst(ty, 1);
                    self.b.ins().bor(b, one)
                } else {
                    b
                };
                match self.rng.below(4) {
                    0 => self.b.ins().udiv(a, b),
                    1 => self.b.ins().sdiv(a, b),
                    2 => self.b.ins().urem(a, b),
                    _ => {
                        // The interpreter traps on `MIN srem -1`, which
                        // Cranelift defines as 0; keep the divisor positive.
                        let one = self.b.ins().iconst(ty, 1);
                        let b = self.b.ins().ushr(b, one);
                        let b = self.b.ins().bor(b, one);
                        self.b.ins().srem(a, b)
                    }
                }
            }
            10 => {
                let a = self.expr(ty, d);
                let ins = self.b.ins();
                match self.rng.below(7) {
                    0 => ins.clz(a),
                    1 => ins.ctz(a),
                    2 => ins.popcnt(a),
                    3 => ins.ineg(a),
                    4 => ins.bnot(a),
                    5 => ins.iabs(a),
                    _ => ins.cls(a),
                }
            }
            11 => {
                let c = self.cond(d);
                let (x, y) = (self.expr(ty, d), self.expr(ty, d));
                self.b.ins().select(c, x, y)
            }
            12 => {
                let c = self.cond(d);
                if ty == types::I8 {
                    c
                } else {
                    self.b.ins().uextend(ty, c)
                }
            }
            13 | 14 => {
                // Conversions from another integer type.
                let src = self
                    .rng
                    .pick(&[types::I8, types::I16, types::I32, types::I64]);
                let a = self.expr(src, d);
                if src == ty {
                    a
                } else if src.bits() > ty.bits() {
                    self.b.ins().ireduce(ty, a)
                } else if self.rng.chance(50) {
                    self.b.ins().uextend(ty, a)
                } else {
                    self.b.ins().sextend(ty, a)
                }
            }
            15 => {
                let f = self.rng.pick(&[types::F32, types::F64]);
                let a = self.expr(f, d);
                if ty == types::I32 || ty == types::I64 {
                    if self.rng.chance(50) {
                        self.b.ins().fcvt_to_sint_sat(ty, a)
                    } else {
                        self.b.ins().fcvt_to_uint_sat(ty, a)
                    }
                } else {
                    self.b.ins().fcvt_to_sint(ty, a)
                }
            }
            16 | 17 => self.load(ty),
            18 => {
                let (a, b) = (self.expr(ty, d), self.expr(ty, d));
                if ty.bits() <= 32 {
                    if self.rng.chance(50) {
                        self.b.ins().umulhi(a, b)
                    } else {
                        self.b.ins().smulhi(a, b)
                    }
                } else {
                    self.b.ins().bitselect(a, b, a)
                }
            }
            19 => {
                // A load whose value is used after a store to the same
                // place: tests effect ordering.
                let v = self.load(ty);
                self.store();
                v
            }
            _ => self.leaf(ty),
        }
    }

    fn load(&mut self, ty: Type) -> Value {
        let (a, off) = self.addr(ty.bytes());
        let ins = self.b.ins();
        if ty.is_int() && ty.bits() < 64 && self.rng.chance(40) {
            let wide = if ty == types::I32 || self.rng.chance(50) {
                types::I64
            } else {
                types::I32
            };
            let w = if ty.bits() == 8 {
                ins.uload8(wide, MemFlagsData::trusted(), a, off)
            } else if ty.bits() == 16 {
                ins.sload16(wide, MemFlagsData::trusted(), a, off)
            } else {
                ins.sload32(MemFlagsData::trusted(), a, off)
            };
            return self.b.ins().ireduce(ty, w);
        }
        ins.load(ty, MemFlagsData::trusted(), a, off)
    }

    fn store(&mut self) {
        let ty = self.rng.pick(&TYPES);
        let v = self.expr(ty, 2);
        let (a, off) = self.addr(ty.bytes());
        if ty == types::I64 && self.rng.chance(30) {
            self.b.ins().istore16(MemFlagsData::trusted(), v, a, off);
        } else {
            self.b.ins().store(MemFlagsData::trusted(), v, a, off);
        }
    }

    fn cond(&mut self, depth: u32) -> Value {
        let ty = self.rng.pick(&TYPES);
        let (a, b) = (self.expr(ty, depth), self.expr(ty, depth));
        if ty.is_float() {
            let cc = self.rng.pick(&[
                FloatCC::Ordered,
                FloatCC::Unordered,
                FloatCC::Equal,
                FloatCC::NotEqual,
                FloatCC::OrderedNotEqual,
                FloatCC::UnorderedOrEqual,
                FloatCC::LessThan,
                FloatCC::LessThanOrEqual,
                FloatCC::GreaterThan,
                FloatCC::GreaterThanOrEqual,
                FloatCC::UnorderedOrLessThan,
                FloatCC::UnorderedOrLessThanOrEqual,
                FloatCC::UnorderedOrGreaterThan,
                FloatCC::UnorderedOrGreaterThanOrEqual,
            ]);
            self.b.ins().fcmp(cc, a, b)
        } else {
            let cc = self.rng.pick(&[
                IntCC::Equal,
                IntCC::NotEqual,
                IntCC::SignedLessThan,
                IntCC::SignedGreaterThanOrEqual,
                IntCC::SignedGreaterThan,
                IntCC::SignedLessThanOrEqual,
                IntCC::UnsignedLessThan,
                IntCC::UnsignedGreaterThanOrEqual,
                IntCC::UnsignedGreaterThan,
                IntCC::UnsignedLessThanOrEqual,
            ]);
            self.b.ins().icmp(cc, a, b)
        }
    }

    fn stmts(&mut self, depth: u32) {
        let n = 1 + self.rng.below(4);
        for _ in 0..n {
            if self.budget == 0 {
                return;
            }
            self.budget -= 1;
            self.stmt(depth);
        }
    }

    fn assign(&mut self) {
        let i = self.rng.below(self.vars.len() as u64) as usize;
        let (var, ty) = self.vars[i];
        let v = self.expr(ty, 3);
        self.b.def_var(var, v);
    }

    fn stmt(&mut self, depth: u32) {
        match self.rng.below(if depth == 0 { 3 } else { 7 }) {
            0 | 1 => self.assign(),
            2 => self.store(),
            3 => {
                // if / else
                let c = self.cond(1);
                let (t, e, m) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().brif(c, t, &[], e, &[]);
                self.b.switch_to_block(t);
                self.stmts(depth - 1);
                self.b.ins().jump(m, &[]);
                self.b.switch_to_block(e);
                if self.rng.chance(70) {
                    self.stmts(depth - 1);
                }
                self.b.ins().jump(m, &[]);
                self.b.switch_to_block(m);
            }
            4 => {
                // A counted loop: 0..4 iterations.
                let counter = self.b.declare_var(types::I32);
                let n = self.b.ins().iconst(types::I32, self.rng.below(5) as i64);
                self.b.def_var(counter, n);
                let (h, body, exit) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().jump(h, &[]);
                self.b.switch_to_block(h);
                let c = self.b.use_var(counter);
                self.b.ins().brif(c, body, &[], exit, &[]);
                self.b.switch_to_block(body);
                self.stmts(depth - 1);
                let c = self.b.use_var(counter);
                let c = self.b.ins().iadd_imm_s(c, -1);
                self.b.def_var(counter, c);
                if self.rng.chance(20) {
                    // An early exit from the loop.
                    let k = self.cond(1);
                    let cont = self.b.create_block();
                    self.b.ins().brif(k, exit, &[], cont, &[]);
                    self.b.switch_to_block(cont);
                }
                self.b.ins().jump(h, &[]);
                self.b.switch_to_block(exit);
            }
            5 => {
                // br_table
                let ity = types::I32;
                let idx = self.expr(ity, 1);
                let n = 1 + self.rng.below(4) as usize;
                let merge = self.b.create_block();
                let targets: Vec<_> = (0..n).map(|_| self.b.create_block()).collect();
                let pool = &mut self.b.func.dfg.value_lists;
                let calls: Vec<_> = (0..n)
                    .map(|i| {
                        let t = targets[(self.rng.below(n as u64) as usize).min(i + 1) % n];
                        cranelift_codegen::ir::BlockCall::new(t, [], pool)
                    })
                    .collect();
                let def = cranelift_codegen::ir::BlockCall::new(targets[0], [], pool);
                let jt = self
                    .b
                    .create_jump_table(cranelift_codegen::ir::JumpTableData::new(def, &calls));
                self.b.ins().br_table(idx, jt);
                for t in targets {
                    self.b.switch_to_block(t);
                    self.stmts(depth - 1);
                    self.b.ins().jump(merge, &[]);
                }
                self.b.switch_to_block(merge);
            }
            _ => {
                // A conditional trap that rarely fires.
                let c = self.cond(1);
                let k = self.b.ins().iconst(types::I8, 1);
                let both = self.b.ins().band(c, k);
                let x = self.b.ins().icmp_imm_s(IntCC::Equal, both, 7);
                self.b
                    .ins()
                    .trapnz(x, cranelift_codegen::ir::TrapCode::unwrap_user(1));
            }
        }
    }
}

fn generate(seed: u64) -> (Function, Vec<Type>) {
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let nparams = 1 + rng.below(4) as usize;
    let params: Vec<Type> = (0..nparams).map(|_| rng.pick(&TYPES)).collect();
    let nvars = 2 + rng.below(5) as usize;
    let var_tys: Vec<Type> = (0..nvars).map(|_| rng.pick(&TYPES)).collect();

    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.extend(params.iter().map(|&t| AbiParam::new(t)));
    sig.returns
        .extend(var_tys.iter().map(|&t| AbiParam::new(t)));
    let mut func = Function::with_name_signature(UserFuncName::testcase("f"), sig);
    let mut ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut func, &mut ctx);
    let slot = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        SLOT_SIZE,
        3,
    ));
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let pvals = b.block_params(entry).to_vec();

    let budget = std::env::var("FUZZ_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let mut g = Gen {
        rng: &mut rng,
        b,
        vars: vec![],
        slot,
        budget,
    };
    // Zero the slot so loads are deterministic.
    for off in (0..SLOT_SIZE).step_by(8) {
        let z = g.b.ins().iconst(types::I64, 0);
        let a = g.b.ins().stack_addr(types::I64, slot, 0);
        g.b.ins().store(MemFlagsData::trusted(), z, a, off as i32);
    }
    for (i, &ty) in var_tys.iter().enumerate() {
        let var = g.b.declare_var(ty);
        let init = match pvals.iter().position(|&p| g.b.func.dfg.value_type(p) == ty) {
            Some(j) if g.rng.chance(70) => pvals[j],
            _ => g.constant(ty),
        };
        g.b.def_var(var, init);
        g.vars.push((var, ty));
        let _ = i;
    }
    for (j, &p) in pvals.iter().enumerate() {
        let var = g.b.declare_var(params[j]);
        g.b.def_var(var, p);
        g.vars.push((var, params[j]));
    }
    g.stmts(3);
    while g.budget > 0 && g.budget < 1000 {
        g.budget -= 1;
        g.stmt(3);
    }
    let rets: Vec<Value> = (0..nvars).map(|i| g.b.use_var(g.vars[i].0)).collect();
    g.b.ins().return_(&rets);
    g.b.seal_all_blocks();
    g.b.finalize(cranelift_codegen::isa::TargetFrontendConfig {
        default_call_conv: CallConv::SystemV,
        pointer_width: target_lexicon::PointerWidth::U64,
        page_size_align_log2: 12,
    });
    (func, params)
}

fn arg(rng: &mut Rng, ty: Type) -> DataValue {
    let i = rng.int();
    match ty {
        types::I8 => DataValue::I8(i as i8),
        types::I16 => DataValue::I16(i as i16),
        types::I32 => DataValue::I32(i as i32),
        types::I64 => DataValue::I64(i),
        types::F32 => DataValue::F32(Ieee32::with_float((i % 1000) as f32 / 8.0)),
        _ => DataValue::F64(Ieee64::with_float((i % 100000) as f64 / 16.0)),
    }
}

#[test]
fn differential() {
    let iters: u64 = std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let seed0: u64 = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut failures = Vec::new();
    let mut compared = 0;
    let mut trapped = 0;
    for seed in seed0..seed0 + iters {
        let (func, params) = generate(seed);
        cranelift_codegen::verify_function(
            &func,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap_or_else(|e| panic!("seed {seed}: generator made invalid IR: {e}\n{func}"));

        for config in [
            Config::default(),
            Config {
                use_tee: false,
                ..Config::default()
            },
        ] {
            let (wasm, unsupported) = harness::compile_module(std::slice::from_ref(&func), &config);
            if !unsupported.is_empty() {
                failures.push(format!("seed {seed}: unsupported {unsupported:?}"));
                continue;
            }
            if let Err(e) = harness::validate(&wasm) {
                failures.push(format!("seed {seed}: invalid wasm: {e}\n{func}"));
                continue;
            }
            let runner = harness::Runner::new(&wasm).unwrap();
            let returns: Vec<Type> = func
                .signature
                .returns
                .iter()
                .map(|p| p.value_type)
                .collect();
            let mut rng = Rng(seed ^ 0xdead_beef);
            for _ in 0..4 {
                let args: Vec<DataValue> = params.iter().map(|&t| arg(&mut rng, t)).collect();

                let mut store = FunctionStore::default();
                store.add("%f".into(), &func);
                let state = InterpreterState::default().with_function_store(store);
                let mut interp = Interpreter::new(state).with_fuel(Some(1_000_000));
                let expected = match interp.call_by_name("%f", &args) {
                    Ok(ControlFlow::Return(vals)) => Some(vals.to_vec()),
                    Ok(ControlFlow::Trap(_)) => None,
                    other => panic!("seed {seed}: interpreter: {other:?}"),
                };
                let got = runner
                    .call("f", &args, &returns)
                    .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
                compared += 1;
                trapped += expected.is_none() as u32;
                let ok = match (&expected, &got) {
                    (None, None) => true,
                    (Some(e), Some(g)) => harness::same_results(e, g),
                    _ => false,
                };
                if !ok {
                    failures.push(format!(
                        "seed {seed} use_tee={}: args {args:?}\n  expected {expected:?}\n  got      {got:?}\n{func}",
                        config.use_tee
                    ));
                    break;
                }
            }
        }
        if failures.len() > 3 {
            break;
        }
    }
    println!("{compared} executions compared, {trapped} trapped");
    if !failures.is_empty() {
        panic!("{} failures; first:\n{}", failures.len(), failures[0]);
    }
}
