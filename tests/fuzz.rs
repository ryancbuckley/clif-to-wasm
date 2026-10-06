//! Differential testing: random CLIF functions are run in Cranelift's
//! interpreter and, compiled by this backend, in wasmtime. Results (or the fact
//! that both trap) must agree.
//!
//! `FUZZ_ITERS` sets the number of functions (default 300); `FUZZ_SEED`
//! picks the starting seed.

mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::Type;
use cranelift_interpreter::environment::FunctionStore;
use cranelift_interpreter::interpreter::{Interpreter, InterpreterState};
use cranelift_interpreter::step::ControlFlow;
use cranelift_wasm_backend::Config;
use harness::clifgen::{Rng, arg, generate};

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
