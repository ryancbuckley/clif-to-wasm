//! Differential testing: random CLIF functions are run in Cranelift's
//! interpreter and, compiled by this backend, in wasmtime. Results (or the fact
//! that both trap) must agree.
//!
//! Each seed gives three functions: scalar only, with vector code, and with
//! `i128` code. `FUZZ_ITERS` sets the number of seeds (default 150),
//! `FUZZ_SEED` the first seed, and `FUZZ_ONLY=scalar|simd|i128` restricts
//! the kind.

mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::Type;
use cranelift_interpreter::environment::FunctionStore;
use cranelift_interpreter::interpreter::{Interpreter, InterpreterState};
use cranelift_interpreter::step::ControlFlow;
use cranelift_wasm_backend::Config;
use harness::clifgen::{Rng, arg, generate, generate_i128, generate_simd};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Scalar,
    Simd,
    I128,
}

#[test]
fn differential() {
    let iters: u64 = std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let seed0: u64 = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mut failures = Vec::new();
    let mut compared = 0;
    let mut trapped = 0;
    let mut skipped = 0;
    let only = match std::env::var("FUZZ_ONLY").as_deref() {
        Ok("scalar") => Some(Kind::Scalar),
        Ok("simd") => Some(Kind::Simd),
        Ok("i128") => Some(Kind::I128),
        _ => None,
    };
    let runs = (seed0..seed0 + iters)
        .flat_map(|seed| [(seed, Kind::Scalar), (seed, Kind::Simd), (seed, Kind::I128)])
        .filter(|&(_, kind)| only.is_none_or(|o| o == kind));
    for (seed, kind) in runs {
        let (func, params) = match kind {
            Kind::Scalar => generate(seed),
            Kind::Simd => generate_simd(seed),
            Kind::I128 => generate_i128(seed),
        };
        cranelift_codegen::verify_function(
            &func,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap_or_else(|e| panic!("seed {seed} {kind:?}: generator made invalid IR: {e}\n{func}"));

        for config in [
            Config::default(),
            Config {
                stackify: false,
                use_tee: false,
                ..Config::default()
            },
            Config {
                use_tee: false,
                ..Config::default()
            },
            Config {
                coalesce: false,
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
                let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    interp.call_by_name("%f", &args).map(|cf| match cf {
                        ControlFlow::Return(vals) => Some(vals.to_vec()),
                        ControlFlow::Trap(_) => None,
                        other => panic!("seed {seed}: interpreter: {other:?}"),
                    })
                }));
                let expected = match call {
                    Ok(Ok(e)) => e,
                    Ok(Err(e)) => panic!("seed {seed} {kind:?}: interpreter error: {e}\n{func}"),
                    // The interpreter does not implement every vector
                    // operation on every type; such cases are inconclusive.
                    Err(_) => {
                        skipped += 1;
                        break;
                    }
                };
                let got = runner
                    .call("f", &args, &returns)
                    .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
                compared += 1;
                trapped += expected.is_none() as u32;
                let ok = match (&expected, &got) {
                    (None, None) => true,
                    (Some(e), Some(g)) => harness::same_results(e, g, &returns),
                    _ => false,
                };
                if !ok {
                    failures.push(format!(
                        "seed {seed} {kind:?} {:?}: args {args:?}\n  expected {expected:?}\n  got      {got:?}\n{func}",
                        config
                    ));
                    break;
                }
            }
        }
        if failures.len() > 3 {
            break;
        }
    }
    println!(
        "{compared} executions compared, {trapped} trapped, {skipped} functions skipped (interpreter panic)"
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {}", f.lines().next().unwrap_or(""));
        }
        panic!("{} failures; first:\n{}", failures.len(), failures[0]);
    }
}
