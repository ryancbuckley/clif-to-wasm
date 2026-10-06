//! The benchmark kernels must compute the same results as Cranelift's
//! interpreter (at small sizes).

mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::types;
use cranelift_interpreter::environment::FunctionStore;
use cranelift_interpreter::interpreter::{Interpreter, InterpreterState};
use cranelift_interpreter::step::ControlFlow;
use cranelift_wasm_backend::Config;
use harness::kernels::{KERNELS, functions};

#[test]
fn kernels_match_interpreter() {
    let funcs = functions();
    for f in &funcs {
        cranelift_codegen::verify_function(
            f,
            &cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap_or_else(|e| panic!("invalid kernel IR: {e}\n{f}"));
    }
    for config in [
        Config::default(),
        Config {
            use_tee: false,
            ..Config::default()
        },
    ] {
        let (wasm, unsupported) = harness::compile_module(&funcs, &config);
        assert!(unsupported.is_empty(), "{unsupported:?}");
        harness::validate(&wasm).unwrap();
        let runner = harness::Runner::new(&wasm).unwrap();
        for k in KERNELS {
            let name = format!("k_{}", k.name);
            let mut store = FunctionStore::default();
            for f in &funcs {
                store.add(f.name.to_string(), f);
            }
            let mut interp =
                Interpreter::new(InterpreterState::default().with_function_store(store));
            let args = [DataValue::I32(k.small)];
            let expected = match interp.call_by_name(&format!("%{name}"), &args) {
                Ok(ControlFlow::Return(v)) => v.to_vec(),
                other => panic!("{name}: interpreter: {other:?}"),
            };
            let got = runner.call(&name, &args, &[types::I64]).unwrap();
            assert_eq!(got, Some(expected.clone()), "{name}({})", k.small);
            println!("{name}({}) = {:?}", k.small, expected[0]);
        }
    }
}
