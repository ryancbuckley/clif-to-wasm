//! The benchmark kernels must compute the same results as Cranelift's
//! interpreter (at small sizes).

mod harness;

use clif_to_wasm::Config;
use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::types;
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
        harness::config(|c| c.opts.use_tee = false),
    ] {
        let (wasm, unsupported) = harness::compile_module(&funcs, &config);
        assert!(unsupported.is_empty(), "{unsupported:?}");
        harness::validate(&wasm).unwrap();
        let runner = harness::Runner::new(&wasm).unwrap();
        for k in KERNELS {
            let name = format!("k_{}", k.name);
            let args = [DataValue::I32(k.small)];
            let expected = match harness::interpret(&funcs, &name, &args) {
                Some(Ok(Some(v))) => v,
                other => panic!("{name}: interpreter: {other:?}"),
            };
            let got = runner.call(&name, &args, &[types::I64]).unwrap();
            assert_eq!(got, Some(expected.clone()), "{name}({})", k.small);
            println!("{name}({}) = {:?}", k.small, expected[0]);
        }
    }
}
