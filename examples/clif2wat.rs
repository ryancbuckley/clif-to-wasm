//! Compile the functions in a CLIF file to a wasm module and print it as WAT.
//!
//! Usage: cargo run --example clif2wat -- file.clif [function-name]

#[path = "../tests/harness/mod.rs"]
mod harness;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).expect("read input");
    let mut funcs = cranelift_reader::parse_functions(&text).expect("parse");
    if let Some(only) = args.get(2) {
        funcs.retain(|f| harness::func_name(f) == *only);
    }
    let (wasm, unsupported) =
        harness::compile_module(&funcs, &clif_to_wasm::Config::default());
    for u in unsupported {
        eprintln!("unsupported: {u}");
    }
    if let Err(e) = harness::validate(&wasm) {
        eprintln!("INVALID: {}", e.lines().next().unwrap_or(""));
    }
    if let Ok(out) = std::env::var("WASM_OUT") {
        std::fs::write(out, &wasm).unwrap();
    }
    println!("{}", wasmprinter::print_bytes(&wasm).unwrap());
}
