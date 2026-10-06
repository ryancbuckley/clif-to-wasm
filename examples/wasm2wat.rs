//! Print a wasm file as WAT, e.g. the benchmark modules in `target/bench`.
//!
//! Usage: cargo run --example wasm2wat -- file.wasm

fn main() {
    let path = std::env::args().nth(1).expect("usage: wasm2wat file.wasm");
    let wasm = std::fs::read(path).expect("read input");
    println!("{}", wasmprinter::print_bytes(&wasm).expect("print"));
}
