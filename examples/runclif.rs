//! Run the `; run:` commands of a CLIF file through the backend.
//!
//! Usage: cargo run --example runclif -- file.clif

#[path = "../tests/harness/mod.rs"]
mod harness;

fn main() {
    let path = std::env::args().nth(1).expect("usage: runclif file.clif");
    let text = std::fs::read_to_string(path).expect("read input");
    match harness::run_clif(&text, &clif_to_wasm::Config::default()) {
        Ok(out) => println!("{out:#?}"),
        Err(e) => println!("error: {e}"),
    }
}
