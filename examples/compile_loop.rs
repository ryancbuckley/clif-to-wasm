//! Compile the random benchmark corpus in a loop (for profiling).
//!
//! Usage: cargo run --release --example compile_loop -- [seconds]

#[path = "../tests/harness/mod.rs"]
mod harness;

fn main() {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let funcs: Vec<_> = (1..=1000)
        .map(|s| harness::clifgen::generate(s).0)
        .collect();
    let config = harness::config(|c| c.module.stack_pointer_global = Some(0));
    // Resolve aliases up front, as Cranelift's own pipeline does, so that the
    // timing covers only this backend.
    let funcs: Vec<_> = funcs
        .into_iter()
        .map(|mut f| {
            f.dfg.resolve_all_aliases();
            f
        })
        .collect();
    struct NoEnv;
    impl cranelift_wasm_backend::Env for NoEnv {
        fn func_index(
            &mut self,
            _: cranelift_wasm_backend::Name<'_>,
        ) -> cranelift_wasm_backend::Result<u32> {
            Ok(0)
        }
    }
    let start = std::time::Instant::now();
    let mut n = 0;
    while start.elapsed().as_secs_f64() < secs {
        for f in &funcs {
            std::hint::black_box(
                cranelift_wasm_backend::compile_function(f, &config, &mut NoEnv).unwrap(),
            );
        }
        n += 1;
    }
    let per = start.elapsed().as_secs_f64() / (n * funcs.len()) as f64;
    println!("{n} rounds, {:.1} µs/function", per * 1e6);
}
