//! Code generation benchmarks.
//!
//! Reports, for the naive (no expression trees) and default configurations:
//!
//! - static output metrics: code bytes, declared locals, `local.*` ops;
//! - compile speed of this backend;
//! - runtime of the kernels in wasmtime (Cranelift), and in V8's baseline
//!   (Liftoff) and optimizing (TurboFan) tiers when `node` is available,
//!   plus V8's compile time for the module.
//!
//! Corpora: the kernels in `tests/harness/kernels.rs`, random functions
//! from `tests/harness/clifgen.rs`, and Cranelift's upstream runtests if
//! `UPSTREAM_RUNTESTS` points at them.
//!
//! `BENCH_QUICK=1` uses fewer repetitions and smaller corpora.

#[path = "../tests/harness/mod.rs"]
mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{Function, types};
use cranelift_wasm_backend::Config;
use harness::kernels::{KERNELS, functions};
use std::time::Instant;

fn variants() -> Vec<(&'static str, Config)> {
    vec![
        (
            "naive",
            Config {
                stackify: false,
                use_tee: false,
                ..Config::default()
            },
        ),
        ("default", Config::default()),
    ]
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    funcs: usize,
    code_bytes: usize,
    locals: u64,
    ops: usize,
    get: usize,
    set: usize,
    tee: usize,
}

fn metrics(wasm: &[u8]) -> Metrics {
    let mut m = Metrics::default();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            m.funcs += 1;
            m.code_bytes += body.range().len();
            for local in body.get_locals_reader().unwrap() {
                m.locals += local.unwrap().0 as u64;
            }
            let mut ops = body.get_operators_reader().unwrap();
            while !ops.eof() {
                m.ops += 1;
                match ops.read().unwrap() {
                    wasmparser::Operator::LocalGet { .. } => m.get += 1,
                    wasmparser::Operator::LocalSet { .. } => m.set += 1,
                    wasmparser::Operator::LocalTee { .. } => m.tee += 1,
                    _ => {}
                }
            }
        }
    }
    m
}

fn upstream_corpus() -> Vec<Vec<Function>> {
    let Ok(dir) = std::env::var("UPSTREAM_RUNTESTS") else {
        return vec![];
    };
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "clif"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let text: String = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .filter(|l| {
                !l.trim_start().starts_with("target ") && !l.trim_start().starts_with("set ")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let Ok(funcs) = cranelift_reader::parse_functions(&text) else {
            continue;
        };
        // Keep files the backend fully supports, so that both variants see
        // the same functions.
        let (_, unsupported) = harness::compile_module(&funcs, &Config::default());
        if unsupported.is_empty() {
            out.push(funcs);
        }
    }
    out
}

fn num_insts(funcs: &[Function]) -> usize {
    funcs
        .iter()
        .map(|f| {
            f.layout
                .blocks()
                .map(|b| f.layout.block_insts(b).count())
                .sum::<usize>()
        })
        .sum()
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    // `cargo bench` passes `--bench`; ignore arguments.
    let quick = std::env::var("BENCH_QUICK").is_ok();
    let reps = if quick { 3 } else { 7 };
    let fuzz_n = if quick { 200 } else { 1000 };

    let kernel_funcs = functions();
    let fuzz_funcs: Vec<Function> = (1..=fuzz_n)
        .map(|s| harness::clifgen::generate(s).0)
        .collect();
    let upstream = upstream_corpus();
    let mut corpora: Vec<(&str, Vec<Vec<Function>>)> = vec![
        ("kernels", vec![kernel_funcs.clone()]),
        (
            "random",
            fuzz_funcs.iter().map(|f| vec![f.clone()]).collect(),
        ),
    ];
    if !upstream.is_empty() {
        corpora.push(("upstream", upstream));
    }

    println!("## Output size\n");
    println!(
        "| corpus | variant | funcs | code bytes | locals | ops | local.get | local.set | local.tee |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|");
    for (cname, modules) in &corpora {
        for (vname, config) in variants() {
            let mut total = Metrics::default();
            for funcs in modules {
                let (wasm, unsupported) = harness::compile_module(funcs, &config);
                assert!(unsupported.is_empty(), "{unsupported:?}");
                let m = metrics(&wasm);
                total.funcs += m.funcs;
                total.code_bytes += m.code_bytes;
                total.locals += m.locals;
                total.ops += m.ops;
                total.get += m.get;
                total.set += m.set;
                total.tee += m.tee;
            }
            let t = total;
            println!(
                "| {cname} | {vname} | {} | {} | {} | {} | {} | {} | {} |",
                t.funcs, t.code_bytes, t.locals, t.ops, t.get, t.set, t.tee
            );
        }
    }

    println!("\n## Compile speed (this backend, default config)\n");
    println!("Time in `compile_function` only, with aliases resolved beforehand.\n");
    println!("| corpus | funcs | CLIF insts | time | µs/func | ns/inst |");
    println!("|---|---:|---:|---:|---:|---:|");
    struct NoEnv;
    impl cranelift_wasm_backend::Env for NoEnv {
        fn func_index(
            &mut self,
            _: cranelift_wasm_backend::Name<'_>,
        ) -> cranelift_wasm_backend::Result<u32> {
            Ok(0)
        }
    }
    let config = Config {
        stack_pointer_global: Some(0),
        ..Config::default()
    };
    for (cname, modules) in &corpora {
        let funcs: Vec<Function> = modules
            .iter()
            .flatten()
            .map(|f| {
                let mut f = f.clone();
                f.dfg.resolve_all_aliases();
                f
            })
            // Skip the few that need more of an environment than `NoEnv`.
            .filter(|f| cranelift_wasm_backend::compile_function(f, &config, &mut NoEnv).is_ok())
            .collect();
        let insts = num_insts(&funcs);
        let mut times = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            for f in &funcs {
                std::hint::black_box(
                    cranelift_wasm_backend::compile_function(f, &config, &mut NoEnv).unwrap(),
                );
            }
            times.push(t.elapsed().as_secs_f64());
        }
        let t = median(&mut times);
        println!(
            "| {cname} | {} | {insts} | {:.2} ms | {:.1} | {:.0} |",
            funcs.len(),
            t * 1e3,
            t * 1e6 / funcs.len() as f64,
            t * 1e9 / insts as f64
        );
    }

    // Runtime.
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bench");
    std::fs::create_dir_all(&out_dir).unwrap();
    let node = std::process::Command::new("node")
        .arg("--version")
        .output()
        .is_ok();
    let tiers: &[(&str, &[&str])] = &[
        (
            "V8 Liftoff",
            &["--liftoff-only", "--no-wasm-lazy-compilation"],
        ),
        (
            "V8 TurboFan",
            &["--no-liftoff", "--no-wasm-lazy-compilation"],
        ),
    ];
    let mut rows: Vec<(String, String, String, f64)> = Vec::new(); // engine, kernel, variant, ms
    let mut v8_compile: Vec<(String, String, f64)> = Vec::new();
    for (vname, config) in variants() {
        let (wasm, _) = harness::compile_module(&kernel_funcs, &config);
        let path = out_dir.join(format!("kernels-{vname}.wasm"));
        std::fs::write(&path, &wasm).unwrap();
        let runner = harness::Runner::new(&wasm).unwrap();
        for k in KERNELS {
            let name = format!("k_{}", k.name);
            let mut ms = Vec::new();
            for _ in 0..reps {
                let (r, t) = runner.call_timed(&name, &[DataValue::I32(k.large)], &[types::I64]);
                r.unwrap().unwrap();
                ms.push(t * 1e3);
            }
            rows.push((
                "wasmtime".into(),
                k.name.into(),
                vname.into(),
                median(&mut ms),
            ));
        }
        if node {
            for (tier, flags) in tiers {
                let specs: Vec<String> = KERNELS
                    .iter()
                    .map(|k| format!("{}:{}", k.name, k.large))
                    .collect();
                let out = std::process::Command::new("node")
                    .args(*flags)
                    .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/benches/run.mjs"))
                    .arg(&path)
                    .arg(reps.to_string())
                    .args(&specs)
                    .output()
                    .unwrap();
                let text = String::from_utf8_lossy(&out.stdout);
                let Some(line) = text.lines().last() else {
                    eprintln!("node failed: {}", String::from_utf8_lossy(&out.stderr));
                    continue;
                };
                let json = parse_json(line);
                v8_compile.push((tier.to_string(), vname.into(), json.0));
                for (kname, mut ms) in json.1 {
                    rows.push((tier.to_string(), kname, vname.into(), median(&mut ms)));
                }
            }
        }
    }

    println!("\n## Kernel runtime (median ms; lower is better)\n");
    let engines: Vec<String> = {
        let mut e: Vec<String> = Vec::new();
        for r in &rows {
            if !e.contains(&r.0) {
                e.push(r.0.clone());
            }
        }
        e
    };
    let vnames: Vec<&str> = variants().iter().map(|v| v.0).collect();
    print!("| kernel |");
    for e in &engines {
        for v in &vnames {
            print!(" {e} {v} |");
        }
    }
    println!();
    println!("|---|{}", "---:|".repeat(engines.len() * vnames.len()));
    for k in KERNELS {
        print!("| {} |", k.name);
        for e in &engines {
            for v in &vnames {
                let r = rows
                    .iter()
                    .find(|r| r.0 == *e && r.1 == k.name && r.2 == *v);
                print!(" {} |", r.map_or("-".into(), |r| format!("{:.2}", r.3)));
            }
        }
        println!();
    }
    if !v8_compile.is_empty() {
        println!("\n## V8 compile time for the kernel module (ms)\n");
        println!("| tier | variant | ms |");
        println!("|---|---|---:|");
        for (t, v, ms) in v8_compile {
            println!("| {t} | {v} | {ms:.3} |");
        }
    }
}

/// Minimal parser for run.mjs's output.
fn parse_json(line: &str) -> (f64, Vec<(String, Vec<f64>)>) {
    let num = |s: &str| s.trim().parse::<f64>().unwrap();
    let compile = line
        .split("\"compile_ms\":")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap();
    let mut kernels = Vec::new();
    let body = line.split("\"kernels\":{").nth(1).unwrap();
    for part in body.split("},").map(|p| p.trim_end_matches("}}")) {
        let Some((name, rest)) = part.split_once(":{") else {
            continue;
        };
        let name = name.trim_matches(|c| c == '"' || c == ',').to_string();
        let ms_list = rest
            .split("\"ms\":[")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        kernels.push((name, ms_list.split(',').map(num).collect()));
    }
    (num(compile), kernels)
}
