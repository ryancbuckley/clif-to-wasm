//! Run Cranelift's upstream runtests (`cranelift/filetests/filetests/runtests`
//! in the wasmtime repository) through the wasm backend and summarize.
//!
//! Usage: cargo run --example upstream -- <runtests dir> [-v]

#[path = "../tests/harness/mod.rs"]
mod harness;

use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "-v");
    let mut paths: Vec<_> = std::fs::read_dir(&args[1])
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "clif"))
        .collect();
    paths.sort();

    let (mut passed, mut failed, mut clean_files) = (0, 0, 0);
    // Files that only run under Cranelift's interpreter (`test interpret`
    // without `test run`) may encode interpreter-specific behavior.
    let mut interp_only_failures = Vec::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut failing_files = Vec::new();
    std::panic::set_hook(Box::new(|_| {}));
    for path in &paths {
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        // Drop ISA lines: the backend has no use for them and the reader
        // would need the corresponding Cranelift backends compiled in.
        let text: String = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| {
                let t = l.trim_start();
                if t.starts_with("target ") || t.starts_with("set ") {
                    ""
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut config = cranelift_wasm_backend::Config::default();
        config.module.demote_multi_results = std::env::var_os("DEMOTE").is_some();
        let result = std::panic::catch_unwind(|| harness::run_clif(&text, &config));
        match result {
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                failing_files.push(format!("{file}: PANIC {msg}"));
                failed += 1;
            }
            Ok(Err(e)) => {
                let first = e.lines().next().unwrap_or("").to_string();
                *reasons
                    .entry(format!("file error: {}", truncate(&first)))
                    .or_default() += 1;
                if verbose {
                    println!("{file}: {first}");
                }
            }
            Ok(Ok(out)) if !out.failures.is_empty() && !text.contains("test run") => {
                passed += out.passed;
                interp_only_failures.push(format!("{file}:\n    {}", out.failures.join("\n    ")));
            }
            Ok(Ok(out)) => {
                passed += out.passed;
                failed += out.failures.len();
                for u in &out.unsupported {
                    let reason = u.split_once(": ").map(|x| x.1).unwrap_or(u);
                    *reasons.entry(truncate(reason)).or_default() += 1;
                }
                if !out.failures.is_empty() {
                    failing_files.push(format!("{file}:\n    {}", out.failures.join("\n    ")));
                } else if out.unsupported.is_empty() {
                    clean_files += 1;
                }
                if verbose {
                    println!(
                        "{file}: {} passed, {} failed, {} unsupported",
                        out.passed,
                        out.failures.len(),
                        out.unsupported.len()
                    );
                }
            }
        }
    }
    let mut reasons: Vec<_> = reasons.into_iter().collect();
    reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
    println!("== unsupported / errors (count) ==");
    for (r, n) in reasons.iter().take(60) {
        println!("{n:5}  {r}");
    }
    println!("== failures ==");
    for f in &failing_files {
        println!("{f}");
    }
    if !interp_only_failures.is_empty() {
        println!("== failures in interpreter-only files ==");
        for f in &interp_only_failures {
            println!("{f}");
        }
    }
    println!(
        "== {} files, {clean_files} fully supported; {passed} runs passed, {failed} failed ==",
        paths.len()
    );
}

fn truncate(s: &str) -> String {
    // Collapse value numbers so that similar reasons group together.
    let s: String = s.chars().take(90).collect();
    s
}
