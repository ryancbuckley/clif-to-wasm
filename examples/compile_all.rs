//! Compile every `.clif` file under a directory (recursively) through the
//! wasm backend under each test configuration and validate the output,
//! without running anything.
//!
//! Each function that passes Cranelift's verifier is compiled in a module of
//! its own, with trapping stubs for its callees: filetests often reuse names
//! and call functions they never define.
//!
//! Usage: cargo run --example compile_all -- <filetests dir> [-v]
//! (`SHOW_PANICS=1` prints panic messages with their locations.)

#[path = "../tests/harness/mod.rs"]
mod harness;

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{ExternalName, Function, InstBuilder, TrapCode, UserFuncName};
use cranelift_codegen::settings;
use cranelift_reader::ParseOptions;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|e| e == "clif") {
            out.push(p);
        }
    }
}

/// `f` followed by trapping stubs for the functions it calls.
fn with_stubs(f: &Function) -> Vec<Function> {
    let mut defined = BTreeSet::from([harness::func_name(f)]);
    let mut out = vec![f.clone()];
    for ext in f.dfg.ext_funcs.values() {
        let name = match &ext.name {
            ExternalName::TestCase(t) => UserFuncName::Testcase(t.clone()),
            ExternalName::User(r) => UserFuncName::User(f.params.user_named_funcs()[*r].clone()),
            _ => continue,
        };
        // The parser leaves placeholder entries for gaps in `fnN` numbering.
        let Some(sig) = f.dfg.signatures.get(ext.signature).cloned() else {
            continue;
        };
        let mut stub = Function::with_name_signature(name, sig.clone());
        if !defined.insert(harness::func_name(&stub)) {
            continue;
        }
        let block = stub.dfg.make_block();
        for p in &sig.params {
            stub.dfg.append_block_param(block, p.value_type);
        }
        stub.layout.append_block(block);
        FuncCursor::new(&mut stub)
            .at_bottom(block)
            .ins()
            .trap(TrapCode::INTEGER_OVERFLOW);
        out.push(stub);
    }
    out
}

fn configs() -> Vec<(&'static str, clif_to_wasm::Config)> {
    vec![
        ("default", clif_to_wasm::Config::default()),
        ("no-tee", harness::config(|c| c.opts.use_tee = false)),
        ("no-coalesce", harness::config(|c| c.opts.coalesce = false)),
        (
            "memory64",
            harness::config(|c| c.module.address_type = clif_to_wasm::AddressType::I64),
        ),
        (
            "demote",
            harness::config(|c| c.module.demote_multi_results = true),
        ),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let verbose = args.iter().any(|a| a == "-v");
    let mut paths = Vec::new();
    walk(root, &mut paths);
    paths.sort();

    if std::env::var_os("SHOW_PANICS").is_none() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let flags = settings::Flags::new(settings::builder());
    let (mut funcs_total, mut clean_funcs, mut unverified) = (0, 0, 0);
    let mut parse_errors = Vec::new();
    let mut bugs = Vec::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for path in &paths {
        let file = path.strip_prefix(root).unwrap().display().to_string();
        // Drop ISA and settings lines, as in `upstream.rs`.
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
        let test = match cranelift_reader::parse_test(&text, ParseOptions::default()) {
            Ok(t) => t,
            Err(e) => {
                parse_errors.push(format!("{file}: {e}"));
                continue;
            }
        };
        for (func, _) in &test.functions {
            // Like Cranelift, compile only verified functions (`parser/` and
            // `verifier/` hold deliberately invalid IR).
            if cranelift_codegen::verify_function(func, &flags).is_err() {
                unverified += 1;
                continue;
            }
            funcs_total += 1;
            let name = harness::func_name(func);
            let module = with_stubs(func);
            let mut clean = true;
            for (cname, config) in configs() {
                let at = format!("{file} %{name} [{cname}]");
                match std::panic::catch_unwind(|| harness::compile_module(&module, &config)) {
                    Err(p) => {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        bugs.push(format!("{at}: PANIC {msg}"));
                        clean = false;
                    }
                    Ok((wasm, unsupported)) => {
                        if let Err(e) = harness::validate(&wasm) {
                            let first = e.lines().next().unwrap_or("");
                            bugs.push(format!("{at}: INVALID {first}"));
                            clean = false;
                        }
                        // Only the function itself, not its stubs.
                        let prefix = format!("{name}: ");
                        let own = unsupported.iter().filter(|u| u.starts_with(&prefix));
                        for u in own {
                            let reason = u.split_once(": ").map(|x| x.1).unwrap_or(u);
                            let reason: String = reason.chars().take(90).collect();
                            *reasons.entry(format!("[{cname}] {reason}")).or_default() += 1;
                            if verbose {
                                println!("{at}: {reason}");
                            }
                            clean = false;
                        }
                    }
                }
            }
            clean_funcs += clean as usize;
        }
    }

    let mut reasons: Vec<_> = reasons.into_iter().collect();
    reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
    println!("== unsupported (count of functions) ==");
    for (r, n) in &reasons {
        println!("{n:5}  {r}");
    }
    println!("== panics / invalid wasm ==");
    for b in &bugs {
        println!("{b}");
    }
    println!("== parse errors ==");
    for e in &parse_errors {
        println!("{e}");
    }
    println!(
        "== {} files, {funcs_total} verified functions ({unverified} failing the verifier skipped), \
         {clean_funcs} compile cleanly under every config; {} panics/invalid, {} parse errors ==",
        paths.len(),
        bugs.len(),
        parse_errors.len()
    );
}
