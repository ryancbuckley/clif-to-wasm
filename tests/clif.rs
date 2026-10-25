mod harness;

use clif_to_wasm::Config;
use std::path::Path;

fn configs() -> Vec<(&'static str, Config)> {
    vec![
        ("default", Config::default()),
        ("no-tee", harness::config(|c| c.opts.use_tee = false)),
        ("no-coalesce", harness::config(|c| c.opts.coalesce = false)),
        ("memory64", harness::config(|c| c.module.memory64 = true)),
        (
            "demote",
            harness::config(|c| c.module.demote_multi_results = true),
        ),
    ]
}

/// Every `.clif` file under `tests/clif` must compile and pass all its run
/// commands under every configuration.
#[test]
fn clif_runtests() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/clif");
    let filter = std::env::var("CLIF_FILTER").unwrap_or_default();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "clif"))
        .filter(|p| p.to_string_lossy().contains(&filter))
        .collect();
    paths.sort();
    let mut problems = Vec::new();
    let mut passed = 0;
    for path in &paths {
        let text = std::fs::read_to_string(path).unwrap();
        let file = path.file_name().unwrap().to_string_lossy();
        for (cname, config) in configs() {
            match harness::run_clif(&text, &config) {
                Err(e) => problems.push(format!("{file} [{cname}]: {e}")),
                Ok(out) => {
                    passed += out.passed;
                    for u in out.unsupported {
                        problems.push(format!("{file} [{cname}]: unsupported: {u}"));
                    }
                    for f in out.failures {
                        problems.push(format!("{file} [{cname}]: {f}"));
                    }
                }
            }
        }
    }
    println!("{passed} run commands passed");
    if !problems.is_empty() {
        panic!("{} problems:\n{}", problems.len(), problems.join("\n"));
    }
}

/// Narrow results of a `try_call` with several results are masked, as for
/// `call` (in `calls.clif`). Not in `exceptions.clif`, because
/// `demote_multi_results` does not support such a `try_call`.
#[test]
fn try_call_masks_several_narrow_results() {
    let text = "
test run

function %two_sext() -> i8 sext, i16 sext tail {
block0:
    v0 = iconst.i8 -1
    v1 = iconst.i16 -32768
    return v0, v1
}

function %try_uextend_two_sext() -> i32, i32 {
    sig0 = () -> i8 sext, i16 sext tail
    fn0 = %two_sext() -> i8 sext, i16 sext tail
block0:
    try_call fn0(), sig0, block1(ret0, ret1), [ default: block2 ]

block1(v0: i8, v1: i16):
    v2 = uextend.i32 v0
    v3 = uextend.i32 v1
    return v2, v3

block2:
    v4 = iconst.i32 0
    return v4, v4
}
; run: %try_uextend_two_sext() == [255, 32768]
";
    for (cname, config) in configs() {
        if config.module.demote_multi_results {
            continue;
        }
        let out = harness::run_clif(text, &config).unwrap();
        assert!(
            out.unsupported.is_empty(),
            "[{cname}] {:?}",
            out.unsupported
        );
        assert!(out.failures.is_empty(), "[{cname}] {:?}", out.failures);
        assert_eq!(out.passed, 1, "[{cname}]");
    }
}
