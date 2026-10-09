mod harness;

use cranelift_wasm_backend::Config;
use std::path::Path;

fn configs() -> Vec<(&'static str, Config)> {
    vec![
        ("default", Config::default()),
        ("no-tee", harness::config(|c| c.opts.use_tee = false)),
        ("no-coalesce", harness::config(|c| c.opts.coalesce = false)),
        ("memory64", harness::config(|c| c.module.memory64 = true)),
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
