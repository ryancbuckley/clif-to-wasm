//! Tests relocatable objects emitted by [`WasmObjectModule`], linked with
//! `wasm-ld` (`rust-lld`), and executed in Wasmtime.
//!
//! Two objects reference each other's functions and data, covering direct and
//! indirect calls, data and function-pointer relocations, stack slots
//! (`__stack_pointer`), demoted multi-value returns, and argument extension.
//! Linking is skipped if `rust-lld` is unavailable (set `RUST_LLD` to override).

use clif_to_wasm::object_module::{
    self, DefineDataError, ImportNameFn, ObjectOptions, WasmObjectModule,
};
use clif_to_wasm::{AddressType, Config};
use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, InstBuilder, MemFlagsData, Signature, StackSlotData,
    StackSlotKind, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage, Module, ModuleError, ModuleResult};
use std::path::PathBuf;
use std::process::Command;

fn new_module() -> WasmObjectModule {
    module_with_imports(Box::new(|_| None))
}

fn module_with_imports(import_names: Box<ImportNameFn>) -> WasmObjectModule {
    module_with(AddressType::I32, import_names)
}

fn module_with(address_type: AddressType, import_names: Box<ImportNameFn>) -> WasmObjectModule {
    let flags = settings::Flags::new(settings::builder());
    let isa = object_module::frontend_isa(flags, address_type).unwrap();
    let mut config = Config::default();
    config.module.address_type = address_type;
    config.module.demote_multi_results = true;
    WasmObjectModule::new(
        isa,
        ObjectOptions {
            config,
            threads: false,
            features: vec!["bulk-memory".into(), "sign-ext".into()],
            stub_unsupported: false,
            libcall_names: cranelift_module::default_libcall_names(),
            import_names,
        },
    )
}

fn sig(params: &[types::Type], returns: &[types::Type]) -> Signature {
    let mut s = Signature::new(CallConv::SystemV);
    s.params.extend(params.iter().map(|&t| AbiParam::new(t)));
    s.returns.extend(returns.iter().map(|&t| AbiParam::new(t)));
    s
}

fn define(
    m: &mut WasmObjectModule,
    id: cranelift_module::FuncId,
    sig: Signature,
    body: impl FnOnce(&mut WasmObjectModule, &mut FunctionBuilder),
) {
    try_define(m, id, sig, body).unwrap();
}

// `ModuleError` is Cranelift's type; its size is not ours to change.
#[allow(clippy::result_large_err)]
fn try_define(
    m: &mut WasmObjectModule,
    id: cranelift_module::FuncId,
    sig: Signature,
    body: impl FnOnce(&mut WasmObjectModule, &mut FunctionBuilder),
) -> ModuleResult<()> {
    let mut ctx = m.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    body(m, &mut b);
    b.finalize(m.target_config());
    m.define_function(id, &mut ctx)
}

/// Object A: `pair(x) -> (i32, i64)`, `narrow(i8 sext) -> i32`, data
/// `TABLE = [&COUNT, pair]` and `COUNT`, and `run() -> i32`, which calls
/// into object B.
fn object_a() -> Vec<u8> {
    let mut m = new_module();
    let i32t = types::I32;
    let pair_sig = sig(&[i32t], &[i32t, types::I64]);
    let pair = m
        .declare_function("pair", Linkage::Export, &pair_sig)
        .unwrap();
    let mut narrow_sig = sig(&[types::I8], &[i32t]);
    narrow_sig.params[0].extension = ArgumentExtension::Sext;
    let narrow = m
        .declare_function("narrow", Linkage::Hidden, &narrow_sig)
        .unwrap();
    let run = m
        .declare_function("run", Linkage::Export, &sig(&[], &[i32t]))
        .unwrap();
    let triple = m
        .declare_function("triple", Linkage::Import, &sig(&[i32t], &[i32t]))
        .unwrap();
    let count = m
        .declare_data("COUNT", Linkage::Export, true, false)
        .unwrap();
    let base = m
        .declare_data("BASE", Linkage::Import, false, false)
        .unwrap();
    let table = m
        .declare_data("TABLE", Linkage::Local, false, false)
        .unwrap();

    let mut d = DataDescription::new();
    d.define(40u32.to_le_bytes().to_vec().into_boxed_slice());
    d.set_align(4);
    m.define_data(count, &d).unwrap();
    let mut d = DataDescription::new();
    d.define(vec![0; 8].into_boxed_slice());
    d.set_align(4);
    let gv = m.declare_data_in_data(count, &mut d);
    d.write_data_addr(0, gv, 0);
    let f = m.declare_func_in_data(pair, &mut d);
    d.write_function_addr(4, f);
    m.define_data(table, &d).unwrap();

    define(&mut m, pair, pair_sig.clone(), |_, b| {
        let x = b.block_params(b.current_block().unwrap())[0];
        let y = b.ins().iadd_imm_s(x, 1);
        let w = b.ins().uextend(types::I64, x);
        let w = b.ins().imul_imm_s(w, 1 << 33);
        b.ins().return_(&[y, w]);
    });
    define(&mut m, narrow, narrow_sig.clone(), |_, b| {
        // The argument arrives sign-extended (and is re-masked on entry).
        let x = b.block_params(b.current_block().unwrap())[0];
        let x = b.ins().sextend(i32t, x);
        b.ins().return_(&[x]);
    });
    define(&mut m, run, sig(&[], &[i32t]), |m, b| {
        let flags = MemFlagsData::trusted();
        // COUNT through the table's data pointer: 40.
        let table_gv = m.declare_data_in_func(table, b.func);
        let t = b.ins().symbol_value(i32t, table_gv);
        let count_ptr = b.ins().load(i32t, flags, t, 0);
        let c = b.ins().load(i32t, flags, count_ptr, 0);
        // pair(c) through the table's function pointer: (41, 40 << 33).
        let fptr = b.ins().load(i32t, flags, t, 4);
        let sref = b.import_signature(pair_sig.clone());
        let call = b.ins().call_indirect(sref, fptr, &[c]);
        let (y, w) = {
            let r = b.inst_results(call);
            (r[0], r[1])
        };
        let w = b.ins().ushr_imm_s(w, 33);
        let w = b.ins().ireduce(i32t, w);
        let s = b.ins().iadd(y, w); // 81
        // A stack slot round trip.
        let slot = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 2));
        b.ins().stack_store(i32t, s, slot, 4);
        let s = b.ins().stack_load(i32t, i32t, slot, 4);
        // triple(s) in object B: 243.
        let tf = m.declare_func_in_func(triple, b.func);
        let call = b.ins().call(tf, &[s]);
        let s = b.inst_results(call)[0];
        // narrow(-2 as i8) == -2: 241.
        let nf = m.declare_func_in_func(narrow, b.func);
        let k = b.ins().iconst(types::I8, 0xfe);
        let call = b.ins().call(nf, &[k]);
        let n = b.inst_results(call)[0];
        let s = b.ins().iadd(s, n);
        // Plus BASE[0] from object B (1000): 1241.
        let base_gv = m.declare_data_in_func(base, b.func);
        let p = b.ins().symbol_value(i32t, base_gv);
        let v = b.ins().load(i32t, flags, p, 0);
        let s = b.ins().iadd(s, v);
        b.ins().return_(&[s]);
    });
    assert!(m.warnings().is_empty());
    m.finish()
}

/// Object B: `triple(x) = 3 * x` and `BASE = [1000]`.
fn object_b() -> Vec<u8> {
    let mut m = new_module();
    let i32t = types::I32;
    let triple = m
        .declare_function("triple", Linkage::Export, &sig(&[i32t], &[i32t]))
        .unwrap();
    let base = m
        .declare_data("BASE", Linkage::Export, false, false)
        .unwrap();
    let mut d = DataDescription::new();
    d.define(1000u32.to_le_bytes().to_vec().into_boxed_slice());
    d.set_align(4);
    m.define_data(base, &d).unwrap();
    define(&mut m, triple, sig(&[i32t], &[i32t]), |_, b| {
        let x = b.block_params(b.current_block().unwrap())[0];
        let y = b.ins().imul_imm_s(x, 3);
        b.ins().return_(&[y]);
    });
    m.finish()
}

fn rust_lld() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RUST_LLD") {
        return Some(p.into());
    }
    let out = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()?;
    let sysroot = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    let out = Command::new("rustc").arg("-vV").output().ok()?;
    let host = String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_owned))?;
    let p = sysroot.join("lib/rustlib").join(host).join("bin/rust-lld");
    p.exists().then_some(p)
}

#[test]
fn objects_validate() {
    for obj in [object_a(), object_b()] {
        let mut features = wasmparser::WasmFeatures::default();
        features.insert(wasmparser::WasmFeatures::MEMORY64);
        wasmparser::Validator::new_with_features(features)
            .validate_all(&obj)
            .unwrap();
        // The linking and relocation sections parse.
        for payload in wasmparser::Parser::new(0).parse_all(&obj) {
            if let wasmparser::Payload::CustomSection(c) = payload.unwrap() {
                match c.as_known() {
                    wasmparser::KnownCustom::Linking(l) => {
                        for s in l {
                            s.unwrap();
                        }
                    }
                    wasmparser::KnownCustom::Reloc(r) => {
                        for e in r.entries() {
                            e.unwrap();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[test]
fn objects_link_and_run() {
    let Some(lld) = rust_lld() else {
        eprintln!("skipping: rust-lld not found (set RUST_LLD)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cwb-object-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.o");
    let b = dir.join("b.o");
    let out = dir.join("linked.wasm");
    std::fs::write(&a, object_a()).unwrap();
    std::fs::write(&b, object_b()).unwrap();
    let status = Command::new(&lld)
        .args(["-flavor", "wasm", "--no-entry", "--export=run", "-o"])
        .arg(&out)
        .arg(&a)
        .arg(&b)
        .status()
        .unwrap();
    assert!(status.success(), "wasm-ld failed");
    let wasm = std::fs::read(&out).unwrap();

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let run = instance
        .get_typed_func::<(), i32>(&mut store, "run")
        .unwrap();
    assert_eq!(run.call(&mut store, ()).unwrap(), 1241);
    let _ = std::fs::remove_dir_all(&dir);
}

/// An object calling `random_get` from `wasi_snapshot_preview1` under a
/// mangled symbol name, as Rust's `#[link(wasm_import_module = "...")]`
/// declares it.
fn object_with_import() -> Vec<u8> {
    const SYMBOL: &str = "_RNvNtCs1_9getrandom7wasi_p110random_get";
    let mut m = module_with_imports(Box::new(|name| {
        (name == SYMBOL).then(|| ("wasi_snapshot_preview1".into(), "random_get".into()))
    }));
    let i32t = types::I32;
    let random_get = m
        .declare_function(SYMBOL, Linkage::Import, &sig(&[i32t, i32t], &[i32t]))
        .unwrap();
    let run = m
        .declare_function("run", Linkage::Export, &sig(&[], &[i32t]))
        .unwrap();
    define(&mut m, run, sig(&[], &[i32t]), |m, b| {
        let f = m.declare_func_in_func(random_get, b.func);
        let p = b.ins().iconst(i32t, 0);
        let n = b.ins().iconst(i32t, 8);
        let call = b.ins().call(f, &[p, n]);
        let r = b.inst_results(call)[0];
        b.ins().return_(&[r]);
    });
    m.finish()
}

/// The function imports of a wasm module, as (module, field).
fn func_imports(wasm: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let wasmparser::Payload::ImportSection(s) = payload.unwrap() {
            for imp in s.into_imports() {
                let imp = imp.unwrap();
                if matches!(imp.ty, wasmparser::TypeRef::Func(_)) {
                    out.push((imp.module.to_owned(), imp.name.to_owned()));
                }
            }
        }
    }
    out
}

#[test]
fn imports_from_other_modules() {
    let obj = object_with_import();
    wasmparser::Validator::new().validate_all(&obj).unwrap();
    assert_eq!(
        func_imports(&obj),
        [("wasi_snapshot_preview1".to_owned(), "random_get".to_owned())]
    );
    // Its symbol keeps the mangled name, with WASM_SYM_EXPLICIT_NAME.
    let mut found = false;
    for payload in wasmparser::Parser::new(0).parse_all(&obj) {
        if let wasmparser::Payload::CustomSection(c) = payload.unwrap()
            && let wasmparser::KnownCustom::Linking(l) = c.as_known()
        {
            for sub in l {
                if let wasmparser::Linking::SymbolTable(syms) = sub.unwrap() {
                    for sym in syms {
                        if let wasmparser::SymbolInfo::Func { flags, name, .. } = sym.unwrap()
                            && flags.contains(wasmparser::SymbolFlags::UNDEFINED)
                        {
                            assert!(flags.contains(wasmparser::SymbolFlags::EXPLICIT_NAME));
                            assert_eq!(name, Some("_RNvNtCs1_9getrandom7wasi_p110random_get"));
                            found = true;
                        }
                    }
                }
            }
        }
    }
    assert!(found, "no undefined function symbol");

    // A linker keeps the import as it is.
    let Some(lld) = rust_lld() else {
        eprintln!("skipping the link: rust-lld not found (set RUST_LLD)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cwb-import-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let o = dir.join("import.o");
    let out = dir.join("linked.wasm");
    std::fs::write(&o, &obj).unwrap();
    let status = Command::new(&lld)
        .args(["-flavor", "wasm", "--no-entry", "--export=run", "-o"])
        .arg(&out)
        .arg(&o)
        .status()
        .unwrap();
    assert!(status.success(), "wasm-ld failed");
    assert_eq!(
        func_imports(&std::fs::read(&out).unwrap()),
        [("wasi_snapshot_preview1".to_owned(), "random_get".to_owned())]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Object P takes function addresses (`get_seven() = &seven`, data
/// `EIGHT = [eight]`) but never calls through the table, so its unused
/// `__indirect_function_table` symbol is pruned.
fn object_pointers() -> Vec<u8> {
    let mut m = new_module();
    let i32t = types::I32;
    let seven = m
        .declare_function("seven", Linkage::Local, &sig(&[], &[i32t]))
        .unwrap();
    let eight = m
        .declare_function("eight", Linkage::Local, &sig(&[], &[i32t]))
        .unwrap();
    let get_seven = m
        .declare_function("get_seven", Linkage::Export, &sig(&[], &[i32t]))
        .unwrap();
    let eight_ptr = m
        .declare_data("EIGHT", Linkage::Export, false, false)
        .unwrap();
    let mut d = DataDescription::new();
    d.define(vec![0; 4].into_boxed_slice());
    d.set_align(4);
    let f = m.declare_func_in_data(eight, &mut d);
    d.write_function_addr(0, f);
    m.define_data(eight_ptr, &d).unwrap();
    for (id, v) in [(seven, 7), (eight, 8)] {
        define(&mut m, id, sig(&[], &[i32t]), |_, b| {
            let v = b.ins().iconst(i32t, v);
            b.ins().return_(&[v]);
        });
    }
    define(&mut m, get_seven, sig(&[], &[i32t]), |m, b| {
        let f = m.declare_func_in_func(seven, b.func);
        let p = b.ins().func_addr(i32t, f);
        b.ins().return_(&[p]);
    });
    m.finish()
}

/// Object Q: `run() = 10 * get_seven()() + EIGHT[0]()`, through the table.
fn object_indirect_caller() -> Vec<u8> {
    let mut m = new_module();
    let i32t = types::I32;
    let get_seven = m
        .declare_function("get_seven", Linkage::Import, &sig(&[], &[i32t]))
        .unwrap();
    let eight_ptr = m
        .declare_data("EIGHT", Linkage::Import, false, false)
        .unwrap();
    let run = m
        .declare_function("run", Linkage::Export, &sig(&[], &[i32t]))
        .unwrap();
    define(&mut m, run, sig(&[], &[i32t]), |m, b| {
        let sref = b.import_signature(sig(&[], &[i32t]));
        let gf = m.declare_func_in_func(get_seven, b.func);
        let call = b.ins().call(gf, &[]);
        let p = b.inst_results(call)[0];
        let call = b.ins().call_indirect(sref, p, &[]);
        let x = b.inst_results(call)[0];
        let x = b.ins().imul_imm_s(x, 10);
        let gv = m.declare_data_in_func(eight_ptr, b.func);
        let a = b.ins().symbol_value(i32t, gv);
        let p = b.ins().load(i32t, MemFlagsData::trusted(), a, 0);
        let call = b.ins().call_indirect(sref, p, &[]);
        let y = b.inst_results(call)[0];
        let s = b.ins().iadd(x, y);
        b.ins().return_(&[s]);
    });
    m.finish()
}

fn imports_table(wasm: &[u8]) -> bool {
    wasmparser::Parser::new(0).parse_all(wasm).any(|payload| {
        let wasmparser::Payload::ImportSection(s) = payload.unwrap() else {
            return false;
        };
        s.into_imports()
            .any(|imp| matches!(imp.unwrap().ty, wasmparser::TypeRef::Table(_)))
    })
}

#[test]
fn function_pointers_without_table_symbol() {
    let p = object_pointers();
    let q = object_indirect_caller();
    for obj in [&p, &q] {
        wasmparser::Validator::new().validate_all(obj).unwrap();
    }
    assert!(!imports_table(&p), "unused table not pruned");
    assert!(imports_table(&q));

    // The linker still gives P's functions slots in Q's table.
    let Some(lld) = rust_lld() else {
        eprintln!("skipping the link: rust-lld not found (set RUST_LLD)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("cwb-pointers-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let po = dir.join("p.o");
    let qo = dir.join("q.o");
    let out = dir.join("linked.wasm");
    std::fs::write(&po, &p).unwrap();
    std::fs::write(&qo, &q).unwrap();
    let status = Command::new(&lld)
        .args(["-flavor", "wasm", "--no-entry", "--export=run", "-o"])
        .arg(&out)
        .arg(&po)
        .arg(&qo)
        .status()
        .unwrap();
    assert!(status.success(), "wasm-ld failed");
    let wasm = std::fs::read(&out).unwrap();

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let run = instance
        .get_typed_func::<(), i32>(&mut store, "run")
        .unwrap();
    assert_eq!(run.call(&mut store, ()).unwrap(), 78);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn uninitialized_data_is_an_error() {
    let mut m = new_module();
    let id = m
        .declare_data("EMPTY", Linkage::Export, false, false)
        .unwrap();
    let err = m.define_data(id, &DataDescription::new()).unwrap_err();
    let ModuleError::Backend(e) = &err else {
        panic!("expected a backend error, got {err}");
    };
    assert_eq!(
        e.downcast_ref::<DefineDataError>(),
        Some(&DefineDataError::Uninitialized {
            name: "EMPTY".into()
        }),
        "the error must name the uninitialized data object"
    );
    // The failed definition must not count, so a corrected one succeeds.
    let mut d = DataDescription::new();
    d.define_zeroinit(4);
    m.define_data(id, &d)
        .expect("defining after a failed attempt must succeed");
}

#[test]
fn invalid_data_alignment_is_an_error() {
    let mut m = new_module();
    let id = m
        .declare_data("ODD", Linkage::Export, false, false)
        .unwrap();
    for align in [3, 1 << 32] {
        let mut d = DataDescription::new();
        d.define_zeroinit(4);
        // Set directly: `set_align` would reject 3 itself.
        d.align = Some(align);
        let err = m.define_data(id, &d).unwrap_err();
        let ModuleError::Backend(e) = &err else {
            panic!("expected a backend error, got {err}");
        };
        assert_eq!(
            e.downcast_ref::<DefineDataError>(),
            Some(&DefineDataError::InvalidAlignment {
                name: "ODD".into(),
                align
            }),
            "alignment {align} must be rejected"
        );
    }
}

#[test]
fn data_relocation_out_of_bounds_is_an_error() {
    let mut m = new_module();
    let target = m
        .declare_data("TARGET", Linkage::Export, false, false)
        .unwrap();
    let id = m
        .declare_data("SHORT", Linkage::Export, false, false)
        .unwrap();
    // A 4-byte pointer at offset 1 or 4 extends past the 4 bytes of data.
    for offset in [1, 4] {
        let mut d = DataDescription::new();
        d.define_zeroinit(4);
        let gv = m.declare_data_in_data(target, &mut d);
        d.write_data_addr(offset, gv, 0);
        let err = m.define_data(id, &d).unwrap_err();
        let ModuleError::Backend(e) = &err else {
            panic!("expected a backend error, got {err}");
        };
        assert_eq!(
            e.downcast_ref::<DefineDataError>(),
            Some(&DefineDataError::RelocationOutOfBounds {
                name: "SHORT".into(),
                offset
            }),
            "a relocation at offset {offset} must be rejected"
        );
    }
    // The failed definitions did not count: a valid one still succeeds.
    let mut d = DataDescription::new();
    d.define_zeroinit(4);
    let gv = m.declare_data_in_data(target, &mut d);
    d.write_data_addr(0, gv, 0);
    m.define_data(id, &d).unwrap();
}

#[test]
fn failed_function_definition_can_be_retried() {
    let mut m = new_module();
    let i32t = types::I32;
    let s = sig(&[], &[i32t]);
    let id = m.declare_function("f", Linkage::Export, &s).unwrap();
    // This backend does not support `get_return_address`.
    let err = try_define(&mut m, id, s.clone(), |_, b| {
        let r = b.ins().get_return_address(i32t);
        b.ins().return_(&[r]);
    })
    .unwrap_err();
    assert!(
        matches!(err, ModuleError::Compilation(_)),
        "get_return_address must fail to compile, got {err}"
    );
    define(&mut m, id, s, |_, b| {
        let r = b.ins().iconst(i32t, 7);
        b.ins().return_(&[r]);
    });
}

/// The address and bytes of each active data segment, in order.
fn data_segments(wasm: &[u8]) -> Vec<(u64, Vec<u8>)> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let wasmparser::Payload::DataSection(s) = payload.unwrap() {
            for seg in s {
                let seg = seg.unwrap();
                let wasmparser::DataKind::Active { offset_expr, .. } = seg.kind else {
                    panic!("passive data segment");
                };
                let addr = match offset_expr.get_operators_reader().read().unwrap() {
                    wasmparser::Operator::I32Const { value } => value as u32 as u64,
                    wasmparser::Operator::I64Const { value } => value as u64,
                    op => panic!("unexpected segment offset {op:?}"),
                };
                out.push((addr, seg.data.to_vec()));
            }
        }
    }
    out
}

/// Before linking, a relocated field holds the value it would have if the
/// object were a module of its own, as in LLVM's objects. Linkers overwrite
/// these fields, so linking cannot check them.
#[test]
fn unlinked_data_pointers_hold_object_addresses() {
    for address_type in [AddressType::I32, AddressType::I64] {
        let mut m = module_with(address_type, Box::new(|_| None));
        let ptr_size = match address_type {
            AddressType::I32 => 4,
            AddressType::I64 => 8,
        };
        let declare = |m: &mut WasmObjectModule, name| {
            m.declare_data(name, Linkage::Export, false, false).unwrap()
        };
        // `FIRST` gives `TARGET` a nonzero address.
        let first = declare(&mut m, "FIRST");
        let target = declare(&mut m, "TARGET");
        let ptr = declare(&mut m, "PTR");
        for id in [first, target] {
            let mut d = DataDescription::new();
            d.define(vec![1; 8].into_boxed_slice());
            m.define_data(id, &d).unwrap();
        }
        let mut d = DataDescription::new();
        d.define(vec![0; ptr_size].into_boxed_slice());
        d.set_align(ptr_size as u64);
        let gv = m.declare_data_in_data(target, &mut d);
        d.write_data_addr(0, gv, 4);
        m.define_data(ptr, &d).unwrap();
        let obj = m.finish();

        // Segments are written in definition order.
        let segs = data_segments(&obj);
        let [_, (target_addr, _), (_, ptr_bytes)] = &segs[..] else {
            panic!("expected three data segments, got {}", segs.len());
        };
        assert_ne!(*target_addr, 0, "{address_type:?}: TARGET follows FIRST");
        let got = match address_type {
            AddressType::I32 => u32::from_le_bytes(ptr_bytes[..4].try_into().unwrap()).into(),
            AddressType::I64 => u64::from_le_bytes(ptr_bytes[..8].try_into().unwrap()),
        };
        assert_eq!(got, target_addr + 4, "{address_type:?}: PTR = &TARGET + 4");
    }
}
