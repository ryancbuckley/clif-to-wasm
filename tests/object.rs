//! Relocatable objects written through `WasmObjectModule`, linked with
//! `wasm-ld` (rustup's `rust-lld`) and run in wasmtime.
//!
//! Two objects reference each other's functions and data; together they
//! cover calls, `func_addr` + `call_indirect`, data relocations (data and
//! function pointers), stack slots (`__stack_pointer`), demoted multi-value
//! returns and argument extension. The test is skipped if `rust-lld` cannot
//! be found (set `RUST_LLD` to point at it).

use cranelift_codegen::ir::{
    AbiParam, ArgumentExtension, InstBuilder, MemFlagsData, Signature, StackSlotData,
    StackSlotKind, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage, Module};
use cranelift_wasm_backend::Config;
use cranelift_wasm_backend::object_module::{ObjectOptions, WasmObjectModule, frontend_isa};
use std::path::PathBuf;
use std::process::Command;

fn new_module() -> WasmObjectModule {
    let flags = settings::Flags::new(settings::builder());
    let isa = frontend_isa(flags, false).unwrap();
    let mut config = Config::default();
    config.module.demote_multi_results = true;
    WasmObjectModule::new(
        isa,
        ObjectOptions {
            config,
            threads: false,
            features: vec!["bulk-memory".into(), "sign-ext".into()],
            stub_unsupported: false,
            libcall_names: cranelift_module::default_libcall_names(),
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
    m.define_function(id, &mut ctx).unwrap();
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
