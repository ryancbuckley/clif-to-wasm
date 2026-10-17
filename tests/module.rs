//! `ModuleBuilder` for a JIT running inside wasm: a "host" module owns the
//! memory, the function table and the shadow stack pointer, and a "JIT"
//! module compiled later imports them and adds its functions to free slots of
//! the host's table.

mod harness;

use cranelift_codegen::ir::{Function, Signature};
use cranelift_reader::parse_functions;
use cranelift_wasm_backend::{
    CompiledFunction, Config, Env, Error, FuncIndex, ModuleBuilder, Name, TableIndex, TypeIndex,
    compile_function, wasm_signature,
};
use std::collections::HashMap;
use wasm_encoder::{ConstExpr, GlobalType, MemoryType, RefType, TableType, ValType};
use wasmtime::{Engine, Instance, Module, Ref, Store};

const HOST: &str = "
function %double(i32) -> i32 {
block0(v0: i32):
    v1 = iadd v0, v0
    return v1
}

; Calls whatever function is in table slot v0.
function %call_slot(i32, i32) -> i32 {
    sig0 = (i32) -> i32
block0(v0: i32, v1: i32):
    v2 = call_indirect sig0, v0(v1)
    return v2
}
";

const JIT: &str = "
function %square(i32) -> i32 {
block0(v0: i32):
    v1 = imul v0, v0
    return v1
}

; square(double(x)) through the shared table, passed through a shadow stack
; slot and the shared memory at p, then doubled again by a direct call to
; the imported host function.
function %jit(i32, i32) -> i32 {
    ss0 = explicit_slot 4
    sig0 = (i32) -> i32
    fn0 = %double(i32) -> i32
    fn1 = %square(i32) -> i32
block0(v0: i32, v1: i32):
    v2 = func_addr.i32 fn0
    v3 = call_indirect sig0, v2(v0)
    v4 = func_addr.i32 fn1
    v5 = call_indirect sig0, v4(v3)
    v9 = stack_addr.i32 ss0
    store v5, v9
    v6 = load.i32 v9
    store v6, v1
    v7 = load.i32 v1
    v8 = call fn0(v7)
    return v8
}
";

const STACK_TOP: i32 = 65536;

/// Resolves names to function indices and table slots.
struct Names<'a> {
    funcs: &'a HashMap<String, FuncIndex>,
    slots: &'a HashMap<String, u64>,
    builder: &'a mut ModuleBuilder,
}

fn key(name: Name<'_>) -> String {
    match name {
        Name::TestCase(s) => s.to_string(),
        other => other.to_string(),
    }
}

impl Env for Names<'_> {
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex, Error> {
        let k = key(name);
        self.funcs.get(&k).copied().ok_or(Error::Env(k))
    }
    fn type_index(&mut self, sig: &Signature) -> Result<TypeIndex, Error> {
        let (p, r) = wasm_signature(sig)?;
        Ok(self.builder.type_index(&p, &r))
    }
    fn func_table_slot(&mut self, name: Name<'_>) -> Result<u64, Error> {
        let k = key(name);
        self.slots.get(&k).copied().ok_or(Error::Env(k))
    }
}

fn name(f: &Function) -> String {
    f.name.to_string().trim_start_matches('%').to_string()
}

/// Declare every function of `funcs`, then compile, define and export them.
/// `slot` gives each declared function's table slot from its index.
fn add_functions(
    builder: &mut ModuleBuilder,
    funcs: &[Function],
    config: &Config,
    mut names: HashMap<String, FuncIndex>,
    mut slots: HashMap<String, u64>,
    slot: impl Fn(usize, FuncIndex) -> u64,
) -> Vec<FuncIndex> {
    let mut indices = Vec::new();
    for (i, f) in funcs.iter().enumerate() {
        let (p, r) = wasm_signature(&f.signature).unwrap();
        let idx = builder.declare_function(&p, &r);
        names.insert(name(f), idx);
        slots.insert(name(f), slot(i, idx));
        indices.push(idx);
    }
    for (f, &idx) in funcs.iter().zip(&indices) {
        let mut env = Names {
            funcs: &names,
            slots: &slots,
            builder,
        };
        let compiled: CompiledFunction = compile_function(f, config, &mut env).unwrap();
        builder.define_function(idx, compiled);
        builder.export_function(&name(f), idx);
    }
    indices
}

#[test]
fn jit_module_links_into_host() {
    // The host: defines memory, table and stack pointer, and exports them.
    let host_funcs = parse_functions(HOST).unwrap();
    let mut host = ModuleBuilder::new();
    let mem = host.memory(1, false);
    let table = host.function_table();
    let sp = host.global(ValType::I32, ConstExpr::i32_const(STACK_TOP));
    host.export("memory", mem);
    host.export("table", table);
    host.export("sp", sp);
    let mut config = Config::default();
    config.module.memory_index = mem;
    config.module.table_index = Some(table);
    config.module.stack_pointer_global = Some(sp);
    // In the host's own table, slot i holds function i.
    let host_indices = add_functions(
        &mut host,
        &host_funcs,
        &config,
        HashMap::new(),
        HashMap::new(),
        |_, idx| idx.0 as u64,
    );
    let host_wasm = host.finish();
    harness::validate(&host_wasm).unwrap();

    let engine = Engine::default();
    let mut store = Store::new(&engine, ());
    let host_inst = Instance::new(&mut store, &Module::new(&engine, &host_wasm).unwrap(), &[])
        .unwrap();
    let memory = host_inst.get_memory(&mut store, "memory").unwrap();
    let table = host_inst.get_table(&mut store, "table").unwrap();
    let sp = host_inst.get_global(&mut store, "sp").unwrap();
    let double = host_inst.get_func(&mut store, "double").unwrap();

    // Later, the JIT grows the table to make room for the new functions.
    let jit_funcs = parse_functions(JIT).unwrap();
    let base = table
        .grow(&mut store, jit_funcs.len() as u64, Ref::Func(None))
        .unwrap();

    // The JIT module imports everything it shares with the host.
    let mut jit = ModuleBuilder::new();
    let i32_ = [ValType::I32];
    let double_idx = jit.import_function("host", "double", &i32_, &i32_);
    let mem = jit.import_memory(
        "host",
        "memory",
        MemoryType {
            minimum: 0,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        },
    );
    let sp_idx = jit.import_global(
        "host",
        "sp",
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
    );
    let table_idx = jit.import_table(
        "host",
        "table",
        TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: 0,
            maximum: None,
            shared: false,
        },
    );
    let mut config = Config::default();
    config.module.memory_index = mem;
    config.module.table_index = Some(table_idx);
    config.module.stack_pointer_global = Some(sp_idx);
    let names = HashMap::from([("double".to_string(), double_idx)]);
    // `double` keeps the slot it has in the host's table.
    let slots = HashMap::from([("double".to_string(), host_indices[0].0 as u64)]);
    let jit_indices = add_functions(&mut jit, &jit_funcs, &config, names, slots, |i, _| {
        base + i as u64
    });
    jit.place_in_table(table_idx, base, &jit_indices);
    let jit_wasm = jit.finish();
    harness::validate(&jit_wasm).unwrap();

    let jit_inst = Instance::new(
        &mut store,
        &Module::new(&engine, &jit_wasm).unwrap(),
        &[double.into(), memory.into(), sp.into(), table.into()],
    )
    .unwrap();

    // JIT code calling host and JIT functions.
    let f = jit_inst
        .get_typed_func::<(i32, i32), i32>(&mut store, "jit")
        .unwrap();
    assert_eq!(f.call(&mut store, (5, 1024)).unwrap(), 200);
    let mut word = [0u8; 4];
    memory.read(&store, 1024, &mut word).unwrap();
    assert_eq!(i32::from_le_bytes(word), 100);
    assert_eq!(sp.get(&mut store).i32(), Some(STACK_TOP));

    // Host code calling JIT functions through the shared table.
    let call_slot = host_inst
        .get_typed_func::<(i32, i32), i32>(&mut store, "call_slot")
        .unwrap();
    assert_eq!(call_slot.call(&mut store, (base as i32, 7)).unwrap(), 49);
    assert!(
        call_slot.call(&mut store, (base as i32 + 1, 3)).is_err(),
        "slot base+1 holds %jit, whose signature differs"
    );
}

/// An environment that only knows table slots.
struct SlotsOnly;

impl Env for SlotsOnly {
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex, Error> {
        Err(Error::Env(key(name)))
    }
    fn type_index(&mut self, _: &Signature) -> Result<TypeIndex, Error> {
        Ok(TypeIndex(0))
    }
    fn func_table_slot(&mut self, _: Name<'_>) -> Result<u64, Error> {
        Ok(7)
    }
}

#[test]
fn call_indirect_needs_a_table() {
    let funcs = parse_functions(
        "
function %call_ptr(i32, i32) -> i32 {
    sig0 = (i32) -> i32
block0(v0: i32, v1: i32):
    v2 = call_indirect sig0, v0(v1)
    return v2
}

function %tail_call_ptr(i32, i32) -> i32 tail {
    sig0 = (i32) -> i32 tail
block0(v0: i32, v1: i32):
    return_call_indirect sig0, v0(v1)
}

function %addr() -> i32 {
    fn0 = %callee(i32) -> i32
block0:
    v0 = func_addr.i32 fn0
    return v0
}
",
    )
    .unwrap();
    let mut config = Config::default();
    assert_eq!(config.module.table_index, None);

    // Calling through a pointer needs to know which table to use.
    for f in &funcs[..2] {
        assert_eq!(
            compile_function(f, &config, &mut SlotsOnly).unwrap_err(),
            Error::Unsupported("call_indirect without ModuleLayout::table_index".into()),
            "{}",
            f.name
        );
    }
    // Taking an address only needs the slot, which the Env provides.
    compile_function(&funcs[2], &config, &mut SlotsOnly).unwrap();

    config.module.table_index = Some(TableIndex(0));
    for f in &funcs {
        compile_function(f, &config, &mut SlotsOnly).unwrap();
    }
}
