#![allow(dead_code)]
//! Runs CLIF filetests (`; run:` directives) through the wasm backend and
//! executes the result with wasmtime.

pub mod clifgen;
pub mod kernels;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{
    ExceptionTableItem, ExceptionTag, ExternalName, Function, Type, UserFuncName, immediates::*,
    types,
};
use cranelift_reader::{Comparison, ParseOptions, RunCommand, parse_run_command, parse_test};
use cranelift_wasm_backend::{
    Config, Env, Error, ModuleBuilder, Name, compile_function, wasm_signature,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use wasm_encoder::{ConstExpr, Encode, Instruction, ValType};
use wasmtime::Val;

pub const MEMORY_PAGES: u64 = 32;
pub const STACK_TOP: i32 = MEMORY_PAGES as i32 * 65536;

struct TestEnv<'a> {
    names: &'a HashMap<String, u32>,
    /// Wasm tag index of each Cranelift exception tag number.
    tags: &'a HashMap<u32, u32>,
    builder: &'a mut ModuleBuilder,
}

impl Env for TestEnv<'_> {
    fn func_index(&mut self, name: Name<'_>) -> Result<u32, Error> {
        self.lookup(name)
    }
    fn type_index(&mut self, sig: &cranelift_codegen::ir::Signature) -> Result<u32, Error> {
        let (p, r) = wasm_signature(sig)?;
        Ok(self.builder.type_index(&p, &r))
    }
    fn func_table_slot(&mut self, name: Name<'_>) -> Result<u64, Error> {
        self.lookup(name).map(u64::from)
    }
    fn exception_tag(&mut self, tag: ExceptionTag) -> Result<u32, Error> {
        self.tags
            .get(&tag.as_u32())
            .copied()
            .ok_or_else(|| Error::Env(format!("unknown exception {tag}")))
    }
}

impl TestEnv<'_> {
    fn lookup(&self, name: Name<'_>) -> Result<u32, Error> {
        let key = match name {
            Name::TestCase(s) => s.to_string(),
            other => other.to_string(),
        };
        self.names
            .get(&key)
            .copied()
            .ok_or_else(|| Error::Env(format!("unknown function {key}")))
    }
}

/// The default configuration, changed by `f`, e.g.
/// `config(|c| c.opts.use_tee = false)`.
pub fn config(f: impl FnOnce(&mut Config)) -> Config {
    let mut c = Config::default();
    f(&mut c);
    c
}

pub fn func_name(f: &Function) -> String {
    match &f.name {
        UserFuncName::Testcase(t) => String::from_utf8_lossy(t.raw()).into_owned(),
        UserFuncName::User(u) => format!("u{}:{}", u.namespace, u.index),
    }
}

#[derive(Default, Debug)]
pub struct Outcome {
    pub passed: usize,
    /// Run commands that gave the wrong result.
    pub failures: Vec<String>,
    /// Functions the backend refused to compile.
    pub unsupported: Vec<String>,
}

/// CLIF implementations of libcalls, added to a module that calls them. In
/// a real embedding these would be runtime functions provided by the host.
const LIBCALLS: &str = "
function %CeilF32(f32) -> f32 { block0(v0: f32): v1 = ceil v0  return v1 }
function %CeilF64(f64) -> f64 { block0(v0: f64): v1 = ceil v0  return v1 }
function %FloorF32(f32) -> f32 { block0(v0: f32): v1 = floor v0  return v1 }
function %FloorF64(f64) -> f64 { block0(v0: f64): v1 = floor v0  return v1 }
function %TruncF32(f32) -> f32 { block0(v0: f32): v1 = trunc v0  return v1 }
function %TruncF64(f64) -> f64 { block0(v0: f64): v1 = trunc v0  return v1 }
function %NearestF32(f32) -> f32 { block0(v0: f32): v1 = nearest v0  return v1 }
function %NearestF64(f64) -> f64 { block0(v0: f64): v1 = nearest v0  return v1 }
";

/// The libcall implementations that `funcs` call.
fn libcalls_used(funcs: &[Function]) -> Vec<Function> {
    let used: Vec<String> = funcs
        .iter()
        .flat_map(|f| f.dfg.ext_funcs.values())
        .filter_map(|ext| match &ext.name {
            ExternalName::LibCall(l) => Some(format!("{l:?}")),
            _ => None,
        })
        .collect();
    cranelift_reader::parse_functions(LIBCALLS)
        .unwrap()
        .into_iter()
        .filter(|f| used.contains(&func_name(f)))
        .collect()
}

/// Exceptions: tests catch them with `try_call`, and throw them by calling
/// `%throw_tag<N>(i64, i64)`, which the harness defines as a wasm `throw`
/// of tag N with that payload (the payload of Cranelift's calling
/// conventions on a 64-bit target). Returns the Cranelift tag numbers used
/// and the tag number of each thrower function.
fn exceptions_used(funcs: &[&Function]) -> (BTreeSet<u32>, BTreeMap<String, u32>) {
    let defined: Vec<String> = funcs.iter().map(|f| func_name(f)).collect();
    let mut tags = BTreeSet::new();
    let mut throwers = BTreeMap::new();
    for f in funcs {
        for table in f.dfg.exception_tables.values() {
            for item in table.items() {
                if let ExceptionTableItem::Tag(tag, _) = item {
                    tags.insert(tag.as_u32());
                }
            }
        }
        for ext in f.dfg.ext_funcs.values() {
            if let ExternalName::TestCase(t) = &ext.name {
                let name = String::from_utf8_lossy(t.raw()).into_owned();
                if let Some(n) = name.strip_prefix("throw_tag").and_then(|n| n.parse().ok())
                    && !defined.contains(&name)
                {
                    tags.insert(n);
                    throwers.insert(name, n);
                }
            }
        }
    }
    (tags, throwers)
}

pub fn compile_module(funcs: &[Function], config: &Config) -> (Vec<u8>, Vec<String>) {
    let libcalls = libcalls_used(funcs);
    let funcs: Vec<&Function> = funcs.iter().chain(&libcalls).collect();
    let mut builder = ModuleBuilder::new();
    let mut names = HashMap::new();
    let mut indices = Vec::new();
    let mut unsupported = Vec::new();
    for &f in &funcs {
        let (p, r) = wasm_signature(&f.signature).unwrap_or_default();
        let idx = builder.declare_function(&p, &r);
        names.insert(func_name(f), idx);
        indices.push(idx);
    }
    let (used_tags, throwers) = exceptions_used(&funcs);
    let payload = [ValType::I64, ValType::I64];
    let tags: HashMap<u32, u32> = used_tags
        .into_iter()
        .map(|n| (n, builder.tag(&payload)))
        .collect();
    for (name, n) in &throwers {
        let idx = builder.declare_function(&payload, &[]);
        names.insert(name.clone(), idx);
        let mut body = Vec::new();
        for i in [
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::Throw(tags[n]),
            Instruction::End,
        ] {
            i.encode(&mut body);
        }
        builder.define_function(
            idx,
            cranelift_wasm_backend::CompiledFunction {
                params: payload.to_vec(),
                results: vec![],
                locals: vec![],
                body,
            },
        );
    }
    builder.memory(MEMORY_PAGES, config.module.memory64);
    let table = builder.function_table();
    let sp = builder.global(
        if config.module.memory64 {
            ValType::I64
        } else {
            ValType::I32
        },
        if config.module.memory64 {
            ConstExpr::i64_const(STACK_TOP as i64)
        } else {
            ConstExpr::i32_const(STACK_TOP)
        },
    );
    let pinned = builder.global(ValType::I64, ConstExpr::i64_const(0));
    let mut config = config.clone();
    config.module.pinned_reg_global = Some(pinned);
    config.module.table_index = Some(table);
    config.module.stack_pointer_global = Some(sp);
    for (&f, &idx) in funcs.iter().zip(&indices) {
        let mut env = TestEnv {
            names: &names,
            tags: &tags,
            builder: &mut builder,
        };
        let compiled = match compile_function(f, &config, &mut env) {
            Ok(c) => c,
            Err(e) => {
                unsupported.push(format!("{}: {e}", func_name(f)));
                // A stub with the right signature keeps the module valid.
                stub(f)
            }
        };
        builder.define_function(idx, compiled);
        builder.export_function(&func_name(f), idx);
    }
    (builder.finish(), unsupported)
}

fn stub(f: &Function) -> cranelift_wasm_backend::CompiledFunction {
    let (params, results) = wasm_signature(&f.signature).unwrap_or_default();
    cranelift_wasm_backend::CompiledFunction {
        params,
        results,
        locals: vec![],
        body: vec![0x00, 0x0b], // unreachable; end
    }
}

pub fn validate(wasm: &[u8]) -> Result<(), String> {
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(wasm)
        .map(|_| ())
        .map_err(|e| {
            format!(
                "{e}\n{}",
                wasmprinter::print_bytes(wasm).unwrap_or_default()
            )
        })
}

/// Executes a compiled module with wasmtime.
pub struct Runner {
    engine: wasmtime::Engine,
    module: wasmtime::Module,
}

impl Runner {
    pub fn new(wasm: &[u8]) -> Result<Self, String> {
        let mut cfg = wasmtime::Config::new();
        cfg.wasm_tail_call(true)
            .wasm_memory64(true)
            .wasm_threads(true);
        let engine = wasmtime::Engine::new(&cfg).map_err(|e| e.to_string())?;
        let module = wasmtime::Module::new(&engine, wasm).map_err(|e| e.to_string())?;
        Ok(Runner { engine, module })
    }

    /// Call `name` in a fresh instance. `Ok(None)` means it trapped.
    pub fn call(
        &self,
        name: &str,
        args: &[DataValue],
        returns: &[Type],
    ) -> Result<Option<Vec<DataValue>>, String> {
        self.call_timed(name, args, returns).0
    }

    /// Like `call`, also returning the time (in seconds) spent in the call
    /// itself, excluding instantiation.
    pub fn call_timed(
        &self,
        name: &str,
        args: &[DataValue],
        returns: &[Type],
    ) -> (Result<Option<Vec<DataValue>>, String>, f64) {
        let mut elapsed = 0.0;
        let r = self.call_inner(name, args, returns, &mut elapsed);
        (r, elapsed)
    }

    fn call_inner(
        &self,
        name: &str,
        args: &[DataValue],
        returns: &[Type],
        elapsed: &mut f64,
    ) -> Result<Option<Vec<DataValue>>, String> {
        let mut store = wasmtime::Store::new(&self.engine, ());
        let instance =
            wasmtime::Instance::new(&mut store, &self.module, &[]).map_err(|e| e.to_string())?;
        let f = instance
            .get_func(&mut store, name)
            .ok_or_else(|| format!("no export {name}"))?;
        // An i128 travels as two i64s, low half first.
        let mut wargs = Vec::new();
        for a in args {
            match *a {
                DataValue::I128(x) => {
                    wargs.extend([Val::I64(x as i64), Val::I64((x >> 64) as i64)])
                }
                _ => wargs.push(to_val(a).ok_or("argument type")?),
            }
        }
        let width = |t: &Type| if *t == types::I128 { 2 } else { 1 };
        let mut results: Vec<Val> = vec![Val::I32(0); returns.iter().map(width).sum::<usize>()];
        let start = std::time::Instant::now();
        let trapped = f.call(&mut store, &wargs, &mut results).is_err();
        *elapsed = start.elapsed().as_secs_f64();
        if trapped {
            return Ok(None);
        }
        let mut out = Vec::new();
        let mut i = 0;
        for &ty in returns {
            if ty == types::I128 {
                let (Val::I64(lo), Val::I64(hi)) = (&results[i], &results[i + 1]) else {
                    return Err("bad i128 result".into());
                };
                out.push(DataValue::I128(
                    (*lo as u64 as i128) | ((*hi as i128) << 64),
                ));
                i += 2;
            } else {
                out.push(from_val(&results[i], ty)?);
                i += 1;
            }
        }
        Ok(Some(out))
    }
}

fn to_val(v: &DataValue) -> Option<Val> {
    Some(match *v {
        DataValue::I8(x) => Val::I32(x as u8 as i32),
        DataValue::I16(x) => Val::I32(x as u16 as i32),
        DataValue::I32(x) => Val::I32(x),
        DataValue::I64(x) => Val::I64(x),
        DataValue::F16(x) => Val::I32(x.bits().into()),
        DataValue::F32(x) => Val::F32(x.bits()),
        DataValue::F64(x) => Val::F64(x.bits()),
        DataValue::V128(b) => Val::V128(u128::from_le_bytes(b).into()),
        // Narrower vectors occupy the low bytes.
        DataValue::V64(b) => Val::V128(u128::from(u64::from_le_bytes(b)).into()),
        DataValue::V32(b) => Val::V128(u128::from(u32::from_le_bytes(b)).into()),
        DataValue::V16(b) => Val::V128(u128::from(u16::from_le_bytes(b)).into()),
        _ => return None,
    })
}

fn from_val(v: &Val, ty: Type) -> Result<DataValue, String> {
    Ok(match (v, ty) {
        (Val::I32(x), types::I8) if (*x as u32) <= 0xff => DataValue::I8(*x as i8),
        (Val::I32(x), types::I16) if (*x as u32) <= 0xffff => DataValue::I16(*x as i16),
        (Val::I32(x), types::F16) if (*x as u32) <= 0xffff => {
            DataValue::F16(Ieee16::with_bits(*x as u16))
        }
        (Val::I32(x), types::I8 | types::I16 | types::F16) => {
            return Err(format!("{ty} result {x:#x} is not zero-extended"));
        }
        (Val::I32(x), _) => DataValue::I32(*x),
        (Val::I64(x), _) => DataValue::I64(*x),
        (Val::F32(x), _) => DataValue::F32(Ieee32::with_bits(*x)),
        (Val::F64(x), _) => DataValue::F64(Ieee64::with_bits(*x)),
        (Val::V128(x), _) => {
            let b = x.as_u128().to_le_bytes();
            match ty.bytes() {
                8 => DataValue::V64(b[..8].try_into().unwrap()),
                4 => DataValue::V32(b[..4].try_into().unwrap()),
                2 => DataValue::V16(b[..2].try_into().unwrap()),
                _ => DataValue::V128(b),
            }
        }
        _ => return Err(format!("unexpected result {v:?}")),
    })
}

/// Compare results bitwise, but treat all NaNs as equal like Cranelift's
/// runtests do. `types` gives the result types, so that float vectors can be
/// compared lane by lane.
pub fn same_results(a: &[DataValue], b: &[DataValue], types: &[Type]) -> bool {
    fn float_eq(x: u64, y: u64, bits: u32) -> bool {
        let (exp, man) = match bits {
            16 => (0x7c00, 0x03ff),
            32 => (0x7f80_0000, 0x007f_ffff),
            _ => (0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff),
        };
        let nan = |v: u64| v & exp == exp && v & man != 0;
        x == y || (nan(x) && nan(y))
    }
    a.len() == b.len()
        && a.iter().zip(b).enumerate().all(|(i, (a, b))| match (a, b) {
            (DataValue::F32(x), DataValue::F32(y)) => {
                float_eq(x.bits().into(), y.bits().into(), 32)
            }
            (DataValue::F64(x), DataValue::F64(y)) => float_eq(x.bits(), y.bits(), 64),
            (DataValue::F16(x), DataValue::F16(y)) => {
                float_eq(x.bits().into(), y.bits().into(), 16)
            }
            (x, y)
                if types
                    .get(i)
                    .is_some_and(|t| t.is_vector() && t.lane_type().is_float()) =>
            {
                let (Some(x), Some(y)) = (vector_bytes(x), vector_bytes(y)) else {
                    return false;
                };
                let lb = types[i].lane_bits() as usize / 8;
                x.chunks(lb).zip(y.chunks(lb)).all(|(p, q)| {
                    let v = |c: &[u8]| c.iter().rev().fold(0u64, |acc, &b| acc << 8 | b as u64);
                    float_eq(v(p), v(q), lb as u32 * 8)
                })
            }
            _ => a == b,
        })
}

fn vector_bytes(v: &DataValue) -> Option<&[u8]> {
    match v {
        DataValue::V128(b) => Some(b),
        DataValue::V64(b) => Some(b),
        DataValue::V32(b) => Some(b),
        DataValue::V16(b) => Some(b),
        _ => None,
    }
}

pub fn run_clif(text: &str, config: &Config) -> Result<Outcome, String> {
    let test = parse_test(text, ParseOptions::default()).map_err(|e| e.to_string())?;
    let funcs: Vec<Function> = test.functions.iter().map(|(f, _)| f.clone()).collect();
    let (wasm, unsupported) = compile_module(&funcs, config);
    validate(&wasm)?;
    let runner = Runner::new(&wasm)?;
    let mut out = Outcome {
        unsupported,
        ..Default::default()
    };

    for (func, details) in &test.functions {
        let name = func_name(func);
        if out
            .unsupported
            .iter()
            .any(|u| u.starts_with(&format!("{name}:")))
        {
            continue;
        }
        for comment in &details.comments {
            let cmd = match parse_run_command(comment.text, &func.signature) {
                Ok(Some(c)) => c,
                Ok(None) => continue,
                Err(e) => return Err(format!("{name}: bad run command: {e}")),
            };
            let (inv, want_equal, expected) = match &cmd {
                RunCommand::Run(inv, cmp, exp) => {
                    (inv, matches!(cmp, Comparison::Equals), exp.clone())
                }
                RunCommand::Print(inv) => (inv, true, vec![]),
            };
            // Like Cranelift's runner, call the function the command is
            // attached to; the name in the command is informational.
            let target = &name;
            let callee = func;
            let returns: Vec<Type> = callee
                .signature
                .returns
                .iter()
                .map(|p| p.value_type)
                .collect();
            let text = comment.text.trim();
            match runner.call(target, &inv.args, &returns) {
                Err(e) if e == "argument type" => out
                    .unsupported
                    .push(format!("{target}: argument types in `{text}`")),
                Err(e) => out.failures.push(format!("{text}: {e}")),
                // A trap may come from the stub of an unsupported callee.
                Ok(None) if !out.unsupported.is_empty() => out
                    .unsupported
                    .push(format!("{target}: may call an unsupported function")),
                Ok(None) => out.failures.push(format!("{text}: trapped")),
                Ok(Some(got)) => {
                    if same_results(&got, &expected, &returns) == want_equal {
                        out.passed += 1;
                    } else {
                        out.failures.push(format!("{text}: got {got:?}"));
                    }
                }
            }
        }
    }
    Ok(out)
}
