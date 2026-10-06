#![allow(dead_code)]
//! Runs CLIF filetests (`; run:` directives) through the wasm backend and
//! executes the result with wasmtime.

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{Function, Type, UserFuncName, immediates::*, types};
use cranelift_reader::{Comparison, ParseOptions, RunCommand, parse_run_command, parse_test};
use cranelift_wasm_backend::{
    Config, Env, Error, ModuleBuilder, Name, compile_function, wasm_signature,
};
use std::collections::HashMap;
use wasm_encoder::{ConstExpr, ValType};
use wasmtime::Val;

pub const MEMORY_PAGES: u64 = 32;
pub const STACK_TOP: i32 = MEMORY_PAGES as i32 * 65536;

struct TestEnv<'a> {
    names: &'a HashMap<String, u32>,
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

pub fn compile_module(funcs: &[Function], config: &Config) -> (Vec<u8>, Vec<String>) {
    let mut builder = ModuleBuilder::new();
    let mut names = HashMap::new();
    let mut indices = Vec::new();
    let mut unsupported = Vec::new();
    for f in funcs {
        let (p, r) = wasm_signature(&f.signature).unwrap_or_default();
        let idx = builder.declare_function(&p, &r);
        names.insert(func_name(f), idx);
        indices.push(idx);
    }
    builder.memory(MEMORY_PAGES, config.memory64);
    builder.function_table();
    let sp = builder.global(
        if config.memory64 {
            ValType::I64
        } else {
            ValType::I32
        },
        if config.memory64 {
            ConstExpr::i64_const(STACK_TOP as i64)
        } else {
            ConstExpr::i32_const(STACK_TOP)
        },
    );
    let config = Config {
        stack_pointer_global: Some(sp),
        ..config.clone()
    };
    for (f, &idx) in funcs.iter().zip(&indices) {
        let mut env = TestEnv {
            names: &names,
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
        cfg.wasm_tail_call(true).wasm_memory64(true);
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
        let mut store = wasmtime::Store::new(&self.engine, ());
        let instance =
            wasmtime::Instance::new(&mut store, &self.module, &[]).map_err(|e| e.to_string())?;
        let f = instance
            .get_func(&mut store, name)
            .ok_or_else(|| format!("no export {name}"))?;
        let args = args
            .iter()
            .map(to_val)
            .collect::<Option<Vec<_>>>()
            .ok_or("argument type")?;
        let mut results: Vec<Val> = returns.iter().map(|_| Val::I32(0)).collect();
        if f.call(&mut store, &args, &mut results).is_err() {
            return Ok(None);
        }
        results
            .iter()
            .zip(returns)
            .map(|(v, &ty)| from_val(v, ty))
            .collect::<Result<_, _>>()
            .map(Some)
    }
}

fn to_val(v: &DataValue) -> Option<Val> {
    Some(match *v {
        DataValue::I8(x) => Val::I32(x as u8 as i32),
        DataValue::I16(x) => Val::I32(x as u16 as i32),
        DataValue::I32(x) => Val::I32(x),
        DataValue::I64(x) => Val::I64(x),
        DataValue::F32(x) => Val::F32(x.bits()),
        DataValue::F64(x) => Val::F64(x.bits()),
        _ => return None,
    })
}

fn from_val(v: &Val, ty: Type) -> Result<DataValue, String> {
    Ok(match (v, ty) {
        (Val::I32(x), types::I8) if (*x as u32) <= 0xff => DataValue::I8(*x as i8),
        (Val::I32(x), types::I16) if (*x as u32) <= 0xffff => DataValue::I16(*x as i16),
        (Val::I32(x), types::I8 | types::I16) => {
            return Err(format!("{ty} result {x:#x} is not zero-extended"));
        }
        (Val::I32(x), _) => DataValue::I32(*x),
        (Val::I64(x), _) => DataValue::I64(*x),
        (Val::F32(x), _) => DataValue::F32(Ieee32::with_bits(*x)),
        (Val::F64(x), _) => DataValue::F64(Ieee64::with_bits(*x)),
        _ => return Err(format!("unexpected result {v:?}")),
    })
}

/// Compare results bitwise, but treat all NaNs as equal like Cranelift's
/// runtests do.
pub fn same_results(a: &[DataValue], b: &[DataValue]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| match (a, b) {
            (DataValue::F32(x), DataValue::F32(y)) => {
                x.bits() == y.bits() || (x.is_nan() && y.is_nan())
            }
            (DataValue::F64(x), DataValue::F64(y)) => {
                x.bits() == y.bits() || (x.is_nan() && y.is_nan())
            }
            _ => a == b,
        })
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
            let target = &inv.func;
            let Some(callee) = funcs.iter().find(|f| func_name(f) == *target) else {
                return Err(format!("no function {target}"));
            };
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
                    if same_results(&got, &expected) == want_equal {
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
