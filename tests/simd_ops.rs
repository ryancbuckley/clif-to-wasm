//! Differential test of single vector operations: every operation on every
//! 128-bit vector type it accepts, with random inputs, against Cranelift's
//! interpreter.

mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, Function, InstBuilder, Signature, Type, UserFuncName, Value, types,
};
use cranelift_codegen::isa::{CallConv, TargetFrontendConfig};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_interpreter::environment::FunctionStore;
use cranelift_interpreter::interpreter::{Interpreter, InterpreterState};
use cranelift_interpreter::step::ControlFlow;
use cranelift_wasm_backend::Config;
use harness::clifgen::{Rng, arg};

const VTYPES: [Type; 13] = [
    types::I8X16,
    types::I16X8,
    types::I32X4,
    types::I64X2,
    types::F32X4,
    types::F64X2,
    // Narrower than 128 bits.
    types::I8X8,
    types::I16X4,
    types::I32X2,
    types::F32X2,
    types::I8X4,
    types::I16X2,
    types::I8X2,
];

type Build = fn(&mut FunctionBuilder, &[Value], Type) -> Value;
type Op = (&'static str, fn(Type) -> Vec<Type>, Build);

fn op(name: &'static str, params: fn(Type) -> Vec<Type>, f: Build) -> Op {
    (name, params, f)
}

/// (name, parameter types for vector type `ty`, builder)
fn ops() -> Vec<Op> {
    fn same2(t: Type) -> Vec<Type> {
        vec![t, t]
    }
    fn same1(t: Type) -> Vec<Type> {
        vec![t]
    }
    fn shift(t: Type) -> Vec<Type> {
        vec![t, types::I32]
    }
    fn wider2(t: Type) -> Vec<Type> {
        let w = t.lane_type().double_width().unwrap_or(t.lane_type());
        let wt = w.by(t.lane_count() / 2).unwrap_or(t);
        vec![wt, wt]
    }
    fn narrower1(t: Type) -> Vec<Type> {
        let n = t.lane_type().half_width().unwrap_or(t.lane_type());
        vec![n.by(t.lane_count() * 2).unwrap_or(t)]
    }
    fn as_int(t: Type) -> Vec<Type> {
        vec![t.as_int()]
    }
    fn as_float(t: Type) -> Vec<Type> {
        let f = match t.lane_bits() {
            32 => types::F32X4,
            64 => types::F64X2,
            _ => t,
        };
        vec![f]
    }
    fn as_float2(t: Type) -> Vec<Type> {
        let f = as_float(t)[0];
        vec![f, f]
    }
    macro_rules! bin {
        ($name:literal, $m:ident) => {
            op($name, same2, |b, a, _| b.ins().$m(a[0], a[1]))
        };
    }
    macro_rules! un {
        ($name:literal, $m:ident) => {
            op($name, same1, |b, a, _| b.ins().$m(a[0]))
        };
    }
    let mut v = vec![
        bin!("iadd", iadd),
        bin!("isub", isub),
        bin!("imul", imul),
        bin!("smin", smin),
        bin!("smax", smax),
        bin!("umin", umin),
        bin!("umax", umax),
        bin!("uadd_sat", uadd_sat),
        bin!("sadd_sat", sadd_sat),
        bin!("usub_sat", usub_sat),
        bin!("ssub_sat", ssub_sat),
        bin!("avg_round", avg_round),
        bin!("sqmul_round_sat", sqmul_round_sat),
        bin!("umulhi", umulhi),
        bin!("smulhi", smulhi),
        bin!("iadd_pairwise", iadd_pairwise),
        bin!("band", band),
        bin!("bor", bor),
        bin!("bxor", bxor),
        bin!("fadd", fadd),
        bin!("fsub", fsub),
        bin!("fmul", fmul),
        bin!("fdiv", fdiv),
        bin!("fmin", fmin),
        bin!("fmax", fmax),
        bin!("fcopysign", fcopysign),
        bin!("swizzle", swizzle),
        un!("ineg", ineg),
        un!("iabs", iabs),
        un!("popcnt", popcnt),
        un!("bnot", bnot),
        un!("fneg", fneg),
        un!("fabs", fabs),
        un!("sqrt", sqrt),
        un!("ceil", ceil),
        un!("floor", floor),
        un!("trunc", trunc),
        un!("nearest", nearest),
        un!("vany_true", vany_true),
        un!("vall_true", vall_true),
        op("vhigh_bits", same1, |b, a, _| {
            b.ins().vhigh_bits(types::I32, a[0])
        }),
        op("vhigh_bits.i64", same1, |b, a, _| {
            b.ins().vhigh_bits(types::I64, a[0])
        }),
        op("vhigh_bits.i8", same1, |b, a, _| {
            b.ins().vhigh_bits(types::I8, a[0])
        }),
        op("ishl", shift, |b, a, _| b.ins().ishl(a[0], a[1])),
        op("ushr", shift, |b, a, _| b.ins().ushr(a[0], a[1])),
        op("sshr", shift, |b, a, _| b.ins().sshr(a[0], a[1])),
        op("rotl", shift, |b, a, _| b.ins().rotl(a[0], a[1])),
        op("rotr", shift, |b, a, _| b.ins().rotr(a[0], a[1])),
        op(
            "bitselect",
            |t| vec![t, t, t],
            |b, a, _| b.ins().bitselect(a[0], a[1], a[2]),
        ),
        op("snarrow", wider2, |b, a, _| b.ins().snarrow(a[0], a[1])),
        op("unarrow", wider2, |b, a, _| b.ins().unarrow(a[0], a[1])),
        op("uunarrow", wider2, |b, a, _| b.ins().uunarrow(a[0], a[1])),
        op("swiden_low", narrower1, |b, a, _| b.ins().swiden_low(a[0])),
        op("swiden_high", narrower1, |b, a, _| {
            b.ins().swiden_high(a[0])
        }),
        op("uwiden_low", narrower1, |b, a, _| b.ins().uwiden_low(a[0])),
        op("uwiden_high", narrower1, |b, a, _| {
            b.ins().uwiden_high(a[0])
        }),
        op("fcvt_from_sint", as_int, |b, a, t| {
            b.ins().fcvt_from_sint(t, a[0])
        }),
        op("fcvt_from_uint", as_int, |b, a, t| {
            b.ins().fcvt_from_uint(t, a[0])
        }),
        op("fcvt_to_sint_sat", as_float, |b, a, t| {
            b.ins().fcvt_to_sint_sat(t, a[0])
        }),
        op("fcvt_to_uint_sat", as_float, |b, a, t| {
            b.ins().fcvt_to_uint_sat(t, a[0])
        }),
        op(
            "fvdemote",
            |_| vec![types::F64X2],
            |b, a, _| b.ins().fvdemote(a[0]),
        ),
        op(
            "fvpromote_low",
            |_| vec![types::F32X4],
            |b, a, _| b.ins().fvpromote_low(a[0]),
        ),
        op(
            "splat",
            |t| vec![t.lane_type()],
            |b, a, t| b.ins().splat(t, a[0]),
        ),
        op("extractlane", same1, |b, a, t| {
            b.ins().extractlane(a[0], (t.lane_count() - 1) as u8)
        }),
        op(
            "insertlane",
            |t| vec![t, t.lane_type()],
            |b, a, _| b.ins().insertlane(a[0], a[1], 1),
        ),
    ];
    for cc in IntCC::all() {
        let f: Build = match cc {
            IntCC::Equal => |b, a, _| b.ins().icmp(IntCC::Equal, a[0], a[1]),
            IntCC::NotEqual => |b, a, _| b.ins().icmp(IntCC::NotEqual, a[0], a[1]),
            IntCC::SignedLessThan => |b, a, _| b.ins().icmp(IntCC::SignedLessThan, a[0], a[1]),
            IntCC::SignedGreaterThanOrEqual => {
                |b, a, _| b.ins().icmp(IntCC::SignedGreaterThanOrEqual, a[0], a[1])
            }
            IntCC::SignedGreaterThan => {
                |b, a, _| b.ins().icmp(IntCC::SignedGreaterThan, a[0], a[1])
            }
            IntCC::SignedLessThanOrEqual => {
                |b, a, _| b.ins().icmp(IntCC::SignedLessThanOrEqual, a[0], a[1])
            }
            IntCC::UnsignedLessThan => |b, a, _| b.ins().icmp(IntCC::UnsignedLessThan, a[0], a[1]),
            IntCC::UnsignedGreaterThanOrEqual => {
                |b, a, _| b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, a[0], a[1])
            }
            IntCC::UnsignedGreaterThan => {
                |b, a, _| b.ins().icmp(IntCC::UnsignedGreaterThan, a[0], a[1])
            }
            IntCC::UnsignedLessThanOrEqual => {
                |b, a, _| b.ins().icmp(IntCC::UnsignedLessThanOrEqual, a[0], a[1])
            }
        };
        v.push(op("icmp", same2, f));
    }
    macro_rules! fcmp {
        ($($cc:ident),*) => {
            $(v.push(op("fcmp", as_float2, |b, a, _| b.ins().fcmp(FloatCC::$cc, a[0], a[1])));)*
        };
    }
    fcmp!(
        Ordered,
        Unordered,
        Equal,
        NotEqual,
        OrderedNotEqual,
        UnorderedOrEqual,
        LessThan,
        LessThanOrEqual,
        GreaterThan,
        GreaterThanOrEqual,
        UnorderedOrLessThan,
        UnorderedOrLessThanOrEqual,
        UnorderedOrGreaterThan,
        UnorderedOrGreaterThanOrEqual
    );
    v
}

fn build(params: &[Type], ty: Type, f: Build) -> Option<Function> {
    let mut func = Function::with_name_signature(
        UserFuncName::testcase("f"),
        Signature::new(CallConv::SystemV),
    );
    let mut ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut func, &mut ctx);
    let entry = b.create_block();
    for &p in params {
        b.func.signature.params.push(AbiParam::new(p));
    }
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let args = b.block_params(entry).to_vec();
    let r = f(&mut b, &args, ty);
    let rty = b.func.dfg.value_type(r);
    b.func.signature.returns.push(AbiParam::new(rty));
    b.ins().return_(&[r]);
    b.seal_all_blocks();
    b.finalize(TargetFrontendConfig {
        default_call_conv: CallConv::SystemV,
        pointer_width: target_lexicon::PointerWidth::U64,
        page_size_align_log2: 12,
    });
    let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
    cranelift_codegen::verify_function(&func, &flags).ok()?;
    Some(func)
}

#[test]
fn simd_ops_match_interpreter() {
    let iters: usize = std::env::var("SIMD_OPS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    std::panic::set_hook(Box::new(|_| {}));
    let mut failures = Vec::new();
    let (mut tested, mut unsupported, mut skipped) = (0, Vec::new(), Vec::new());
    for (name, params_of, f) in ops() {
        for ty in VTYPES {
            let params = params_of(ty);
            // Building may panic on type combinations the builder rejects.
            let Ok(Some(func)) = std::panic::catch_unwind(|| build(&params, ty, f)) else {
                continue;
            };
            let (wasm, unsup) =
                harness::compile_module(std::slice::from_ref(&func), &Config::default());
            if !unsup.is_empty() {
                unsupported.push(format!("{name} {ty}"));
                continue;
            }
            harness::validate(&wasm).unwrap_or_else(|e| panic!("{name} {ty}: {e}"));
            let runner = harness::Runner::new(&wasm).unwrap();
            let returns: Vec<Type> = func
                .signature
                .returns
                .iter()
                .map(|p| p.value_type)
                .collect();
            let mut rng = Rng(0x5eed ^ tested as u64);
            for _ in 0..iters {
                let args: Vec<DataValue> = params.iter().map(|&t| arg(&mut rng, t)).collect();
                let mut store = FunctionStore::default();
                store.add("%f".into(), &func);
                let mut interp =
                    Interpreter::new(InterpreterState::default().with_function_store(store));
                let expected = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    interp.call_by_name("%f", &args)
                })) {
                    Ok(Ok(ControlFlow::Return(v))) => Some(v.to_vec()),
                    Ok(Ok(ControlFlow::Trap(_))) => None,
                    _ => {
                        skipped.push(format!("{name} {ty}"));
                        break;
                    }
                };
                let got = runner.call("f", &args, &returns).unwrap();
                let ok = match (&expected, &got) {
                    (None, None) => true,
                    (Some(e), Some(g)) => harness::same_results(e, g, &returns),
                    _ => false,
                };
                if !ok {
                    failures.push(format!(
                        "{name} {ty}: args {args:?}\n  expected {expected:?}\n  got      {got:?}"
                    ));
                    break;
                }
            }
            tested += 1;
        }
    }
    let _ = std::panic::take_hook();
    println!("{tested} op/type pairs tested");
    println!("unsupported by the backend: {unsupported:?}");
    println!("skipped (interpreter): {skipped:?}");
    if !failures.is_empty() {
        panic!("{} failures:\n{}", failures.len(), failures.join("\n"));
    }
}
