//! Tests that vectors narrower than 128 bits (`i32x2`, `i8x4`, ...) ignore
//! junk bits in the unused upper bytes of their `v128` representation.
//!
//! Each operand is constructed by splatting a junk lane across all 128 bits and
//! then inserting the active lanes, ensuring any operation that reads the upper
//! bytes diverges from `cranelift-interpreter`.

mod harness;

use cranelift_codegen::data_value::DataValue;
use cranelift_codegen::ir::{Type, types};
use clif_to_wasm::Config;
use proptest::prelude::*;

const NARROW: [Type; 7] = [
    types::I8X8,
    types::I16X4,
    types::I32X2,
    types::F32X2,
    types::I8X4,
    types::I16X2,
    types::I8X2,
];

/// An operation on one or two vectors `a` and `b`, as CLIF text, with its
/// result types; `None` if it does not apply to `ty`.
fn op(ty: Type, k: usize) -> Option<(String, Vec<Type>)> {
    let int = ty.lane_type().is_int();
    let n = ty.lane_count();
    let lb = ty.lane_bits();
    let half = |t: Type| t.lane_type().half_width().and_then(|l| l.by(n * 2));
    let double = |t: Type| t.lane_type().double_width().and_then(|l| l.by(n / 2));
    let int_of = Type::int(ty.bits() as u16).unwrap();
    Some(match k {
        0 => ("vall_true a".into(), vec![types::I8]),
        1 => ("vany_true a".into(), vec![types::I8]),
        2 if int => ("vhigh_bits.i64 a".into(), vec![types::I64]),
        3 if int && lb <= 32 => ("iadd_pairwise a, b".into(), vec![ty]),
        4 if int && n >= 4 => ("swiden_high a".into(), vec![double(ty)?]),
        5 if int && n >= 4 => ("uwiden_high a".into(), vec![double(ty)?]),
        6 if int && n >= 4 => ("swiden_low a".into(), vec![double(ty)?]),
        7 if int && lb >= 16 => ("snarrow a, b".into(), vec![half(ty)?]),
        8 if int && lb >= 16 => ("unarrow a, b".into(), vec![half(ty)?]),
        9 if int && lb >= 16 => ("uunarrow a, b".into(), vec![half(ty)?]),
        10 => (format!("bitcast.{int_of} little a"), vec![int_of]),
        11 if int => ("imul a, b".into(), vec![ty]),
        12 if int => ("umulhi a, b".into(), vec![ty]),
        13 if int => ("smulhi a, b".into(), vec![ty]),
        14 if int => ("popcnt a".into(), vec![ty]),
        15 if int => ("avg_round a, b".into(), vec![ty]),
        16 if int => ("uadd_sat a, b".into(), vec![ty]),
        17 if int && (lb == 16 || lb == 32) => ("sqmul_round_sat a, b".into(), vec![ty]),
        18 if int => ("icmp ult a, b".into(), vec![ty]),
        19 if !int => ("fcmp lt a, b".into(), vec![Type::int(lb as u16)?.by(n)?]),
        20 if !int => ("fadd a, b".into(), vec![ty]),
        21 if int && lb == 32 => (format!("fcvt_from_sint.f32x{n} a"), vec![types::F32.by(n)?]),
        22 if !int => (format!("fcvt_to_sint_sat.i32x{n} a"), vec![types::I32.by(n)?]),
        // Every lane, extracted.
        23 => (
            (0..n)
                .map(|i| format!("extractlane a, {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
            vec![ty.lane_type(); n as usize],
        ),
        // A store must write the vector's bytes only.
        24 => ("store".into(), vec![types::I64, types::I64]),
        _ => return None,
    })
}

const NUM_OPS: usize = 25;

fn build(ty: Type, k: usize) -> String {
    let (body, rets) = op(ty, k).unwrap();
    let lane = ty.lane_type();
    let n = ty.lane_count() as usize;
    let mut params = Vec::new();
    let mut text = String::new();
    let mut v = 0;
    let mut next = || {
        v += 1;
        format!("v{v}")
    };
    let mut vec = |name: &str, text: &mut String, params: &mut Vec<String>| {
        let junk = next();
        params.push(format!("{junk}: {lane}"));
        let mut cur = next();
        text.push_str(&format!("    {cur} = splat.{ty} {junk}\n"));
        for i in 0..n {
            let x = next();
            params.push(format!("{x}: {lane}"));
            let nv = if i == n - 1 { name.to_string() } else { next() };
            text.push_str(&format!("    {nv} = insertlane {cur}, {x}, {i}\n"));
            cur = nv;
        }
    };
    vec("va", &mut text, &mut params);
    vec("vb", &mut text, &mut params);
    if body == "store" {
        text.push_str(
            "    p = stack_addr.i64 ss0
    pat = iconst.i64 0x0123456789abcdef
    store notrap pat, p
    store notrap pat, p+8
    store notrap va, p
    r0 = load.i64 notrap p
    r1 = load.i64 notrap p+8
",
        );
    } else {
        for (i, line) in body.lines().enumerate() {
            let line = line.replace(" a", " va").replace(" b", " vb");
            text.push_str(&format!("    r{i} = {line}\n"));
        }
    }
    let rs: Vec<String> = (0..rets.len()).map(|i| format!("r{i}")).collect();
    // The text parser wants numbered values.
    let mut text = format!(
        "function %f({}) -> {} {{\n    ss0 = explicit_slot 16, align = 16\nblock0({}):\n{text}    return {}\n}}\n",
        params.iter().map(|p| p.split(": ").nth(1).unwrap()).collect::<Vec<_>>().join(", "),
        rets.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
        params.join(", "),
        rs.join(", "),
    );
    for (i, name) in ["va", "vb", "p", "pat"]
        .iter()
        .chain(&["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"])
        .enumerate()
    {
        text = replace_word(&text, name, &format!("v{}", 1000 + i));
    }
    text
}

fn replace_word(s: &str, from: &str, to: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find(from) {
        let before = rest[..i].chars().last();
        let after = rest[i + from.len()..].chars().next();
        let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        out.push_str(&rest[..i]);
        out.push_str(if word(before) || word(after) { from } else { to });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

fn lane_value(lane: Type, bits: u64) -> DataValue {
    match lane {
        types::I8 => DataValue::I8(bits as i8),
        types::I16 => DataValue::I16(bits as i16),
        types::I32 => DataValue::I32(bits as i32),
        types::F32 => DataValue::F32(cranelift_codegen::ir::immediates::Ieee32::with_bits(
            bits as u32,
        )),
        _ => unreachable!(),
    }
}

/// The (type, operation) pairs that make valid functions.
fn cases() -> Vec<(Type, usize)> {
    NARROW
        .iter()
        .flat_map(|&t| (0..NUM_OPS).map(move |k| (t, k)))
        .filter(|&(t, k)| op(t, k).is_some())
        // Not every operation accepts every narrow type.
        .filter(|&(t, k)| {
            cranelift_reader::parse_functions(&build(t, k)).is_ok_and(|fs| {
                let flags = cranelift_codegen::settings::Flags::new(
                    cranelift_codegen::settings::builder(),
                );
                cranelift_codegen::verify_function(&fs[0], &flags).is_ok()
            })
        })
        .collect()
}

/// With `NARROW_STATS` set, report what happened to each case on stderr.
fn stat(what: &str) {
    if std::env::var_os("NARROW_STATS").is_some() {
        eprintln!("STAT {what}");
    }
}

fn bits() -> impl Strategy<Value = u64> {
    prop_oneof![Just(0u64), Just(1), Just(u64::MAX), Just(0x8000_0000_8080), any::<u64>()]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(1024),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn junk_in_unused_bytes_is_never_observed(
        (ty, k) in prop::sample::select(cases()),
        vals in prop::collection::vec(bits(), 2 * 9),
    ) {
        let text = build(ty, k);
        let funcs = cranelift_reader::parse_functions(&text)
            .unwrap_or_else(|e| panic!("{e}\n{text}"));
        let func = &funcs[0];
        let n = ty.lane_count() as usize;
        let args: Vec<DataValue> = vals[..n + 1]
            .iter()
            .chain(&vals[9..9 + n + 1])
            .map(|&b| lane_value(ty.lane_type(), b))
            .collect();
        let returns: Vec<Type> = func.signature.returns.iter().map(|p| p.value_type).collect();
        let Some(Ok(expected)) = harness::interpret(&funcs, "f", &args) else {
            // Not implemented by the interpreter for this type.
            stat(&format!("skipped {ty} {k}"));
            return Ok(());
        };
        // cranelift-interpreter sometimes returns values of the wrong type.
        if let Some(e) = &expected
            && e.iter().zip(&returns).any(|(v, t)| v.ty().bits() != t.bits())
        {
            stat(&format!("ill-typed {ty} {k}"));
            return Ok(());
        }
        stat("compared");
        let (wasm, unsupported) = harness::compile_module(&funcs, &Config::default());
        prop_assert!(unsupported.is_empty(), "unsupported: {:?}\n{}", unsupported, text);
        let got = harness::Runner::new(&wasm).unwrap().call("f", &args, &returns).unwrap();
        let ok = match (&expected, &got) {
            (None, None) => true,
            (Some(e), Some(g)) => harness::same_results(e, g, &returns),
            _ => false,
        };
        prop_assert!(ok, "args {:?}\n  expected {:?}\n  got      {:?}\n{}", args, expected, got, text);
    }
}

#[test]
fn case_count() {
    let c = cases();
    println!("{} cases: {:?}", c.len(), c);
    assert!(c.len() > 60);
}
