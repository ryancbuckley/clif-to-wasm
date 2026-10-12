//! The emulated floating point (`f16`, and `fma` via round-to-odd), checked
//! against independent references: exhaustively where the inputs allow it,
//! and on random and crafted inputs elsewhere.

mod harness;

use cranelift_wasm_backend::Config;
use harness::clifgen::Rng;
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

/// Compile CLIF functions into one instance.
fn instance(clif: &str) -> (Store<()>, Instance) {
    let funcs = cranelift_reader::parse_functions(clif).unwrap();
    let (wasm, unsupported) = harness::compile_module(&funcs, &Config::default());
    assert!(unsupported.is_empty(), "{unsupported:?}");
    harness::validate(&wasm).unwrap();
    let engine = Engine::default();
    let mut store = Store::new(&engine, ());
    let module = Module::new(&engine, &wasm).unwrap();
    let inst = Instance::new(&mut store, &module, &[]).unwrap();
    (store, inst)
}

fn func<P: wasmtime::WasmParams, R: wasmtime::WasmResults>(
    store: &mut Store<()>,
    inst: &Instance,
    name: &str,
) -> TypedFunc<P, R> {
    inst.get_typed_func(store, name).unwrap()
}

/// The value of finite `f16` bits, exactly.
fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (h >> 10) & 0x1f;
    let m = (h & 0x3ff) as f64;
    let mag = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1024.0 + m) * 2f64.powi(e as i32 - 25)
    };
    sign * mag
}

fn f16_is_nan(h: u16) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x3ff != 0
}

/// Round a non-NaN `f64` to `f16` bits, to nearest with ties to even, by
/// searching the sorted `f16` values rather than manipulating bits.
fn round_to_f16(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    // The largest finite f16 is 65504; halfway to the next step (65536)
    // rounds to even, which is infinity.
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    // The largest h with value <= a.
    let (mut lo, mut hi) = (0u16, 0x7bff);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if f16_value(mid) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = lo;
    let above = below + 1;
    let up = if above == 0x7c00 {
        65536.0
    } else {
        f16_value(above)
    };
    let (d_below, d_above) = (a - f16_value(below), up - a);
    let h = if d_below < d_above || (d_below == d_above && below % 2 == 0) {
        below
    } else {
        above
    };
    sign | h
}

#[test]
fn f16_converts_exactly_and_back() {
    let (mut store, inst) = instance(
        "
function %via_f64(f16) -> f16 {
block0(v0: f16):
    v1 = fpromote.f64 v0
    v2 = fdemote.f16 v1
    return v2
}
function %via_f32(f16) -> f16 {
block0(v0: f16):
    v1 = fpromote.f32 v0
    v2 = fdemote.f16 v1
    return v2
}
function %to_f64(f16) -> f64 {
block0(v0: f16):
    v1 = fpromote.f64 v0
    return v1
}
",
    );
    let via_f64 = func::<i32, i32>(&mut store, &inst, "via_f64");
    let via_f32 = func::<i32, i32>(&mut store, &inst, "via_f32");
    let to_f64 = func::<i32, f64>(&mut store, &inst, "to_f64");
    for h in 0..=u16::MAX {
        let x = to_f64.call(&mut store, h.into()).unwrap();
        for f in [&via_f64, &via_f32] {
            let back = f.call(&mut store, h.into()).unwrap() as u16;
            if f16_is_nan(h) {
                // A NaN stays a NaN with its sign; its payload survives
                // as far as the conversions keep it.
                assert!(f16_is_nan(back) && back & 0x8000 == h & 0x8000, "{h:#x} -> {back:#x}");
            } else {
                assert_eq!(back, h, "{h:#x}");
            }
        }
        if f16_is_nan(h) {
            assert!(x.is_nan());
        } else if h & 0x7fff == 0x7c00 {
            assert_eq!(x, if h & 0x8000 != 0 { f64::NEG_INFINITY } else { f64::INFINITY });
        } else {
            assert_eq!(x.to_bits(), f16_value(h).to_bits(), "{h:#x}");
        }
    }
}

#[test]
fn f16_rounds_to_nearest_even() {
    let (mut store, inst) = instance(
        "
function %from_f64(f64) -> f16 {
block0(v0: f64):
    v1 = fdemote.f16 v0
    return v1
}
function %from_f32(f32) -> f16 {
block0(v0: f32):
    v1 = fdemote.f16 v0
    return v1
}
",
    );
    let from_f64 = func::<f64, i32>(&mut store, &inst, "from_f64");
    let from_f32 = func::<f32, i32>(&mut store, &inst, "from_f32");
    // Each midpoint between neighboring f16 values, and the closest values
    // on either side of it, in f64 and in f32 (where all are exact).
    for h in 0..0x7c00u16 {
        let lo = f16_value(h);
        let hi = if h + 1 == 0x7c00 {
            65536.0
        } else {
            f16_value(h + 1)
        };
        let mid = (lo + hi) / 2.0;
        let even = if h % 2 == 0 { h } else { h + 1 };
        let cases64 = [
            (mid, even),
            (mid.next_up(), h + 1),
            (mid.next_down(), h),
        ];
        let mid32 = mid as f32;
        assert_eq!(mid32 as f64, mid);
        let cases32 = [
            (mid32, even),
            (mid32.next_up(), h + 1),
            (mid32.next_down(), h),
        ];
        for sign in [1.0, -1.0] {
            let s = if sign < 0.0 { 0x8000 } else { 0 };
            for (x, want) in cases64 {
                let got = from_f64.call(&mut store, sign * x).unwrap() as u16;
                assert_eq!(got, s | want, "f64 {:e}", sign * x);
            }
            for (x, want) in cases32 {
                let got = from_f32.call(&mut store, sign as f32 * x).unwrap() as u16;
                assert_eq!(got, s | want, "f32 {:e}", sign as f32 * x);
            }
        }
    }
    // Far outside the range.
    for (x, want) in [
        (1e300, 0x7c00),
        (-1e300, 0xfc00),
        (1e-300, 0),
        (-1e-300, 0x8000),
        (f64::MIN_POSITIVE / 4.0, 0),
    ] {
        assert_eq!(from_f64.call(&mut store, x).unwrap() as u16, want, "{x:e}");
    }
}

#[test]
fn f16_arithmetic_matches_reference() {
    let ops = [
        ("fadd", 2),
        ("fsub", 2),
        ("fmul", 2),
        ("fdiv", 2),
        ("sqrt", 1),
    ];
    let clif: String = ops
        .iter()
        .map(|(op, n)| {
            if *n == 2 {
                format!(
                    "function %{op}(f16, f16) -> f16 {{ block0(v0: f16, v1: f16): v2 = {op} v0, v1  return v2 }}\n"
                )
            } else {
                format!("function %{op}(f16) -> f16 {{ block0(v0: f16): v1 = {op} v0  return v1 }}\n")
            }
        })
        .collect();
    let (mut store, inst) = instance(&clif);
    let bin: Vec<TypedFunc<(i32, i32), i32>> = ops[..4]
        .iter()
        .map(|(op, _)| func(&mut store, &inst, op))
        .collect();
    let sqrt = func::<i32, i32>(&mut store, &inst, "sqrt");
    let mut rng = Rng(0xf16);
    let finite = |rng: &mut Rng| loop {
        let h = rng.next() as u16;
        if h & 0x7c00 != 0x7c00 {
            return h;
        }
    };
    for _ in 0..100_000 {
        let (a, b) = (finite(&mut rng), finite(&mut rng));
        let (x, y) = (f16_value(a), f16_value(b));
        // Sums, differences and products of f16 values are exact in f64;
        // a quotient or square root rounded to f64 first rounds the same.
        let exact = [x + y, x - y, x * y, x / y];
        for (f, want) in bin.iter().zip(exact) {
            let got = f.call(&mut store, (a.into(), b.into())).unwrap() as u16;
            if want.is_nan() {
                assert!(f16_is_nan(got));
            } else {
                assert_eq!(got, round_to_f16(want), "{a:#x} {b:#x} -> {want:e}");
            }
        }
        let got = sqrt.call(&mut store, a.into()).unwrap() as u16;
        if x < 0.0 {
            assert!(f16_is_nan(got));
        } else {
            assert_eq!(got, round_to_f16(x.sqrt()), "sqrt {a:#x}");
        }
    }
}

#[test]
fn fma_f32_matches_mul_add() {
    let (mut store, inst) = instance(
        "
function %fma(f32, f32, f32) -> f32 {
block0(v0: f32, v1: f32, v2: f32):
    v3 = fma v0, v1, v2
    return v3
}
",
    );
    let fma = func::<(f32, f32, f32), f32>(&mut store, &inst, "fma");
    let mut rng = Rng(0xf3a);
    let check = |store: &mut Store<()>, a: f32, b: f32, c: f32| {
        let got = fma.call(&mut *store, (a, b, c)).unwrap();
        let want = a.mul_add(b, c);
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "fma({a:e}, {b:e}, {c:e}) = {got:e}, want {want:e}"
        );
    };
    for _ in 0..200_000 {
        let a = f32::from_bits(rng.next() as u32);
        let b = f32::from_bits(rng.next() as u32);
        let c = f32::from_bits(rng.next() as u32);
        check(&mut store, a, b, c);
        // Moderate magnitudes, with c cancelling most of the product: the
        // cases where rounding the product first would go wrong.
        let a = (rng.next() as u32 % 2_000_000) as f32 / 1024.0 - 900.0;
        let b = (rng.next() as u32 % 2_000_000) as f32 / 1024.0 - 900.0;
        let near = -(a * b);
        let c = f32::from_bits(near.to_bits().wrapping_add(rng.next() as u32 % 5).wrapping_sub(2));
        check(&mut store, a, b, c);
        // Products near a halfway point of the result.
        let c = (rng.next() as u32 % 1000) as f32 * 2f32.powi(-30);
        check(&mut store, a, b, c);
    }
    // The double-rounding trap: c has an odd last bit and a * b is half an
    // ulp of c minus a sliver that f64 cannot hold. Rounding to f64 lands
    // exactly halfway between c and its even neighbor, and rounding that
    // to f32 picks the neighbor; the right answer is c.
    for _ in 0..100_000 {
        let c = f32::from_bits((rng.next() as u32 % 0x6000_0000 + 0x1000_0000) | 1);
        // ulp(c) / 2 = 2^k, split into 2^p * 2^q.
        let k = ((c.to_bits() >> 23) & 0xff) as i32 - 127 - 24;
        let (p, q) = (k / 2, k - k / 2);
        let a = 2f32.powi(p) * (1.0 + f32::EPSILON);
        let b = 2f32.powi(q) * (1.0 - f32::EPSILON);
        for (a, c) in [(a, c), (-a, -c)] {
            check(&mut store, a, b, c);
            assert_eq!(fma.call(&mut store, (a, b, c)).unwrap(), c);
        }
    }
}

#[test]
fn fma_f16_matches_exact_arithmetic() {
    let (mut store, inst) = instance(
        "
function %fma(f16, f16, f16) -> f16 {
block0(v0: f16, v1: f16, v2: f16):
    v3 = fma v0, v1, v2
    return v3
}
",
    );
    let fma = func::<(i32, i32, i32), i32>(&mut store, &inst, "fma");
    // f16 values are multiples of 2^-24, so a * b + c is an integer
    // multiple of 2^-48, exact in an i128.
    let scaled = |h: u16| (f16_value(h) * 2f64.powi(24)) as i128;
    let mut rng = Rng(0xf16a);
    let finite = |rng: &mut Rng| loop {
        let h = rng.next() as u16;
        if h & 0x7c00 != 0x7c00 {
            return h;
        }
    };
    for _ in 0..100_000 {
        let (a, b, c) = (finite(&mut rng), finite(&mut rng), finite(&mut rng));
        let exact = scaled(a) * scaled(b) + (scaled(c) << 24);
        let got = fma.call(&mut store, (a.into(), b.into(), c.into())).unwrap() as u16;
        let want = if exact == 0 {
            // An exact zero is +0, unless both terms are -0 or the
            // product's and c's zeros are both negative.
            let p_neg = (a ^ b) & 0x8000 != 0;
            let c_neg = c & 0x8000 != 0;
            let p_zero = f16_value(a) == 0.0 || f16_value(b) == 0.0;
            if p_zero && f16_value(c) == 0.0 && p_neg && c_neg {
                0x8000
            } else {
                0
            }
        } else {
            round_scaled(exact)
        };
        assert_eq!(got, want, "fma({a:#x}, {b:#x}, {c:#x})");
    }
}

/// Round `n * 2^-48` to `f16` bits, comparing integers only.
fn round_scaled(n: i128) -> u16 {
    let sign = if n < 0 { 0x8000 } else { 0 };
    let a = n.unsigned_abs();
    let value = |h: u16| (f16_value(h) * 2f64.powi(48)) as u128;
    let top = 65536u128 << 48;
    if a >= (value(0x7bff) + top) / 2 {
        return sign | 0x7c00;
    }
    let (mut lo, mut hi) = (0u16, 0x7bff);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if value(mid) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let (below, above) = (lo, lo + 1);
    let up = if above == 0x7c00 { top } else { value(above) };
    let (d_below, d_above) = (a - value(below), up - a);
    let h = if d_below < d_above || (d_below == d_above && below % 2 == 0) {
        below
    } else {
        above
    };
    sign | h
}
