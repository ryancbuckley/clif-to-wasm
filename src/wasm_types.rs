//! Mapping Cranelift types and signatures to wasm value types.

use crate::config::ModuleLayout;
use crate::error::{self, Result};
use cranelift_codegen::ir::{AbiParam, Signature, Type, types};
use wasm_encoder::ValType;

/// Maps a Cranelift type to the wasm type that represents it.
///
/// `i8`, `i16`, and `f16` live zero-extended in `i32`. Vectors up to 128 bits
/// map to `v128` (narrower vectors occupy the low bytes).
///
/// # Errors
///
/// Returns [`Error::Unsupported`](crate::Error::Unsupported) for types with no
/// single wasm value type, such as `i128` (which [`wasm_signature`] splits in
/// two) or vectors wider than 128 bits.
pub fn wasm_type(ty: Type) -> Result<ValType> {
    Ok(match fixed_type(ty) {
        types::I8 | types::I16 | types::I32 | types::F16 => ValType::I32,
        types::I64 => ValType::I64,
        types::F32 => ValType::F32,
        types::F64 => ValType::F64,
        ty if ty.is_vector()
            && (16..=128).contains(&ty.bits())
            && matches!(
                ty.lane_type(),
                types::I8
                    | types::I16
                    | types::I32
                    | types::I64
                    | types::F16
                    | types::F32
                    | types::F64
            ) =>
        {
            ValType::V128
        }
        _ => return error::unsupported(format!("type {ty}")),
    })
}

/// Maps a dynamic vector type (e.g. `i32x4*gv0`) to its fixed-size base vector
/// (scale 1), or return `ty` unchanged.
pub(crate) fn fixed_type(ty: Type) -> Type {
    if ty.is_dynamic_vector() {
        ty.dynamic_to_vector().unwrap_or(ty)
    } else {
        ty
    }
}

/// Maps a Cranelift signature to wasm parameter and result types, splitting each
/// `i128` or `f128` into two `i64`s (low half first) and applying
/// [`ModuleLayout::demote_multi_results`] when enabled in `layout`.
///
/// # Errors
///
/// Returns [`Error::Unsupported`](crate::Error::Unsupported) if a parameter or
/// return type has no wasm representation (see [`wasm_type`]).
pub fn wasm_signature(
    sig: &Signature,
    layout: &ModuleLayout,
) -> Result<(Vec<ValType>, Vec<ValType>)> {
    let lower = |ps: &[AbiParam]| -> Result<Vec<ValType>> {
        let mut out = Vec::with_capacity(ps.len());
        for p in ps {
            if matches!(p.value_type, types::I128 | types::F128) {
                out.extend([ValType::I64, ValType::I64]);
            } else {
                out.push(wasm_type(p.value_type)?);
            }
        }
        Ok(out)
    };
    let (mut params, mut results) = (lower(&sig.params)?, lower(&sig.returns)?);
    if layout.demote_multi_results && results.len() > 1 {
        params.insert(0, layout.ptr_ty());
        results.clear();
    }
    Ok((params, results))
}
