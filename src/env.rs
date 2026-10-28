//! The [`Env`] trait, through which compiled functions resolve references to
//! the surrounding wasm module.

use crate::error::{Error, Result};
use crate::module::{FuncIndex, GlobalIndex, TagIndex, TypeIndex};
use crate::object::SymbolId;
use alloc::format;
use cranelift_codegen::ir::{
    ExceptionTag, ExternalName, Function, KnownSymbol, LibCall, UserExternalName,
};
use wasm_encoder::ValType;

/// A resolved external name, passed to the [`Env`] callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Name<'func> {
    /// A user-defined name, from the function's table of named functions.
    User(&'func UserExternalName),
    /// A test-case name (`%name` in CLIF text).
    TestCase(&'func str),
    /// A Cranelift runtime library call.
    LibCall(LibCall),
    /// A well-known symbol, such as the ELF global offset table.
    KnownSymbol(KnownSymbol),
}

impl<'func> Name<'func> {
    pub(crate) fn resolve(func: &'func Function, name: &'func ExternalName) -> Name<'func> {
        match name {
            ExternalName::User(r) => Name::User(&func.params.user_named_funcs()[*r]),
            ExternalName::TestCase(t) => {
                Name::TestCase(core::str::from_utf8(t.raw()).unwrap_or("<invalid>"))
            }
            ExternalName::LibCall(l) => Name::LibCall(*l),
            ExternalName::KnownSymbol(k) => Name::KnownSymbol(*k),
        }
    }
}

impl core::fmt::Display for Name<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Name::User(u) => write!(f, "u{}:{}", u.namespace, u.index),
            Name::TestCase(s) => write!(f, "%{s}"),
            Name::LibCall(l) => write!(f, "{l:?}"),
            Name::KnownSymbol(k) => write!(f, "{k:?}"),
        }
    }
}

/// Resolved value of a `symbol_value` or TLS offset for `tls_value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SymbolValue {
    /// Link-time constant address or offset.
    Const(u64),
    /// Address held in a wasm global.
    Global(GlobalIndex),
    /// Data symbol in the object's symbol table (for
    /// [`ModuleLayout::relocatable`](crate::ModuleLayout::relocatable)).
    Symbol(SymbolId),
}

/// Resolves external references from a function to entities in the surrounding
/// wasm module.
///
/// An error returned by any method fails
/// [`compile_function`](crate::compile_function) with that error. Only
/// [`Env::func_index`] is required: the other methods default to returning
/// [`Error::Env`], for environments whose functions never need them.
pub trait Env {
    /// Wasm function index (or symbol index when relocatable) of a direct callee.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not name a known function.
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex>;

    /// Wasm type index of a function or block type (`params -> results`).
    ///
    /// # Errors
    ///
    /// Returns an error if the type cannot be added to the module. The default
    /// implementation always returns [`Error::Env`].
    fn type_index(&mut self, params: &[ValType], results: &[ValType]) -> Result<TypeIndex> {
        let _ = (params, results);
        Err(Error::Env("call_indirect requires Env::type_index".into()))
    }

    /// Function table slot for `func_addr`, or function symbol index when
    /// [`ModuleLayout::relocatable`](crate::ModuleLayout::relocatable) is set.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not name a function with a table slot.
    /// The default implementation always returns [`Error::Env`].
    fn func_table_slot(&mut self, name: Name<'_>) -> Result<u64> {
        Err(Error::Env(format!(
            "func_addr {name} requires Env::func_table_slot"
        )))
    }

    /// Value of `symbol_value` (or TLS offset for `tls_value`).
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not name a known symbol. The default
    /// implementation always returns [`Error::Env`].
    fn symbol_value(&mut self, name: Name<'_>) -> Result<SymbolValue> {
        Err(Error::Env(format!(
            "symbol_value {name} requires Env::symbol_value"
        )))
    }

    /// Wasm tag index for a `try_call` exception handler. The tag carries two
    /// pointer-sized integers matching the calling convention's payload.
    ///
    /// # Errors
    ///
    /// Returns an error if `tag` has no wasm tag. The default implementation
    /// always returns [`Error::Env`].
    fn exception_tag(&mut self, tag: ExceptionTag) -> Result<TagIndex> {
        Err(Error::Env(format!(
            "exception tag {tag} requires Env::exception_tag"
        )))
    }
}
