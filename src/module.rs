//! A minimal WebAssembly module assembler for compiled functions.
//!
//! Supports two module layouts:
//!
//! - **Self-contained**: defines its own memory, globals, and a `funcref`
//!   table holding every function at its own index
//!   ([`ModuleBuilder::function_table`]). Used by tests and standalone outputs.
//! - **Linked into a host program** (e.g., an in-wasm JIT): imports memory,
//!   the function table, and the stack-pointer global, and places functions in
//!   free slots of the imported table ([`ModuleBuilder::place_in_table`]).
//!
//! Either layout can also define or import exception tags for `try_call`.
//!
//! Returned indices plug directly into [`crate::Config`] and [`crate::Env`].
//! Embedders with custom section needs can use `wasm_encoder` directly with
//! [`CompiledFunction::to_wasm_function`].

use crate::{AddressType, CompiledFunction};
use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    FunctionSection, GlobalSection, GlobalType, ImportSection, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TagKind, TagSection, TagType, TypeSection, ValType,
};

macro_rules! define_index {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
        pub struct $name(pub u32);

        impl From<u32> for $name {
            fn from(i: u32) -> Self {
                Self(i)
            }
        }

        impl From<$name> for u32 {
            fn from(i: $name) -> u32 {
                i.0
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

define_index!(
    /// Index of a function signature in the WebAssembly type section.
    TypeIndex
);
define_index!(
    /// Index of a function (imported or defined) in the WebAssembly function index space.
    FuncIndex
);
define_index!(
    /// Index of a table (imported or defined) in the WebAssembly table index space.
    TableIndex
);
define_index!(
    /// Index of a linear memory (imported or defined) in the WebAssembly memory index space.
    MemoryIndex
);
define_index!(
    /// Index of a global (imported or defined) in the WebAssembly global index space.
    GlobalIndex
);
define_index!(
    /// Index of an exception tag (imported or defined) in the WebAssembly tag index space.
    TagIndex
);

/// Any WebAssembly module entity that can be exported (`FuncIndex`,
/// `TableIndex`, `MemoryIndex`, `GlobalIndex`, or `TagIndex`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub struct Export {
    /// The kind of entity.
    pub kind: ExportKind,
    /// Its index in the index space of that kind.
    pub index: u32,
}

macro_rules! impl_export {
    ($ty:ident, $kind:ident) => {
        impl From<$ty> for Export {
            fn from(i: $ty) -> Self {
                Self {
                    kind: ExportKind::$kind,
                    index: i.0,
                }
            }
        }
    };
}

impl_export!(FuncIndex, Func);
impl_export!(TableIndex, Table);
impl_export!(MemoryIndex, Memory);
impl_export!(GlobalIndex, Global);
impl_export!(TagIndex, Tag);

/// Assembles compiled functions into a WebAssembly module. See the [module
/// documentation](self), and the [crate documentation](crate) for an example.
///
/// # Example
///
/// A module that defines and exports a memory and a stack pointer for the
/// functions it will hold:
///
/// ```
/// use clif_to_wasm::wasm_encoder::{ConstExpr, ValType};
/// use clif_to_wasm::{AddressType, Config, ModuleBuilder};
///
/// let mut builder = ModuleBuilder::new();
/// let memory = builder.define_memory(1, AddressType::I32);
/// let sp = builder.define_global(ValType::I32, ConstExpr::i32_const(65536));
/// builder.export("memory", memory);
///
/// let mut config = Config::default();
/// config.module.memory_index = memory;
/// config.module.stack_pointer_global = Some(sp);
/// // ... compile functions with `config`, then declare and define them.
/// let wasm = builder.finish();
/// # wasmparser::validate(&wasm)?;
/// # Ok::<(), wasmparser::BinaryReaderError>(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct ModuleBuilder {
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    imports: Vec<(String, String, EntityType)>,
    /// Type index and body of each defined function.
    funcs: IndexSpace<FuncIndex, (TypeIndex, Option<CompiledFunction>)>,
    memories: IndexSpace<MemoryIndex, MemoryType>,
    globals: IndexSpace<GlobalIndex, (GlobalType, ConstExpr)>,
    tables: IndexSpace<TableIndex, TableType, TableType>,
    /// The type index of each defined exception tag.
    tags: IndexSpace<TagIndex, TypeIndex>,
    /// The table made by `function_table`, which is sized in `finish`.
    function_table: Option<TableIndex>,
    /// Active element segments: table, first slot, functions.
    elements: Vec<(TableIndex, u64, Vec<FuncIndex>)>,
    exports: Vec<(String, Export)>,
}

/// A WebAssembly index space (functions, memories, globals, tables, or tags).
///
/// In every WebAssembly index space, imports precede definitions, so all
/// imports of a given kind must be registered before any definition of that
/// kind; otherwise indices already handed out would shift.
#[derive(Debug, Clone)]
struct IndexSpace<Idx, D, I = ()> {
    imported: Vec<I>,
    defined: Vec<D>,
    _marker: PhantomData<fn() -> Idx>,
}

impl<Idx, D, I> Default for IndexSpace<Idx, D, I> {
    fn default() -> Self {
        Self {
            imported: Vec::new(),
            defined: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<Idx: From<u32> + Into<u32>, D, I> IndexSpace<Idx, D, I> {
    fn import(&mut self, item: I, msg: &str) -> Idx {
        assert!(self.defined.is_empty(), "{msg}");
        let idx = self.imported.len() as u32;
        self.imported.push(item);
        Idx::from(idx)
    }

    fn define(&mut self, item: D) -> Idx {
        let idx = self.total();
        self.defined.push(item);
        Idx::from(idx)
    }

    fn defined(&self) -> &[D] {
        &self.defined
    }

    fn defined_mut(&mut self, index: Idx) -> &mut D {
        let i = index
            .into()
            .checked_sub(self.imported.len() as u32)
            .expect("expected a defined index, got an import index");
        &mut self.defined[i as usize]
    }

    fn total(&self) -> u32 {
        (self.imported.len() + self.defined.len()) as u32
    }
}

impl<Idx: Into<u32>, T> IndexSpace<Idx, T, T> {
    fn get(&self, index: Idx) -> Option<&T> {
        let idx = index.into() as usize;
        self.imported
            .get(idx)
            .or_else(|| self.defined.get(idx.wrapping_sub(self.imported.len())))
    }
}

impl ModuleBuilder {
    /// An empty module.
    pub fn new() -> Self {
        Self::default()
    }

    /// The index of a function type, adding it if needed.
    pub fn type_index(&mut self, params: &[ValType], results: &[ValType]) -> TypeIndex {
        if let Some(i) = self
            .types
            .iter()
            .position(|(p, r)| p == params && r == results)
        {
            return TypeIndex(i as u32);
        }
        self.types.push((params.to_vec(), results.to_vec()));
        TypeIndex(self.types.len() as u32 - 1)
    }

    // ---- Imports ------------------------------------------------------------

    /// Import a function. Returns its function index.
    ///
    /// # Panics
    ///
    /// Panics if a function has already been declared.
    pub fn import_function(
        &mut self,
        module: &str,
        name: &str,
        params: &[ValType],
        results: &[ValType],
    ) -> FuncIndex {
        let idx = self
            .funcs
            .import((), "function imports must precede declared functions");
        let ty = self.type_index(params, results);
        self.record_import(module, name, EntityType::Function(ty.0));
        idx
    }

    /// Import a linear memory. Returns its memory index.
    ///
    /// The declared limits are a lower bound on what the provided memory must
    /// have: `minimum: 0, maximum: None` accepts any memory of the right
    /// address width (and sharedness).
    ///
    /// # Panics
    ///
    /// Panics if a memory has already been defined.
    pub fn import_memory(&mut self, module: &str, name: &str, ty: MemoryType) -> MemoryIndex {
        let idx = self
            .memories
            .import((), "memory imports must precede defined memories");
        self.record_import(module, name, EntityType::Memory(ty));
        idx
    }

    /// Import a global, such as the shadow stack pointer (which must be
    /// mutable). Returns its global index.
    ///
    /// # Panics
    ///
    /// Panics if a global has already been defined.
    pub fn import_global(&mut self, module: &str, name: &str, ty: GlobalType) -> GlobalIndex {
        let idx = self
            .globals
            .import((), "global imports must precede defined globals");
        self.record_import(module, name, EntityType::Global(ty));
        idx
    }

    /// Import a table, such as the program's table of function references.
    /// Returns its table index. Like memories, the declared limits are a
    /// lower bound.
    ///
    /// # Panics
    ///
    /// Panics if a table has already been defined.
    pub fn import_table(&mut self, module: &str, name: &str, ty: TableType) -> TableIndex {
        let idx = self
            .tables
            .import(ty, "table imports must precede defined tables");
        self.record_import(module, name, EntityType::Table(ty));
        idx
    }

    /// Import an exception tag whose payload has the given types. Returns
    /// its tag index.
    ///
    /// # Panics
    ///
    /// Panics if a tag has already been defined.
    pub fn import_tag(&mut self, module: &str, name: &str, params: &[ValType]) -> TagIndex {
        let idx = self
            .tags
            .import((), "tag imports must precede defined tags");
        let ty = self.type_index(params, &[]);
        self.record_import(
            module,
            name,
            EntityType::Tag(TagType {
                kind: TagKind::Exception,
                func_type_idx: ty.0,
            }),
        );
        idx
    }

    fn record_import(&mut self, module: &str, name: &str, ty: EntityType) {
        self.imports.push((module.into(), name.into(), ty));
    }

    // ---- Definitions --------------------------------------------------------

    /// Reserve a function index for a function with the given type.
    pub fn declare_function(&mut self, params: &[ValType], results: &[ValType]) -> FuncIndex {
        let ty = self.type_index(params, results);
        self.funcs.define((ty, None))
    }

    /// Provide the body of a declared function.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not a declared function, or if the signature of
    /// `func` differs from the declared one.
    pub fn define_function(&mut self, index: FuncIndex, func: CompiledFunction) {
        let (ty, body) = self.funcs.defined_mut(index);
        assert_eq!(
            self.types[ty.0 as usize],
            (func.params.clone(), func.results.clone()),
            "define_function: {index:?} defined with a signature other than its declared one"
        );
        *body = Some(func);
    }

    /// Define a linear memory. Returns its memory index.
    pub fn define_memory(&mut self, min_pages: u64, address_type: AddressType) -> MemoryIndex {
        self.memories.define(MemoryType {
            minimum: min_pages,
            maximum: None,
            memory64: address_type == AddressType::I64,
            shared: false,
            page_size_log2: None,
        })
    }

    /// Define a mutable global. Returns its global index.
    pub fn define_global(&mut self, ty: ValType, init: ConstExpr) -> GlobalIndex {
        self.globals.define((
            GlobalType {
                val_type: ty,
                mutable: true,
                shared: false,
            },
            init,
        ))
    }

    /// Define an exception tag whose payload has the given types. Returns its
    /// tag index.
    pub fn define_tag(&mut self, params: &[ValType]) -> TagIndex {
        let ty = self.type_index(params, &[]);
        self.tags.define(ty)
    }

    /// Define a funcref table where slot `i` holds function `i`, for every
    /// function in the module, imported or defined. Returns its table index.
    /// The table has no maximum, so the embedder can grow it later, for
    /// example to make room for functions of other modules.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    pub fn function_table(&mut self) -> TableIndex {
        assert!(self.function_table.is_none(), "function_table called twice");
        // The size is filled in by `finish`, once all functions are known.
        let index = self.tables.define(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: 0,
            maximum: None,
            shared: false,
        });
        self.function_table = Some(index);
        index
    }

    /// Store `funcs` in consecutive slots of `table` starting at `first_slot`,
    /// when the module is instantiated. The slot of `funcs[i]` is
    /// `first_slot + i`, which is what `Env::func_table_slot` should return
    /// for it. Instantiation fails if the slots are out of bounds.
    ///
    /// # Panics
    ///
    /// Panics if `table` is not a table of this module.
    pub fn place_in_table(&mut self, table: TableIndex, first_slot: u64, funcs: &[FuncIndex]) {
        assert!(self.tables.get(table).is_some(), "no table {table}");
        self.elements.push((table, first_slot, funcs.to_vec()));
    }

    // ---- Exports ------------------------------------------------------------

    /// Export a function. Same as [`Self::export`].
    pub fn export_function(&mut self, name: &str, index: FuncIndex) {
        self.export(name, index);
    }

    /// Export any entity (`FuncIndex`, `TableIndex`, `MemoryIndex`,
    /// `GlobalIndex`, or `TagIndex`), e.g. `export("memory", mem)`.
    pub fn export(&mut self, name: &str, item: impl Into<Export>) {
        self.exports.push((name.into(), item.into()));
    }

    /// Write the module.
    ///
    /// # Panics
    ///
    /// Panics if a declared function was never defined, or if a slot given
    /// to [`Self::place_in_table`] does not fit in a 32-bit table.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let total = self.funcs.total();
        if let Some(t) = self.function_table {
            self.tables.defined_mut(t).minimum = total as u64;
            self.elements
                .push((t, 0, (0..total).map(FuncIndex).collect()));
        }

        let mut module = Module::new();

        let mut types = TypeSection::new();
        for (p, r) in &self.types {
            types.ty().function(p.iter().copied(), r.iter().copied());
        }
        module.section(&types);

        if !self.imports.is_empty() {
            let mut imports = ImportSection::new();
            for (m, n, t) in &self.imports {
                imports.import(m, n, *t);
            }
            module.section(&imports);
        }

        let mut functions = FunctionSection::new();
        for (ty, _) in self.funcs.defined() {
            functions.function(ty.0);
        }
        module.section(&functions);

        if !self.tables.defined().is_empty() {
            let mut tables = TableSection::new();
            for t in self.tables.defined() {
                tables.table(*t);
            }
            module.section(&tables);
        }

        if !self.memories.defined().is_empty() {
            let mut memories = MemorySection::new();
            for m in self.memories.defined() {
                memories.memory(*m);
            }
            module.section(&memories);
        }

        if !self.tags.defined().is_empty() {
            let mut tags = TagSection::new();
            for &ty in self.tags.defined() {
                tags.tag(TagType {
                    kind: TagKind::Exception,
                    func_type_idx: ty.0,
                });
            }
            module.section(&tags);
        }

        if !self.globals.defined().is_empty() {
            let mut globals = GlobalSection::new();
            for (t, init) in self.globals.defined() {
                globals.global(*t, init);
            }
            module.section(&globals);
        }

        if !self.exports.is_empty() {
            let mut exports = ExportSection::new();
            for (n, e) in &self.exports {
                exports.export(n, e.kind, e.index);
            }
            module.section(&exports);
        }

        if !self.elements.is_empty() {
            let mut elems = ElementSection::new();
            for (table, first, funcs) in &self.elements {
                let offset = if self
                    .tables
                    .get(*table)
                    .expect("table checked by place_in_table")
                    .table64
                {
                    ConstExpr::i64_const(*first as i64)
                } else {
                    let first = u32::try_from(*first).expect("slot out of range for table32");
                    ConstExpr::i32_const(first as i32)
                };
                let func_indices: Vec<u32> = funcs.iter().map(|f| f.0).collect();
                // Table 0 uses the shorter MVP encoding.
                let table = (table.0 != 0).then_some(table.0);
                elems.active(table, &offset, Elements::Functions(func_indices.into()));
            }
            module.section(&elems);
        }

        let mut code = CodeSection::new();
        for (_, f) in self.funcs.defined() {
            let f = f.as_ref().expect("declared function was not defined");
            code.function(&f.to_wasm_function());
        }
        module.section(&code);

        module.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use proptest::prelude::*;
    use wasm_encoder::Instruction as I;
    use wasmparser::{Validator, WasmFeatures};

    fn valtype_strat() -> impl Strategy<Value = ValType> {
        prop_oneof![
            Just(ValType::I32),
            Just(ValType::I64),
            Just(ValType::F32),
            Just(ValType::F64),
            Just(ValType::V128),
        ]
    }

    fn sig_strat() -> impl Strategy<Value = (Vec<ValType>, Vec<ValType>)> {
        (
            prop::collection::vec(valtype_strat(), 0..4),
            prop::collection::vec(valtype_strat(), 0..3),
        )
    }

    fn trivial_body(params: &[ValType], results: &[ValType]) -> CompiledFunction {
        use wasm_encoder::Encode;
        let mut body = Vec::new();
        for &r in results {
            match r {
                ValType::I32 => I::I32Const(0).encode(&mut body),
                ValType::I64 => I::I64Const(0).encode(&mut body),
                ValType::F32 => I::F32Const(wasm_encoder::Ieee32::new(0)).encode(&mut body),
                ValType::F64 => I::F64Const(wasm_encoder::Ieee64::new(0)).encode(&mut body),
                ValType::V128 => I::V128Const(0).encode(&mut body),
                _ => unreachable!(),
            }
        }
        I::End.encode(&mut body);
        CompiledFunction {
            params: params.to_vec(),
            results: results.to_vec(),
            locals: Vec::new(),
            body,
            relocs: Vec::new(),
        }
    }

    proptest! {
        #[test]
        fn module_builder_produces_valid_wasm(
            imported_funcs in prop::collection::vec(sig_strat(), 0..5),
            defined_funcs in prop::collection::vec(sig_strat(), 0..6),
            imported_globals in prop::collection::vec(valtype_strat(), 0..4),
            defined_globals in prop::collection::vec(valtype_strat(), 0..4),
            imported_tags in prop::collection::vec(prop::collection::vec(valtype_strat(), 0..3), 0..3),
            defined_tags in prop::collection::vec(prop::collection::vec(valtype_strat(), 0..3), 0..3),
            import_mem in any::<bool>(),
            def_mem in any::<bool>(),
            mem64 in any::<bool>(),
            use_func_table in any::<bool>(),
            import_tbl in any::<bool>(),
            table64 in any::<bool>(),
        ) {
            let mut b = ModuleBuilder::new();
            let mut all_funcs = Vec::new();

            // Type deduplication property: registering identical signatures returns the same TypeIndex.
            for (i, (params, results)) in imported_funcs.iter().enumerate() {
                let t1 = b.type_index(params, results);
                let idx = b.import_function("env", &format!("ifunc_{i}"), params, results);
                let t2 = b.type_index(params, results);
                prop_assert_eq!(t1, t2, "identical signatures must share a type index");
                prop_assert_eq!(idx, FuncIndex(i as u32), "imported functions are numbered in import order");
                all_funcs.push(idx);
            }

            if import_mem {
                let m = b.import_memory(
                    "env",
                    "imem",
                    MemoryType {
                        minimum: 1,
                        maximum: None,
                        memory64: mem64,
                        shared: false,
                        page_size_log2: None,
                    },
                );
                prop_assert_eq!(m, MemoryIndex(0), "an imported memory comes first");
            }
            if def_mem {
                let m = b.define_memory(1, if mem64 { AddressType::I64 } else { AddressType::I32 });
                prop_assert_eq!(m, MemoryIndex(import_mem as u32), "a defined memory follows any imported one");
                b.export("mem_export", m);
            }

            for (i, &vt) in imported_globals.iter().enumerate() {
                let g = b.import_global(
                    "env",
                    &format!("iglob_{i}"),
                    GlobalType {
                        val_type: vt,
                        mutable: true,
                        shared: false,
                    },
                );
                prop_assert_eq!(g, GlobalIndex(i as u32), "imported globals are numbered in import order");
            }
            for (i, &vt) in defined_globals.iter().enumerate() {
                let init = match vt {
                    ValType::I32 => ConstExpr::i32_const(0),
                    ValType::I64 => ConstExpr::i64_const(0),
                    ValType::F32 => ConstExpr::f32_const(wasm_encoder::Ieee32::new(0)),
                    ValType::F64 => ConstExpr::f64_const(wasm_encoder::Ieee64::new(0)),
                    ValType::V128 => ConstExpr::v128_const(0),
                    _ => unreachable!(),
                };
                let g = b.define_global(vt, init);
                prop_assert_eq!(g, GlobalIndex((imported_globals.len() + i) as u32), "defined globals follow the imported ones");
                b.export(&format!("eglob_{i}"), g);
            }

            for (i, params) in imported_tags.iter().enumerate() {
                let t = b.import_tag("env", &format!("itag_{i}"), params);
                prop_assert_eq!(t, TagIndex(i as u32), "imported tags are numbered in import order");
            }
            for (i, params) in defined_tags.iter().enumerate() {
                let t = b.define_tag(params);
                prop_assert_eq!(t, TagIndex((imported_tags.len() + i) as u32), "defined tags follow the imported ones");
                b.export(&format!("etag_{i}"), t);
            }

            let imported_tbl = if import_tbl {
                let t = b.import_table(
                    "env",
                    "itbl",
                    TableType {
                        element_type: RefType::FUNCREF,
                        table64,
                        minimum: 10,
                        maximum: None,
                        shared: false,
                    },
                );
                prop_assert_eq!(t, TableIndex(0), "an imported table comes first");
                Some(t)
            } else {
                None
            };

            let mut decls = Vec::new();
            for (i, (params, results)) in defined_funcs.iter().enumerate() {
                let idx = b.declare_function(params, results);
                prop_assert_eq!(idx, FuncIndex((imported_funcs.len() + i) as u32), "declared functions follow the imported ones");
                decls.push((idx, params.clone(), results.clone()));
                all_funcs.push(idx);
            }

            // Define functions in reverse order to test out-of-order definition.
            for (idx, params, results) in decls.iter().rev() {
                b.define_function(*idx, trivial_body(params, results));
            }
            for (i, &f) in all_funcs.iter().enumerate() {
                b.export_function(&format!("fn_{i}"), f);
            }

            if use_func_table {
                let ft = b.function_table();
                prop_assert_eq!(ft, TableIndex(import_tbl as u32), "the function table follows any imported table");
                b.export("ftbl", ft);
            } else if let Some(t) = imported_tbl {
                b.place_in_table(t, 2, &all_funcs);
            }

            let wasm = b.finish();
            let mut validator = Validator::new_with_features(WasmFeatures::all());
            prop_assert!(
                validator.validate_all(&wasm).is_ok(),
                "invalid wasm produced by ModuleBuilder: {:?}",
                validator.validate_all(&wasm).err()
            );
        }
    }
}
