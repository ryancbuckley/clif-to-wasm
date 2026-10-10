//! A small module assembler for compiled functions.
//!
//! It covers two kinds of module:
//!
//! - **Self-contained**: the module defines its own memory, a funcref table
//!   holding every function at its own index ([`ModuleBuilder::function_table`],
//!   so a function's table slot is its function index) and globals. The tests
//!   build modules like this.
//! - **Linked into a running program**, as a JIT running inside wasm needs:
//!   the module imports the program's memory, function table and stack
//!   pointer global, and places its functions in free slots of that table
//!   ([`ModuleBuilder::place_in_table`]).
//!
//! The indices this returns are the ones to put in [`crate::Config`] and to
//! answer [`crate::Env`] queries with. Embedders with other needs can use
//! `wasm_encoder` directly with [`CompiledFunction::to_wasm_function`].

use crate::CompiledFunction;
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    FunctionSection, GlobalSection, GlobalType, ImportSection, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};

#[derive(Default)]
pub struct ModuleBuilder {
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    imports: Vec<(String, String, EntityType)>,
    num_imported_funcs: u32,
    num_imported_memories: u32,
    num_imported_globals: u32,
    num_imported_tables: u32,
    /// Type index and body of each defined function.
    funcs: Vec<(u32, Option<CompiledFunction>)>,
    exports: Vec<(String, ExportKind, u32)>,
    memories: Vec<MemoryType>,
    globals: Vec<(GlobalType, ConstExpr)>,
    /// Every table, imported ones first, indexed by table index.
    tables: Vec<TableType>,
    /// The table made by `function_table`, which is sized in `finish`.
    function_table: Option<u32>,
    /// Active element segments: table, first slot, functions.
    elements: Vec<(u32, u64, Vec<u32>)>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The index of a function type, adding it if needed.
    pub fn type_index(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
        if let Some(i) = self
            .types
            .iter()
            .position(|(p, r)| p == params && r == results)
        {
            return i as u32;
        }
        self.types.push((params.to_vec(), results.to_vec()));
        self.types.len() as u32 - 1
    }

    // Imports. In each index space (functions, memories, globals, tables)
    // imports come first, so an import must precede every definition of its
    // kind; otherwise indices already handed out would shift.

    /// Import a function. Returns its function index.
    pub fn import_function(
        &mut self,
        module: &str,
        name: &str,
        params: &[ValType],
        results: &[ValType],
    ) -> u32 {
        assert!(
            self.funcs.is_empty(),
            "function imports must precede declared functions"
        );
        let ty = self.type_index(params, results);
        self.import(module, name, EntityType::Function(ty));
        self.num_imported_funcs += 1;
        self.num_imported_funcs - 1
    }

    /// Import a linear memory. Returns its memory index.
    ///
    /// The declared limits are a lower bound on what the provided memory must
    /// have: `minimum: 0, maximum: None` accepts any memory of the right
    /// address width (and sharedness).
    pub fn import_memory(&mut self, module: &str, name: &str, ty: MemoryType) -> u32 {
        assert!(
            self.memories.is_empty(),
            "memory imports must precede defined memories"
        );
        self.import(module, name, EntityType::Memory(ty));
        self.num_imported_memories += 1;
        self.num_imported_memories - 1
    }

    /// Import a global, such as the shadow stack pointer (which must be
    /// mutable). Returns its global index.
    pub fn import_global(&mut self, module: &str, name: &str, ty: GlobalType) -> u32 {
        assert!(
            self.globals.is_empty(),
            "global imports must precede defined globals"
        );
        self.import(module, name, EntityType::Global(ty));
        self.num_imported_globals += 1;
        self.num_imported_globals - 1
    }

    /// Import a table, such as the program's table of function references.
    /// Returns its table index. Like memories, the declared limits are a
    /// lower bound.
    pub fn import_table(&mut self, module: &str, name: &str, ty: TableType) -> u32 {
        assert!(
            self.tables.len() as u32 == self.num_imported_tables,
            "table imports must precede defined tables"
        );
        self.import(module, name, EntityType::Table(ty));
        self.tables.push(ty);
        self.num_imported_tables += 1;
        self.num_imported_tables - 1
    }

    fn import(&mut self, module: &str, name: &str, ty: EntityType) {
        self.imports.push((module.into(), name.into(), ty));
    }

    // Definitions.

    /// Reserve a function index for a function with the given type.
    pub fn declare_function(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
        let ty = self.type_index(params, results);
        self.funcs.push((ty, None));
        self.num_imported_funcs + self.funcs.len() as u32 - 1
    }

    /// Provide the body of a declared function.
    pub fn define_function(&mut self, index: u32, func: CompiledFunction) {
        let slot = &mut self.funcs[(index - self.num_imported_funcs) as usize];
        assert_eq!(
            self.types[slot.0 as usize],
            (func.params.clone(), func.results.clone())
        );
        slot.1 = Some(func);
    }

    /// Define a linear memory. Returns its memory index.
    pub fn memory(&mut self, min_pages: u64, memory64: bool) -> u32 {
        self.memories.push(MemoryType {
            minimum: min_pages,
            maximum: None,
            memory64,
            shared: false,
            page_size_log2: None,
        });
        self.num_imported_memories + self.memories.len() as u32 - 1
    }

    /// Define a mutable global. Returns its global index.
    pub fn global(&mut self, ty: ValType, init: ConstExpr) -> u32 {
        self.globals.push((
            GlobalType {
                val_type: ty,
                mutable: true,
                shared: false,
            },
            init,
        ));
        self.num_imported_globals + self.globals.len() as u32 - 1
    }

    /// Define a funcref table where slot `i` holds function `i`, for every
    /// function in the module, imported or defined. Returns its table index.
    /// The table has no maximum, so the embedder can grow it later, for
    /// example to make room for functions of other modules.
    pub fn function_table(&mut self) -> u32 {
        assert!(self.function_table.is_none(), "function_table called twice");
        let index = self.tables.len() as u32;
        // The size is filled in by `finish`, once all functions are known.
        self.tables.push(TableType {
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
    pub fn place_in_table(&mut self, table: u32, first_slot: u64, funcs: &[u32]) {
        assert!(
            (table as usize) < self.tables.len(),
            "no table {table}"
        );
        self.elements.push((table, first_slot, funcs.to_vec()));
    }

    // Exports.

    pub fn export_function(&mut self, name: &str, index: u32) {
        self.export(name, ExportKind::Func, index);
    }

    /// Export any entity, e.g. `export("memory", ExportKind::Memory, 0)`.
    pub fn export(&mut self, name: &str, kind: ExportKind, index: u32) {
        self.exports.push((name.into(), kind, index));
    }

    pub fn finish(mut self) -> Vec<u8> {
        let total = self.num_imported_funcs + self.funcs.len() as u32;
        if let Some(t) = self.function_table {
            self.tables[t as usize].minimum = total as u64;
            self.elements.push((t, 0, (0..total).collect()));
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
        for (ty, _) in &self.funcs {
            functions.function(*ty);
        }
        module.section(&functions);

        let defined_tables = &self.tables[self.num_imported_tables as usize..];
        if !defined_tables.is_empty() {
            let mut tables = TableSection::new();
            for t in defined_tables {
                tables.table(*t);
            }
            module.section(&tables);
        }

        if !self.memories.is_empty() {
            let mut memories = MemorySection::new();
            for m in &self.memories {
                memories.memory(*m);
            }
            module.section(&memories);
        }

        if !self.globals.is_empty() {
            let mut globals = GlobalSection::new();
            for (t, init) in &self.globals {
                globals.global(*t, init);
            }
            module.section(&globals);
        }

        if !self.exports.is_empty() {
            let mut exports = ExportSection::new();
            for (n, k, i) in &self.exports {
                exports.export(n, *k, *i);
            }
            module.section(&exports);
        }

        if !self.elements.is_empty() {
            let mut elems = ElementSection::new();
            for (table, first, funcs) in &self.elements {
                let offset = if self.tables[*table as usize].table64 {
                    ConstExpr::i64_const(*first as i64)
                } else {
                    let first = u32::try_from(*first).expect("slot out of range for table32");
                    ConstExpr::i32_const(first as i32)
                };
                // Table 0 uses the shorter MVP encoding.
                let table = (*table != 0).then_some(*table);
                elems.active(table, &offset, Elements::Functions(funcs.into()));
            }
            module.section(&elems);
        }

        let mut code = CodeSection::new();
        for (_, f) in &self.funcs {
            let f = f.as_ref().expect("declared function was not defined");
            code.function(&f.to_wasm_function());
        }
        module.section(&code);

        module.finish()
    }
}
