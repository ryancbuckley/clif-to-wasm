//! A minimal module assembler for compiled functions.
//!
//! This is enough to produce a self-contained module from a set of
//! functions: imports, one memory, one funcref table holding every function
//! (so that a function's table slot is its function index), and globals.
//! Embedders with more specific needs can use `wasm_encoder` directly with
//! [`CompiledFunction::to_wasm_function`].

use crate::CompiledFunction;
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    FunctionSection, GlobalSection, GlobalType, ImportSection, MemoryType, Module, RefType,
    TableType, TypeSection, ValType,
};

#[derive(Default)]
pub struct ModuleBuilder {
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    imports: Vec<(String, String, EntityType)>,
    num_imported_funcs: u32,
    /// Type index and body of each defined function.
    funcs: Vec<(u32, Option<CompiledFunction>)>,
    exports: Vec<(String, ExportKind, u32)>,
    memory: Option<MemoryType>,
    globals: Vec<(GlobalType, ConstExpr)>,
    table: bool,
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

    /// Import a function. All imports must come before the first
    /// `declare_function`.
    pub fn import_function(
        &mut self,
        module: &str,
        name: &str,
        params: &[ValType],
        results: &[ValType],
    ) -> u32 {
        assert!(
            self.funcs.is_empty(),
            "imports must precede defined functions"
        );
        let ty = self.type_index(params, results);
        self.imports
            .push((module.into(), name.into(), EntityType::Function(ty)));
        self.num_imported_funcs += 1;
        self.num_imported_funcs - 1
    }

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

    pub fn export_function(&mut self, name: &str, index: u32) {
        self.exports.push((name.into(), ExportKind::Func, index));
    }

    /// Define the module's memory (index 0) and export it as `memory`.
    pub fn memory(&mut self, min_pages: u64, memory64: bool) {
        self.memory = Some(MemoryType {
            minimum: min_pages,
            maximum: None,
            memory64,
            shared: false,
            page_size_log2: None,
        });
        self.exports.push(("memory".into(), ExportKind::Memory, 0));
    }

    /// Define a mutable global. Returns its index.
    pub fn global(&mut self, ty: ValType, init: ConstExpr) -> u32 {
        self.globals.push((
            GlobalType {
                val_type: ty,
                mutable: true,
                shared: false,
            },
            init,
        ));
        self.globals.len() as u32 - 1
    }

    /// Add a funcref table (index 0) where slot `i` holds function `i`.
    pub fn function_table(&mut self) {
        self.table = true;
    }

    pub fn finish(self) -> Vec<u8> {
        let mut module = Module::new();

        let mut types = TypeSection::new();
        for (p, r) in &self.types {
            types.ty().function(p.iter().copied(), r.iter().copied());
        }
        module.section(&types);

        let mut imports = ImportSection::new();
        for (m, n, t) in &self.imports {
            imports.import(m, n, *t);
        }
        if !self.imports.is_empty() {
            module.section(&imports);
        }

        let mut functions = FunctionSection::new();
        for (ty, _) in &self.funcs {
            functions.function(*ty);
        }
        module.section(&functions);

        let total = self.num_imported_funcs + self.funcs.len() as u32;
        if self.table {
            let mut tables = wasm_encoder::TableSection::new();
            tables.table(TableType {
                element_type: RefType::FUNCREF,
                table64: false,
                minimum: total as u64,
                maximum: Some(total as u64),
                shared: false,
            });
            module.section(&tables);
        }

        if let Some(mem) = self.memory {
            let mut memories = wasm_encoder::MemorySection::new();
            memories.memory(mem);
            module.section(&memories);
        }

        if !self.globals.is_empty() {
            let mut globals = GlobalSection::new();
            for (t, init) in &self.globals {
                globals.global(*t, init);
            }
            module.section(&globals);
        }

        let mut exports = ExportSection::new();
        for (n, k, i) in &self.exports {
            exports.export(n, *k, *i);
        }
        module.section(&exports);

        if self.table {
            let mut elems = ElementSection::new();
            let all: Vec<u32> = (0..total).collect();
            elems.active(
                None,
                &ConstExpr::i32_const(0),
                Elements::Functions(all.into()),
            );
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
