//! Relocatable wasm object files, following the WebAssembly [linking conventions]
//! consumed by `wasm-ld` and `wild`.
//!
//! [`ObjectBuilder`] collects symbols (functions, data, and linker-provided
//! globals and tables such as `__stack_pointer` and
//! `__indirect_function_table`) along with their definitions. Functions are
//! compiled with [`crate::ModuleLayout::relocatable`] enabled and an
//! [`crate::Env`] that returns symbol indices. [`ObjectBuilder::finish`] then
//! assigns final indices, patches placeholders, and writes the `linking` and
//! `reloc.*` custom sections. Undefined functions, globals, and tables become
//! imports from `"env"` (unless overridden via [`ObjectBuilder::import_function`]).
//!
//! Data segments are laid out sequentially from address 0 in the object's own
//! address space; relocated fields hold provisional addresses until linked.
//!
//! [linking conventions]: https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md

use crate::{AddressType, CompiledFunction, RelocKind, TypeIndex};
use alloc::borrow::{Cow, ToOwned};
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use hashbrown::HashMap;
use wasm_encoder::{CustomSection, Encode, Section, ValType};

/// An entry of the object's symbol table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SymbolId(pub u32);

impl From<SymbolId> for crate::FuncIndex {
    fn from(s: SymbolId) -> Self {
        Self(s.0)
    }
}

impl From<SymbolId> for crate::GlobalIndex {
    fn from(s: SymbolId) -> Self {
        Self(s.0)
    }
}

impl From<SymbolId> for crate::TableIndex {
    fn from(s: SymbolId) -> Self {
        Self(s.0)
    }
}

/// How a symbol binds across objects.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[non_exhaustive]
pub enum Binding {
    /// Visible to other objects; one definition wins.
    #[default]
    Global,
    /// Like `Global`, but other definitions take precedence, and an
    /// undefined weak symbol is allowed.
    Weak,
    /// Private to this object.
    Local,
}

/// The attributes of a symbol.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SymbolFlags {
    /// How the symbol binds across objects.
    pub binding: Binding,
    /// Not exported from a shared library (`WASM_SYM_VISIBILITY_HIDDEN`).
    pub hidden: bool,
    /// Kept even if nothing references it (`WASM_SYM_NO_STRIP`).
    pub no_strip: bool,
}

impl SymbolFlags {
    fn bits(self) -> u32 {
        let mut f = match self.binding {
            Binding::Global => 0,
            Binding::Weak => FLAG_WEAK,
            Binding::Local => FLAG_LOCAL,
        };
        if self.hidden {
            f |= FLAG_HIDDEN;
        }
        if self.no_strip {
            f |= FLAG_NO_STRIP;
        }
        f
    }
}

const FLAG_WEAK: u32 = 0x1;
const FLAG_LOCAL: u32 = 0x2;
const FLAG_HIDDEN: u32 = 0x4;
const FLAG_UNDEFINED: u32 = 0x10;
const FLAG_EXPLICIT_NAME: u32 = 0x40;
const FLAG_NO_STRIP: u32 = 0x80;
const FLAG_TLS: u32 = 0x100;

const SEG_FLAG_TLS: u32 = 0x2;
const SEG_FLAG_RETAIN: u32 = 0x4;

/// Where a data symbol lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[non_exhaustive]
pub enum DataStorage {
    /// Once in the program, at a fixed address.
    #[default]
    Static,
    /// Once per thread, at an offset from `__tls_base`.
    ThreadLocal,
}

/// Whether a global can be written.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mutability {
    /// The global is read-only.
    Immutable,
    /// The global can be written with `global.set`.
    Mutable,
}

/// A relocated field in a data object: a pointer-sized address.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub struct DataReloc {
    /// Byte offset of the field in the object.
    pub offset: u32,
    /// The function (its table slot is written) or data symbol (its address
    /// plus `addend` is written).
    pub symbol: SymbolId,
    /// Added to the address of a data symbol; ignored for functions.
    pub addend: i64,
}

impl DataReloc {
    /// A reference to `symbol` (plus `addend` for a data symbol) in the field
    /// at `offset`.
    pub fn new(offset: u32, symbol: SymbolId, addend: i64) -> Self {
        DataReloc {
            offset,
            symbol,
            addend,
        }
    }
}

/// The contents of a data object.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub struct DataDef {
    /// The initial bytes; for zero-initialized data, all zeros.
    pub bytes: Vec<u8>,
    /// Alignment in bytes (a power of two).
    pub align: u32,
    /// Whether the program may write to it (`.data`/`.bss` rather than
    /// `.rodata`).
    pub mutability: Mutability,
    /// The fields of `bytes` that hold addresses.
    pub relocs: Vec<DataReloc>,
}

impl DataDef {
    /// A data object holding `bytes`, aligned to `align` bytes (a power of
    /// two), with no relocated fields. Add any to [`DataDef::relocs`].
    pub fn new(bytes: Vec<u8>, align: u32, mutability: Mutability) -> Self {
        DataDef {
            bytes,
            align,
            mutability,
            relocs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum SymKind {
    Func {
        ty: TypeIndex,
        /// Index into `ObjectBuilder::funcs` once defined.
        def: Option<usize>,
    },
    Data {
        storage: DataStorage,
        /// Index into `ObjectBuilder::segments` once defined.
        def: Option<usize>,
    },
    Global {
        ty: ValType,
        mutability: Mutability,
    },
    Table,
}

#[derive(Debug, Clone)]
struct Symbol {
    name: String,
    flags: SymbolFlags,
    kind: SymKind,
    /// For an undefined function: the wasm module and field it is imported
    /// as, if not `env` and its name.
    import: Option<(String, String)>,
}

#[derive(Debug, Clone)]
struct Segment {
    symbol: SymbolId,
    def: DataDef,
}

/// Builds a relocatable object file. See the [module documentation](self).
///
/// # Example
///
/// An object defining a string constant:
///
/// ```
/// use clif_to_wasm::AddressType;
/// use clif_to_wasm::object::{DataDef, DataStorage, Mutability, ObjectBuilder, SymbolFlags};
///
/// let mut obj = ObjectBuilder::new(AddressType::I32);
/// let greeting = obj.declare_data("greeting", SymbolFlags::default(), DataStorage::Static);
/// obj.define_data(
///     greeting,
///     DataDef::new(b"hello\0".to_vec(), 1, Mutability::Immutable),
/// );
/// let object_file = obj.finish();
/// # wasmparser::validate(&object_file)?;
/// # Ok::<(), wasmparser::BinaryReaderError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ObjectBuilder {
    address_type: AddressType,
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    type_map: HashMap<(Vec<ValType>, Vec<ValType>), TypeIndex>,
    symbols: Vec<Symbol>,
    funcs: Vec<(SymbolId, CompiledFunction)>,
    segments: Vec<Segment>,
    init_funcs: Vec<(u32, SymbolId)>,
    features: Vec<String>,
}

impl ObjectBuilder {
    /// An empty object for a linear memory with addresses of type
    /// `address_type`.
    pub fn new(address_type: AddressType) -> Self {
        ObjectBuilder {
            address_type,
            types: Vec::new(),
            type_map: HashMap::new(),
            symbols: Vec::new(),
            funcs: Vec::new(),
            segments: Vec::new(),
            init_funcs: Vec::new(),
            features: Vec::new(),
        }
    }

    /// The index of a function type in the type section, for
    /// [`crate::Env::type_index`].
    pub fn type_index(&mut self, params: &[ValType], results: &[ValType]) -> TypeIndex {
        let key = (params.to_vec(), results.to_vec());
        if let Some(&i) = self.type_map.get(&key) {
            return i;
        }
        let i = TypeIndex(self.types.len() as u32);
        self.types.push(key.clone());
        self.type_map.insert(key, i);
        i
    }

    fn symbol(&mut self, name: &str, flags: SymbolFlags, kind: SymKind) -> SymbolId {
        let id = SymbolId(self.symbols.len() as u32);
        self.symbols.push(Symbol {
            name: name.to_owned(),
            flags,
            kind,
            import: None,
        });
        id
    }

    /// Declare a function symbol. It is imported unless defined with
    /// [`Self::define_function`].
    pub fn declare_function(
        &mut self,
        name: &str,
        flags: SymbolFlags,
        params: &[ValType],
        results: &[ValType],
    ) -> SymbolId {
        let ty = self.type_index(params, results);
        self.symbol(name, flags, SymKind::Func { ty, def: None })
    }

    /// Declare a function imported as `field` from the wasm module `module`
    /// (Rust's `#[link(wasm_import_module = "...")]`), under the symbol name
    /// `name`. Functions declared with [`Self::declare_function`] and not defined
    /// are imported from `env` under their symbol name instead.
    pub fn import_function(
        &mut self,
        name: &str,
        flags: SymbolFlags,
        module: &str,
        field: &str,
        params: &[ValType],
        results: &[ValType],
    ) -> SymbolId {
        let s = self.declare_function(name, flags, params, results);
        self.symbols[s.0 as usize].import = Some((module.to_owned(), field.to_owned()));
        s
    }

    /// Declare a data symbol. It is undefined unless defined with
    /// [`Self::define_data`].
    pub fn declare_data(
        &mut self,
        name: &str,
        flags: SymbolFlags,
        storage: DataStorage,
    ) -> SymbolId {
        self.symbol(name, flags, SymKind::Data { storage, def: None })
    }

    /// Declare a global the linker provides, such as `__stack_pointer`.
    pub fn declare_global(&mut self, name: &str, ty: ValType, mutability: Mutability) -> SymbolId {
        self.symbol(
            name,
            SymbolFlags::default(),
            SymKind::Global { ty, mutability },
        )
    }

    /// Declare a table the linker provides: `__indirect_function_table`.
    pub fn declare_table(&mut self, name: &str) -> SymbolId {
        let flags = SymbolFlags {
            no_strip: true,
            ..SymbolFlags::default()
        };
        self.symbol(name, flags, SymKind::Table)
    }

    /// Define a declared function with code compiled for this object.
    ///
    /// # Panics
    ///
    /// Panics if `sym` is not a function symbol of this object, or is already
    /// defined.
    pub fn define_function(&mut self, sym: SymbolId, func: CompiledFunction) {
        let n = self.funcs.len();
        match &mut self.symbols[sym.0 as usize].kind {
            SymKind::Func { def, .. } => {
                assert!(def.is_none(), "function defined twice");
                *def = Some(n);
            }
            _ => panic!("not a function symbol"),
        }
        self.funcs.push((sym, func));
    }

    /// Define a declared data symbol.
    ///
    /// # Panics
    ///
    /// Panics if `data.align` is not a power of two, if a relocation's
    /// pointer-sized field does not fit in `data.bytes`, or if `sym` is not a
    /// data symbol of this object or is already defined.
    pub fn define_data(&mut self, sym: SymbolId, data: DataDef) {
        assert!(
            data.align.is_power_of_two(),
            "define_data: alignment {} is not a power of two",
            data.align
        );
        let ptr_size = self.address_type.bytes();
        for r in &data.relocs {
            assert!(
                reloc_fits(r.offset, ptr_size, data.bytes.len()),
                "define_data: relocation at offset {} does not fit in {} bytes",
                r.offset,
                data.bytes.len()
            );
        }
        let n = self.segments.len();
        match &mut self.symbols[sym.0 as usize].kind {
            SymKind::Data { def, .. } => {
                assert!(def.is_none(), "data defined twice");
                *def = Some(n);
            }
            _ => panic!("not a data symbol"),
        }
        self.segments.push(Segment {
            symbol: sym,
            def: data,
        });
    }

    /// Run function `sym` (of type `[] -> []`) at startup, lower
    /// `priority` first.
    pub fn init_function(&mut self, priority: u32, sym: SymbolId) {
        self.init_funcs.push((priority, sym));
    }

    /// Record a wasm feature the code uses (`target_features` section), such
    /// as `"atomics"` or `"bulk-memory"`. Linkers check these: shared memory
    /// requires every object to declare `atomics` and `bulk-memory`.
    pub fn add_feature(&mut self, name: &str) {
        if !self.features.iter().any(|f| f == name) {
            self.features.push(name.to_owned());
        }
    }

    /// Change the attributes of a symbol.
    pub fn set_flags(&mut self, sym: SymbolId, flags: SymbolFlags) {
        self.symbols[sym.0 as usize].flags = flags;
    }

    /// The name of a symbol.
    pub fn symbol_name(&self, sym: SymbolId) -> &str {
        &self.symbols[sym.0 as usize].name
    }

    /// Whether a data symbol is thread-local.
    pub fn is_tls(&self, sym: SymbolId) -> bool {
        matches!(
            self.symbols[sym.0 as usize].kind,
            SymKind::Data {
                storage: DataStorage::ThreadLocal,
                ..
            }
        )
    }

    /// Whether a symbol is a function.
    pub fn is_function(&self, sym: SymbolId) -> bool {
        matches!(self.symbols[sym.0 as usize].kind, SymKind::Func { .. })
    }

    /// Drop the globals and tables nothing refers to, so that an object only
    /// imports what it uses, and renumber the symbols.
    fn prune(&mut self) {
        let mut used = vec![false; self.symbols.len()];
        for (_, f) in &self.funcs {
            for r in &f.relocs {
                if r.kind != RelocKind::TypeIndexLeb {
                    used[r.index as usize] = true;
                }
            }
        }
        let keep: Vec<bool> = self
            .symbols
            .iter()
            .zip(&used)
            .map(|(s, &u)| u || !matches!(s.kind, SymKind::Global { .. } | SymKind::Table))
            .collect();
        if keep.iter().all(|&k| k) {
            return;
        }
        let mut new = vec![u32::MAX; self.symbols.len()];
        let mut n = 0;
        for (i, &k) in keep.iter().enumerate() {
            if k {
                new[i] = n;
                n += 1;
            }
        }
        let mut i = 0;
        self.symbols.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let remap = |s: &mut SymbolId| s.0 = new[s.0 as usize];
        for (sym, f) in &mut self.funcs {
            remap(sym);
            for r in &mut f.relocs {
                if r.kind != RelocKind::TypeIndexLeb {
                    r.index = new[r.index as usize];
                }
            }
        }
        for seg in &mut self.segments {
            remap(&mut seg.symbol);
            for r in &mut seg.def.relocs {
                remap(&mut r.symbol);
            }
        }
        for (_, sym) in &mut self.init_funcs {
            remap(sym);
        }
    }

    /// Write the object file.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.prune();
        // Index spaces: imports first, then definitions.
        let mut index = vec![0u32; self.symbols.len()];
        let mut func_imports = Vec::new();
        let mut global_imports = Vec::new();
        let mut table_imports = Vec::new();
        for (i, s) in self.symbols.iter().enumerate() {
            match s.kind {
                SymKind::Func { def: None, ty } => {
                    index[i] = func_imports.len() as u32;
                    func_imports.push((i, ty));
                }
                SymKind::Global { ty, mutability } => {
                    index[i] = global_imports.len() as u32;
                    global_imports.push((i, ty, mutability));
                }
                SymKind::Table => {
                    index[i] = table_imports.len() as u32;
                    table_imports.push(i);
                }
                _ => {}
            }
        }
        for (n, &(sym, _)) in self.funcs.iter().enumerate() {
            index[sym.0 as usize] = (func_imports.len() + n) as u32;
        }
        // Segment addresses in the object's own address space.
        let mut seg_addr = Vec::with_capacity(self.segments.len());
        let mut addr = 0u64;
        for seg in &self.segments {
            addr = addr.next_multiple_of(seg.def.align as u64);
            seg_addr.push(addr);
            addr += seg.def.bytes.len() as u64;
        }
        let sym_addr = |sym: u32| -> u64 {
            match self.symbols[sym as usize].kind {
                SymKind::Data { def: Some(seg), .. } => seg_addr[seg],
                _ => 0,
            }
        };

        let mut out = Vec::new();
        out.extend_from_slice(b"\0asm");
        out.extend_from_slice(&1u32.to_le_bytes());

        // `finish` always writes the type, import, function, code and data
        // sections in that order, so relocation sections can name the code and
        // data sections by their predetermined indexes.
        const CODE_SECTION: u32 = 3;
        const DATA_SECTION: u32 = 4;

        // Type section.
        let mut types = wasm_encoder::TypeSection::new();
        for (p, r) in &self.types {
            types.ty().function(p.iter().copied(), r.iter().copied());
        }
        types.append_to(&mut out);

        // Import section.
        let mut imports = wasm_encoder::ImportSection::new();
        let mem_ty = wasm_encoder::MemoryType {
            minimum: 0,
            maximum: None,
            memory64: self.address_type == AddressType::I64,
            shared: false,
            page_size_log2: None,
        };
        imports.import("env", "__linear_memory", mem_ty);
        for &(i, ty) in &func_imports {
            let s = &self.symbols[i];
            let (module, field) = match &s.import {
                Some((m, f)) => (m.as_str(), f.as_str()),
                None => ("env", s.name.as_str()),
            };
            imports.import(module, field, wasm_encoder::EntityType::Function(ty.0));
        }
        for &(i, ty, mutability) in &global_imports {
            let s = &self.symbols[i];
            let gt = wasm_encoder::GlobalType {
                val_type: ty,
                mutable: mutability == Mutability::Mutable,
                shared: false,
            };
            imports.import("env", &s.name, gt);
        }
        for &i in &table_imports {
            let tt = wasm_encoder::TableType {
                element_type: wasm_encoder::RefType::FUNCREF,
                table64: false,
                minimum: 0,
                maximum: None,
                shared: false,
            };
            imports.import("env", &self.symbols[i].name, tt);
        }
        imports.append_to(&mut out);

        // Function section.
        let mut funcs = wasm_encoder::FunctionSection::new();
        for &(sym, _) in &self.funcs {
            let SymKind::Func { ty, .. } = self.symbols[sym.0 as usize].kind else {
                unreachable!()
            };
            funcs.function(ty.0);
        }
        funcs.append_to(&mut out);

        // Code section, with relocated fields filled in.
        let mut code = Vec::new();
        (self.funcs.len() as u32).encode(&mut code);
        let mut code_relocs = Vec::new();
        for (_, f) in &self.funcs {
            let mut head = Vec::new();
            (f.locals.len() as u32).encode(&mut head);
            for &(n, ty) in &f.locals {
                n.encode(&mut head);
                ty.encode(&mut head);
            }
            ((head.len() + f.body.len()) as u32).encode(&mut code);
            code.extend_from_slice(&head);
            let base = code.len() as u32;
            let mut body = f.body.clone();
            for r in &f.relocs {
                let value = match r.kind {
                    RelocKind::TypeIndexLeb => r.index as i64,
                    RelocKind::FunctionIndexLeb
                    | RelocKind::GlobalIndexLeb
                    | RelocKind::TableNumberLeb
                    | RelocKind::TagIndexLeb => index[r.index as usize] as i64,
                    RelocKind::TableIndexSleb | RelocKind::TableIndexSleb64 => 0,
                    _ => sym_addr(r.index) as i64 + r.addend,
                };
                patch(&mut body, r.offset as usize, r.kind, value);
                code_relocs.push((base + r.offset, *r));
            }
            code.extend_from_slice(&body);
        }
        section(&mut out, 10, &code);

        // Data section: one active segment per data object.
        let mut data = Vec::new();
        (self.segments.len() as u32).encode(&mut data);
        let mut data_relocs = Vec::new();
        for (seg, &addr) in self.segments.iter().zip(&seg_addr) {
            data.push(0); // active, memory 0
            match self.address_type {
                AddressType::I32 => {
                    data.push(0x41);
                    (addr as i32).encode(&mut data);
                }
                AddressType::I64 => {
                    data.push(0x42);
                    (addr as i64).encode(&mut data);
                }
            }
            data.push(0x0b);
            (seg.def.bytes.len() as u32).encode(&mut data);
            let base = data.len() as u32;
            let mut bytes = seg.def.bytes.clone();
            for r in &seg.def.relocs {
                let func = matches!(self.symbols[r.symbol.0 as usize].kind, SymKind::Func { .. });
                let kind = match (func, self.address_type) {
                    (true, AddressType::I32) => RelocKind::TableIndexI32,
                    (true, AddressType::I64) => RelocKind::TableIndexI64,
                    (false, AddressType::I32) => RelocKind::MemoryAddrI32,
                    (false, AddressType::I64) => RelocKind::MemoryAddrI64,
                };
                let value = if func {
                    0
                } else {
                    sym_addr(r.symbol.0) as i64 + r.addend
                };
                patch(&mut bytes, r.offset as usize, kind, value);
                data_relocs.push((
                    base + r.offset,
                    crate::Reloc {
                        offset: r.offset,
                        kind,
                        index: r.symbol.0,
                        addend: if func { 0 } else { r.addend },
                    },
                ));
            }
            data.extend_from_slice(&bytes);
        }
        section(&mut out, 11, &data);

        // The linking section.
        let mut linking = Vec::new();
        2u32.encode(&mut linking); // version
        {
            let mut sub = Vec::new();
            (self.segments.len() as u32).encode(&mut sub);
            for seg in &self.segments {
                let sym = &self.symbols[seg.symbol.0 as usize];
                let SymKind::Data { storage, .. } = sym.kind else {
                    unreachable!()
                };
                let tls = storage == DataStorage::ThreadLocal;
                let zero = seg.def.bytes.iter().all(|&b| b == 0) && seg.def.relocs.is_empty();
                let prefix = match (storage, seg.def.mutability, zero) {
                    (DataStorage::ThreadLocal, _, false) => ".tdata.",
                    (DataStorage::ThreadLocal, _, true) => ".tbss.",
                    (DataStorage::Static, Mutability::Immutable, _) => ".rodata.",
                    (DataStorage::Static, Mutability::Mutable, false) => ".data.",
                    (DataStorage::Static, Mutability::Mutable, true) => ".bss.",
                };
                format!("{prefix}{}", sym.name).as_str().encode(&mut sub);
                seg.def.align.trailing_zeros().encode(&mut sub);
                let mut flags = 0;
                if tls {
                    flags |= SEG_FLAG_TLS;
                }
                if sym.flags.no_strip {
                    flags |= SEG_FLAG_RETAIN;
                }
                flags.encode(&mut sub);
            }
            section(&mut linking, 5, &sub);
        }
        if !self.init_funcs.is_empty() {
            let mut sub = Vec::new();
            (self.init_funcs.len() as u32).encode(&mut sub);
            for &(prio, sym) in &self.init_funcs {
                prio.encode(&mut sub);
                sym.0.encode(&mut sub);
            }
            section(&mut linking, 6, &sub);
        }
        {
            let mut sub = Vec::new();
            (self.symbols.len() as u32).encode(&mut sub);
            for (i, s) in self.symbols.iter().enumerate() {
                let mut flags = s.flags.bits();
                match s.kind {
                    SymKind::Func { def, .. } => {
                        if def.is_none() {
                            flags |= FLAG_UNDEFINED;
                        }
                        // An import's symbol takes the import's name unless
                        // it has its own.
                        let explicit = def.is_none() && s.import.is_some();
                        if explicit {
                            flags |= FLAG_EXPLICIT_NAME;
                        }
                        sub.push(0);
                        flags.encode(&mut sub);
                        index[i].encode(&mut sub);
                        if def.is_some() || explicit {
                            s.name.as_str().encode(&mut sub);
                        }
                    }
                    SymKind::Data { storage, def } => {
                        if storage == DataStorage::ThreadLocal {
                            flags |= FLAG_TLS;
                        }
                        if def.is_none() {
                            flags |= FLAG_UNDEFINED;
                        }
                        sub.push(1);
                        flags.encode(&mut sub);
                        s.name.as_str().encode(&mut sub);
                        if let Some(seg) = def {
                            (seg as u32).encode(&mut sub);
                            0u32.encode(&mut sub);
                            (self.segments[seg].def.bytes.len() as u32).encode(&mut sub);
                        }
                    }
                    SymKind::Global { .. } | SymKind::Table => {
                        flags |= FLAG_UNDEFINED;
                        sub.push(if matches!(s.kind, SymKind::Table) {
                            5
                        } else {
                            2
                        });
                        flags.encode(&mut sub);
                        index[i].encode(&mut sub);
                    }
                }
            }
            section(&mut linking, 8, &sub);
        }
        custom(&mut out, "linking", &linking);

        for (name, target, relocs) in [
            ("reloc.CODE", CODE_SECTION, &code_relocs),
            ("reloc.DATA", DATA_SECTION, &data_relocs),
        ] {
            if relocs.is_empty() {
                continue;
            }
            let mut relocs = relocs.clone();
            relocs.sort_by_key(|&(offset, _)| offset);
            let mut payload = Vec::new();
            target.encode(&mut payload);
            (relocs.len() as u32).encode(&mut payload);
            for &(offset, r) in &relocs {
                payload.push(r.kind as u8);
                offset.encode(&mut payload);
                r.index.encode(&mut payload);
                if r.kind.has_addend() {
                    r.addend.encode(&mut payload);
                }
            }
            custom(&mut out, name, &payload);
        }

        if !self.features.is_empty() {
            let mut payload = Vec::new();
            (self.features.len() as u32).encode(&mut payload);
            for f in &self.features {
                payload.push(b'+');
                f.as_str().encode(&mut payload);
            }
            custom(&mut out, "target_features", &payload);
        }
        out
    }
}

/// Appends a section or a linking subsection with its id the payload length,
/// and finally the payload bytes.
fn section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    (payload.len() as u32).encode(out);
    out.extend_from_slice(payload);
}

fn custom(out: &mut Vec<u8>, name: &str, payload: &[u8]) {
    CustomSection {
        name: name.into(),
        data: Cow::Borrowed(payload),
    }
    .append_to(out);
}

/// Whether a `size`-byte relocated field at `offset` fits in `len` bytes.
pub(crate) fn reloc_fits(offset: u32, size: u32, len: usize) -> bool {
    (offset as usize)
        .checked_add(size as usize)
        .is_some_and(|end| end <= len)
}

/// Overwrite the relocated field of kind `kind` at `at` with `value`.
fn patch(buf: &mut [u8], at: usize, kind: RelocKind, value: i64) {
    use RelocKind::*;
    let mut put = |bytes: &[u8]| buf[at..at + bytes.len()].copy_from_slice(bytes);
    match kind {
        FunctionIndexLeb | TypeIndexLeb | GlobalIndexLeb | TableNumberLeb | TagIndexLeb => {
            put(&crate::emit::padded_leb::<5>(value as u32 as i64));
        }
        TableIndexSleb | MemoryAddrSleb | MemoryAddrTlsSleb => {
            put(&crate::emit::padded_leb::<5>(value as i32 as i64));
        }
        TableIndexSleb64 | MemoryAddrSleb64 | MemoryAddrTlsSleb64 => {
            put(&crate::emit::padded_leb::<10>(value));
        }
        TableIndexI32 | MemoryAddrI32 => {
            put(&(value as u32).to_le_bytes());
        }
        TableIndexI64 | MemoryAddrI64 => {
            put(&(value as u64).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the relocated field of kind `kind` at the start of `field`.
    /// Returns its value and width in bytes.
    fn read_field(field: &[u8], kind: RelocKind) -> (i64, usize) {
        use RelocKind::*;
        let mut r = wasmparser::BinaryReader::new(field, 0);
        let (value, width) = match kind {
            FunctionIndexLeb | TypeIndexLeb | GlobalIndexLeb | TableNumberLeb | TagIndexLeb => {
                (r.read_var_u32().unwrap().into(), 5)
            }
            TableIndexSleb | MemoryAddrSleb | MemoryAddrTlsSleb => {
                (r.read_var_i32().unwrap().into(), 5)
            }
            TableIndexSleb64 | MemoryAddrSleb64 | MemoryAddrTlsSleb64 => {
                (r.read_var_i64().unwrap(), 10)
            }
            TableIndexI32 | MemoryAddrI32 => {
                let bytes = field[..4].try_into().unwrap();
                return (u32::from_le_bytes(bytes).into(), 4);
            }
            TableIndexI64 | MemoryAddrI64 => {
                let bytes = field[..8].try_into().unwrap();
                return (i64::from_le_bytes(bytes), 8);
            }
        };
        assert_eq!(
            r.current_position(),
            width,
            "{kind:?}: LEB padded to full width"
        );
        (value, width)
    }

    /// `patch` writes exactly the field of each relocation kind. The buffer
    /// starts nonzero so that a field left unwritten cannot pass.
    #[test]
    fn patch_writes_every_kind() {
        use RelocKind::*;
        // `read_field` matches exhaustively; add new kinds here too.
        let kinds = [
            FunctionIndexLeb,
            TypeIndexLeb,
            GlobalIndexLeb,
            TableNumberLeb,
            TagIndexLeb,
            TableIndexSleb,
            MemoryAddrSleb,
            MemoryAddrTlsSleb,
            TableIndexSleb64,
            MemoryAddrSleb64,
            MemoryAddrTlsSleb64,
            TableIndexI32,
            MemoryAddrI32,
            TableIndexI64,
            MemoryAddrI64,
        ];
        let at = 3;
        for kind in kinds {
            let wide = matches!(
                kind,
                TableIndexSleb64
                    | MemoryAddrSleb64
                    | MemoryAddrTlsSleb64
                    | TableIndexI64
                    | MemoryAddrI64
            );
            // 0x8000_0010 is a 32-bit address that a signed LEB stores as
            // a negative `i32`.
            let values: [i64; 2] = if wide {
                [0x1234_5678_9abc_def0, -0x10]
            } else {
                [0x1234_5678, 0x8000_0010]
            };
            for value in values {
                let mut buf = [0xaa; 16];
                patch(&mut buf, at, kind, value);
                // Decoding also checks the LEB is well formed, including
                // the sign bits of its last byte.
                let (got, width) = read_field(&buf[at..], kind);
                if wide {
                    assert_eq!(got, value, "{kind:?}: field value");
                } else {
                    assert_eq!(got as u32, value as u32, "{kind:?}: field value");
                }
                assert!(
                    buf[..at]
                        .iter()
                        .chain(&buf[at + width..])
                        .all(|&b| b == 0xaa),
                    "{kind:?}: bytes outside the field are untouched"
                );
            }
        }
    }
}
