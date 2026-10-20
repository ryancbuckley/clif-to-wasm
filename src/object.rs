//! Relocatable wasm object files, as defined by the wasm [tool conventions]
//! and consumed by `wasm-ld` and `wild`.
//!
//! An [`ObjectBuilder`] collects symbols — functions, data, and the globals
//! and tables a linker provides (such as `__stack_pointer` and
//! `__indirect_function_table`) — and the definitions of some of them.
//! Functions are compiled with [`crate::ModuleLayout::relocatable`] set, with
//! an [`crate::Env`] that answers with symbol indices; [`ObjectBuilder::finish`]
//! then assigns the final indices, fills in the placeholders, and writes the
//! `linking` and `reloc.*` sections. Undefined functions, globals and tables
//! become imports from `env`, as LLVM writes them.
//!
//! The data of an object lives in its own address space: segments are laid
//! out one after the other from address 0, and relocated fields hold
//! addresses in that space until the linker moves them.
//!
//! [tool conventions]: https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md

use crate::{CompiledFunction, RelocKind};
use std::collections::HashMap;
use wasm_encoder::{Encode, ValType};

/// An entry of the object's symbol table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SymbolId(pub u32);

/// How a symbol binds across objects.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SymbolFlags {
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
const FLAG_NO_STRIP: u32 = 0x80;
const FLAG_TLS: u32 = 0x100;

const SEG_FLAG_TLS: u32 = 0x2;
const SEG_FLAG_RETAIN: u32 = 0x4;

/// A relocated field in a data object: a pointer-sized address.
#[derive(Clone, Copy, Debug)]
pub struct DataReloc {
    /// Byte offset of the field in the object.
    pub offset: u32,
    /// The function (its table slot is written) or data symbol (its address
    /// plus `addend` is written).
    pub symbol: SymbolId,
    pub addend: i64,
}

/// The contents of a data object.
#[derive(Clone, Debug)]
pub struct DataDef {
    /// The initial bytes; for zero-initialized data, all zeros.
    pub bytes: Vec<u8>,
    /// Alignment in bytes (a power of two).
    pub align: u32,
    /// Whether the program may write to it (`.data`/`.bss` rather than
    /// `.rodata`).
    pub writable: bool,
    pub relocs: Vec<DataReloc>,
}

enum SymKind {
    Func {
        ty: u32,
        /// Index into `ObjectBuilder::funcs` once defined.
        def: Option<usize>,
    },
    Data {
        tls: bool,
        /// Index into `ObjectBuilder::segments` once defined.
        def: Option<usize>,
    },
    Global {
        ty: ValType,
        mutable: bool,
    },
    Table,
}

struct Symbol {
    name: String,
    flags: SymbolFlags,
    kind: SymKind,
}

struct Segment {
    symbol: SymbolId,
    def: DataDef,
}

/// Builds a relocatable object file. See the [module documentation](self).
pub struct ObjectBuilder {
    memory64: bool,
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    type_map: HashMap<(Vec<ValType>, Vec<ValType>), u32>,
    symbols: Vec<Symbol>,
    funcs: Vec<(SymbolId, CompiledFunction)>,
    segments: Vec<Segment>,
    init_funcs: Vec<(u32, SymbolId)>,
    features: Vec<String>,
}

impl ObjectBuilder {
    /// An empty object for a 32-bit or 64-bit (memory64) linear memory.
    pub fn new(memory64: bool) -> Self {
        ObjectBuilder {
            memory64,
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
    pub fn type_index(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
        let key = (params.to_vec(), results.to_vec());
        if let Some(&i) = self.type_map.get(&key) {
            return i;
        }
        let i = self.types.len() as u32;
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
        });
        id
    }

    /// Declare a function symbol. It is imported unless defined with
    /// [`Self::define_function`].
    pub fn function(
        &mut self,
        name: &str,
        flags: SymbolFlags,
        params: &[ValType],
        results: &[ValType],
    ) -> SymbolId {
        let ty = self.type_index(params, results);
        self.symbol(name, flags, SymKind::Func { ty, def: None })
    }

    /// Declare a data symbol, thread-local if `tls`. It is undefined unless
    /// defined with [`Self::define_data`].
    pub fn data(&mut self, name: &str, flags: SymbolFlags, tls: bool) -> SymbolId {
        self.symbol(name, flags, SymKind::Data { tls, def: None })
    }

    /// Declare a global the linker provides, such as `__stack_pointer`.
    pub fn global(&mut self, name: &str, ty: ValType, mutable: bool) -> SymbolId {
        self.symbol(
            name,
            SymbolFlags::default(),
            SymKind::Global { ty, mutable },
        )
    }

    /// Declare a table the linker provides: `__indirect_function_table`.
    pub fn table(&mut self, name: &str) -> SymbolId {
        let flags = SymbolFlags {
            no_strip: true,
            ..SymbolFlags::default()
        };
        self.symbol(name, flags, SymKind::Table)
    }

    /// Define a declared function with code compiled for this object.
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
    pub fn define_data(&mut self, sym: SymbolId, data: DataDef) {
        assert!(data.align.is_power_of_two());
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
    pub fn feature(&mut self, name: &str) {
        if !self.features.iter().any(|f| f == name) {
            self.features.push(name.to_owned());
        }
    }

    /// Change the attributes of a symbol.
    pub fn set_flags(&mut self, sym: SymbolId, flags: SymbolFlags) {
        self.symbols[sym.0 as usize].flags = flags;
    }

    pub fn symbol_name(&self, sym: SymbolId) -> &str {
        &self.symbols[sym.0 as usize].name
    }

    /// Whether a data symbol is thread-local.
    pub fn is_tls(&self, sym: SymbolId) -> bool {
        matches!(
            self.symbols[sym.0 as usize].kind,
            SymKind::Data { tls: true, .. }
        )
    }

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
    pub fn finish(mut self) -> Vec<u8> {
        self.prune();
        // Index spaces: imports first, then definitions.
        let mut index = vec![0u32; self.symbols.len()];
        let mut func_imports = Vec::new();
        let mut global_imports = Vec::new();
        let mut table_imports = Vec::new();
        for (i, s) in self.symbols.iter().enumerate() {
            match s.kind {
                SymKind::Func { def: None, .. } => {
                    index[i] = func_imports.len() as u32;
                    func_imports.push(i);
                }
                SymKind::Global { .. } => {
                    index[i] = global_imports.len() as u32;
                    global_imports.push(i);
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

        // Type section.
        let mut types = wasm_encoder::TypeSection::new();
        for (p, r) in &self.types {
            types.ty().function(p.iter().copied(), r.iter().copied());
        }
        let mut section_index = 0u32;
        section(&mut out, 1, &encoded(&types));
        section_index += 1;

        // Import section.
        let mut imports = wasm_encoder::ImportSection::new();
        let mem_ty = wasm_encoder::MemoryType {
            minimum: 0,
            maximum: None,
            memory64: self.memory64,
            shared: false,
            page_size_log2: None,
        };
        imports.import("env", "__linear_memory", mem_ty);
        for &i in &func_imports {
            let s = &self.symbols[i];
            let SymKind::Func { ty, .. } = s.kind else {
                unreachable!()
            };
            imports.import("env", &s.name, wasm_encoder::EntityType::Function(ty));
        }
        for &i in &global_imports {
            let s = &self.symbols[i];
            let SymKind::Global { ty, mutable } = s.kind else {
                unreachable!()
            };
            let gt = wasm_encoder::GlobalType {
                val_type: ty,
                mutable,
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
        section(&mut out, 2, &encoded(&imports));
        section_index += 1;

        // Function section.
        let mut funcs = wasm_encoder::FunctionSection::new();
        for &(sym, _) in &self.funcs {
            let SymKind::Func { ty, .. } = self.symbols[sym.0 as usize].kind else {
                unreachable!()
            };
            funcs.function(ty);
        }
        section(&mut out, 3, &encoded(&funcs));
        section_index += 1;

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
        let code_section = section_index;
        section(&mut out, 10, &code);
        section_index += 1;

        // Data section: one active segment per data object.
        let mut data = Vec::new();
        (self.segments.len() as u32).encode(&mut data);
        let mut data_relocs = Vec::new();
        for (seg, &addr) in self.segments.iter().zip(&seg_addr) {
            data.push(0); // active, memory 0
            if self.memory64 {
                data.push(0x42);
                (addr as i64).encode(&mut data);
            } else {
                data.push(0x41);
                (addr as i32).encode(&mut data);
            }
            data.push(0x0b);
            (seg.def.bytes.len() as u32).encode(&mut data);
            let base = data.len() as u32;
            let mut bytes = seg.def.bytes.clone();
            for r in &seg.def.relocs {
                let func = matches!(self.symbols[r.symbol.0 as usize].kind, SymKind::Func { .. });
                let kind = match (func, self.memory64) {
                    (true, false) => RelocKind::TableIndexI32,
                    (true, true) => RelocKind::TableIndexI64,
                    (false, false) => RelocKind::MemoryAddrI32,
                    (false, true) => RelocKind::MemoryAddrI64,
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
        let data_section = section_index;
        section(&mut out, 11, &data);

        // The linking section.
        let mut linking = Vec::new();
        2u32.encode(&mut linking); // version
        {
            let mut sub = Vec::new();
            (self.segments.len() as u32).encode(&mut sub);
            for seg in &self.segments {
                let sym = &self.symbols[seg.symbol.0 as usize];
                let SymKind::Data { tls, .. } = sym.kind else {
                    unreachable!()
                };
                let zero = seg.def.bytes.iter().all(|&b| b == 0) && seg.def.relocs.is_empty();
                let prefix = match (tls, seg.def.writable, zero) {
                    (true, _, false) => ".tdata.",
                    (true, _, true) => ".tbss.",
                    (false, false, _) => ".rodata.",
                    (false, true, false) => ".data.",
                    (false, true, true) => ".bss.",
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
            subsection(&mut linking, 5, &sub);
        }
        if !self.init_funcs.is_empty() {
            let mut sub = Vec::new();
            (self.init_funcs.len() as u32).encode(&mut sub);
            for &(prio, sym) in &self.init_funcs {
                prio.encode(&mut sub);
                sym.0.encode(&mut sub);
            }
            subsection(&mut linking, 6, &sub);
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
                        sub.push(0);
                        flags.encode(&mut sub);
                        index[i].encode(&mut sub);
                        if def.is_some() {
                            s.name.as_str().encode(&mut sub);
                        }
                    }
                    SymKind::Data { tls, def } => {
                        if tls {
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
            subsection(&mut linking, 8, &sub);
        }
        custom(&mut out, "linking", &linking);

        for (name, target, relocs) in [
            ("reloc.CODE", code_section, &code_relocs),
            ("reloc.DATA", data_section, &data_relocs),
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

/// The payload of a `wasm_encoder` section, whose `Encode` writes it with
/// a length prefix.
fn encoded(s: &impl Encode) -> Vec<u8> {
    let mut v = Vec::new();
    s.encode(&mut v);
    let prefix = v
        .iter()
        .position(|&b| b & 0x80 == 0)
        .map_or(v.len(), |i| i + 1);
    v.drain(..prefix);
    v
}

fn section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    (payload.len() as u32).encode(out);
    out.extend_from_slice(payload);
}

fn subsection(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    (payload.len() as u32).encode(out);
    out.extend_from_slice(payload);
}

fn custom(out: &mut Vec<u8>, name: &str, payload: &[u8]) {
    let mut content = Vec::new();
    name.encode(&mut content);
    content.extend_from_slice(payload);
    section(out, 0, &content);
}

/// Overwrite the relocated field of kind `kind` at `at` with `value`.
fn patch(buf: &mut [u8], at: usize, kind: RelocKind, value: i64) {
    use RelocKind::*;
    let mut tmp = Vec::new();
    match kind {
        FunctionIndexLeb | TypeIndexLeb | GlobalIndexLeb | TableNumberLeb | TagIndexLeb => {
            crate::emit::write_padded_uleb(&mut tmp, value as u32)
        }
        TableIndexSleb | MemoryAddrSleb | MemoryAddrTlsSleb => {
            crate::emit::write_padded_sleb(&mut tmp, value as i32 as i64, 5)
        }
        TableIndexSleb64 | MemoryAddrSleb64 | MemoryAddrTlsSleb64 => {
            crate::emit::write_padded_sleb(&mut tmp, value, 10)
        }
        TableIndexI32 | MemoryAddrI32 => tmp.extend_from_slice(&(value as u32).to_le_bytes()),
        TableIndexI64 | MemoryAddrI64 => tmp.extend_from_slice(&(value as u64).to_le_bytes()),
    }
    buf[at..at + tmp.len()].copy_from_slice(&tmp);
}
