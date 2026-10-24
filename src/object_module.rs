//! A [`cranelift_module::Module`] implementation that emits relocatable wasm
//! objects, allowing `cranelift-module` frontends (such as
//! `rustc_codegen_cranelift`) to target wasm and link with `wasm-ld` or `wild`.
//!
//! Functions are compiled as they are defined with
//! [`crate::ModuleLayout::relocatable`] enabled. Like LLVM objects, the output
//! imports `__stack_pointer`, `__indirect_function_table`, and (when threads
//! are enabled) `__tls_base` from the linker.
//!
//! Because Cranelift has no built-in wasm `TargetIsa` and external crates
//! cannot implement one, [`frontend_isa`] provides the Pulley ISA of matching
//! pointer width so frontends can query pointer size, endianness, and shared
//! flags.

use crate::object::{Binding, DataDef, DataReloc, ObjectBuilder, SymbolFlags, SymbolId};
use crate::{
    CompiledFunction, Config, Env, Error, FuncIndex, GlobalIndex, Name, Result, SymbolValue,
    TableIndex, TypeIndex, compile_function, wasm_signature_in,
};
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::ir::{self, ExternalName, InstructionData, LibCall, Opcode};
use cranelift_codegen::isa::{OwnedTargetIsa, TargetIsa};
use cranelift_codegen::{CodegenError, Context, settings};
use cranelift_module::{
    DataDescription, DataId, FuncId, Init, Linkage, Module, ModuleDeclarations, ModuleError,
    ModuleReloc, ModuleRelocTarget, ModuleResult,
};
use std::collections::HashMap;
use wasm_encoder::ValType;

/// The wasm features LLVM declares for `wasm32-wasip1-threads` objects; a
/// linker producing shared memory requires `atomics` and `bulk-memory`.
pub const THREADS_FEATURES: &[&str] = &[
    "atomics",
    "bulk-memory",
    "bulk-memory-opt",
    "call-indirect-overlong",
    "multivalue",
    "mutable-globals",
    "nontrapping-fptoint",
    "reference-types",
    "sign-ext",
];

/// The ISA to give a frontend that builds IR for this module: Pulley with
/// the pointer width of wasm32 (or wasm64 if `memory64`).
pub fn frontend_isa(flags: settings::Flags, memory64: bool) -> ModuleResult<OwnedTargetIsa> {
    let name = if memory64 { "pulley64" } else { "pulley32" };
    let builder = cranelift_codegen::isa::lookup_by_name(name)
        .map_err(|e| ModuleError::Compilation(CodegenError::Unsupported(e.to_string())))?;
    builder.finish(flags).map_err(ModuleError::Compilation)
}

/// Settings for a [`WasmObjectModule`].
pub struct ObjectOptions {
    /// Code generation settings. The module sets the relocatable layout
    /// (`ModuleLayout::relocatable`, the stack pointer, table and TLS base);
    /// set `module.demote_multi_results` to link with LLVM-compiled code.
    pub config: Config,
    /// Whether the object is for a program with threads (shared memory):
    /// thread-local data then lives in TLS segments, addressed from
    /// `__tls_base`. Otherwise it is ordinary data.
    pub threads: bool,
    /// The wasm features to declare in the `target_features` section.
    pub features: Vec<String>,
    /// Define functions this backend cannot compile as traps instead of
    /// failing, and record why in [`WasmObjectModule::warnings`].
    pub stub_unsupported: bool,
    /// The symbol names of libcalls.
    pub libcall_names: Box<dyn Fn(LibCall) -> String + Send + Sync>,
    /// For the symbol name of an imported function, the wasm module and
    /// field to import it as, if not `env` and the symbol name (Rust's
    /// `#[link(wasm_import_module = "...")]`).
    pub import_names: Box<dyn Fn(&str) -> Option<(String, String)> + Send + Sync>,
}

/// A `cranelift_module::Module` writing a relocatable wasm object.
pub struct WasmObjectModule {
    isa: OwnedTargetIsa,
    declarations: ModuleDeclarations,
    obj: ObjectBuilder,
    config: Config,
    threads: bool,
    stub_unsupported: bool,
    libcall_names: Box<dyn Fn(LibCall) -> String + Send + Sync>,
    import_names: Box<dyn Fn(&str) -> Option<(String, String)> + Send + Sync>,
    funcs: SecondaryMap<FuncId, Option<SymbolId>>,
    data: SecondaryMap<DataId, Option<SymbolId>>,
    defined_funcs: SecondaryMap<FuncId, bool>,
    defined_data: SecondaryMap<DataId, bool>,
    used_data: SecondaryMap<DataId, bool>,
    libcalls: HashMap<LibCall, SymbolId>,
    warnings: Vec<String>,
}

fn unsupported_err(msg: impl Into<String>) -> ModuleError {
    ModuleError::Compilation(CodegenError::Unsupported(msg.into()))
}

fn flags_for(linkage: Linkage) -> SymbolFlags {
    match linkage {
        Linkage::Import | Linkage::Export => SymbolFlags::default(),
        Linkage::Local => SymbolFlags {
            binding: Binding::Local,
            ..SymbolFlags::default()
        },
        Linkage::Preemptible => SymbolFlags {
            binding: Binding::Weak,
            ..SymbolFlags::default()
        },
        Linkage::Hidden => SymbolFlags {
            hidden: true,
            ..SymbolFlags::default()
        },
    }
}

impl WasmObjectModule {
    /// A module whose frontend uses `isa` (see [`frontend_isa`]).
    pub fn new(isa: OwnedTargetIsa, options: ObjectOptions) -> Self {
        let mut config = options.config;
        let memory64 = config.module.memory64;
        let ptr = if memory64 { ValType::I64 } else { ValType::I32 };
        let mut obj = ObjectBuilder::new(memory64);
        // Unused ones are dropped when the object is written.
        let sp = obj.global("__stack_pointer", ptr, true);
        let table = obj.table("__indirect_function_table");
        config.module.relocatable = true;
        config.module.stack_pointer_global = Some(GlobalIndex(sp.0));
        config.module.table_index = Some(TableIndex(table.0));
        config.module.tls_base_global = None;
        if options.threads {
            let tls = obj.global("__tls_base", ptr, true);
            config.module.tls_base_global = Some(GlobalIndex(tls.0));
        }
        config.module.pinned_reg_global = None;
        config.module.atomic_lock_table_addr = None;
        for f in &options.features {
            obj.feature(f);
        }
        WasmObjectModule {
            isa,
            declarations: ModuleDeclarations::default(),
            obj,
            config,
            threads: options.threads,
            stub_unsupported: options.stub_unsupported,
            libcall_names: options.libcall_names,
            import_names: options.import_names,
            funcs: SecondaryMap::new(),
            data: SecondaryMap::new(),
            defined_funcs: SecondaryMap::new(),
            defined_data: SecondaryMap::new(),
            used_data: SecondaryMap::new(),
            libcalls: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    /// Why functions were replaced by traps (`stub_unsupported`).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    fn wasm_sig(&self, sig: &ir::Signature) -> ModuleResult<(Vec<ValType>, Vec<ValType>)> {
        wasm_signature_in(sig, &self.config.module).map_err(|e| unsupported_err(e.to_string()))
    }

    fn func_symbol(&mut self, id: FuncId) -> ModuleResult<SymbolId> {
        if let Some(s) = self.funcs[id] {
            return Ok(s);
        }
        let decl = self.declarations.get_function_decl(id);
        let name = decl.linkage_name(id).into_owned();
        let flags = flags_for(decl.linkage);
        let sig = decl.signature.clone();
        let (p, r) = self.wasm_sig(&sig)?;
        let import = match decl.linkage {
            Linkage::Import => (self.import_names)(&name),
            _ => None,
        };
        let s = match import {
            Some((module, field)) => self.obj.import_function(&name, flags, &module, &field, &p, &r),
            None => self.obj.function(&name, flags, &p, &r),
        };
        self.funcs[id] = Some(s);
        Ok(s)
    }

    fn data_symbol(&mut self, id: DataId) -> SymbolId {
        if let Some(s) = self.data[id] {
            return s;
        }
        let decl = self.declarations.get_data_decl(id);
        let name = decl.linkage_name(id).into_owned();
        let flags = flags_for(decl.linkage);
        let tls = decl.tls && self.threads;
        let s = self.obj.data(&name, flags, tls);
        self.data[id] = Some(s);
        s
    }

    fn libcall_symbol(&mut self, lc: LibCall, sig: &ir::Signature) -> ModuleResult<SymbolId> {
        if let Some(&s) = self.libcalls.get(&lc) {
            return Ok(s);
        }
        let name = (self.libcall_names)(lc);
        let (p, r) = self.wasm_sig(sig)?;
        let s = self.obj.function(&name, SymbolFlags::default(), &p, &r);
        self.libcalls.insert(lc, s);
        Ok(s)
    }

    fn reloc_target(&mut self, target: &ModuleRelocTarget) -> ModuleResult<SymbolId> {
        match *target {
            ModuleRelocTarget::User {
                namespace: 0,
                index,
            } => self.func_symbol(FuncId::from_u32(index)),
            ModuleRelocTarget::User {
                namespace: 1,
                index,
            } => Ok(self.data_symbol(DataId::from_u32(index))),
            ref t => Err(unsupported_err(format!("relocation to {t}"))),
        }
    }

    /// Write the object file.
    pub fn finish(mut self) -> Vec<u8> {
        // Linkage may have changed since a symbol was declared.
        let funcs: Vec<(FuncId, Linkage)> = self
            .declarations
            .get_functions()
            .map(|(id, d)| (id, d.linkage))
            .collect();
        for (id, linkage) in funcs {
            if let Some(s) = self.funcs[id] {
                self.obj.set_flags(s, flags_for(linkage));
            }
        }
        let data: Vec<(DataId, Linkage)> = self
            .declarations
            .get_data_objects()
            .map(|(id, d)| (id, d.linkage))
            .collect();
        for (id, linkage) in data {
            if let Some(s) = self.data[id] {
                let mut flags = flags_for(linkage);
                flags.no_strip = self.used_data[id];
                self.obj.set_flags(s, flags);
            }
        }
        self.obj.finish()
    }
}

/// Resolves names during compilation to the module's symbols.
struct ObjEnv<'a> {
    m: &'a mut WasmObjectModule,
}

impl ObjEnv<'_> {
    fn symbol(&mut self, name: Name<'_>) -> Result<SymbolId> {
        let r = match name {
            Name::User(u) if u.namespace == 0 => self.m.func_symbol(FuncId::from_u32(u.index)),
            Name::User(u) if u.namespace == 1 => Ok(self.m.data_symbol(DataId::from_u32(u.index))),
            Name::LibCall(lc) => self
                .m
                .libcalls
                .get(&lc)
                .copied()
                .ok_or_else(|| unsupported_err(format!("libcall {lc}"))),
            _ => Err(unsupported_err(format!("reference to {name}"))),
        };
        r.map_err(|e| Error::Env(e.to_string()))
    }
}

impl Env for ObjEnv<'_> {
    fn func_index(&mut self, name: Name<'_>) -> Result<FuncIndex> {
        Ok(FuncIndex(self.symbol(name)?.0))
    }

    fn type_index(&mut self, sig: &ir::Signature) -> Result<TypeIndex> {
        // Signatures here are already adjusted for demoted results.
        let (p, r) = crate::wasm_signature(sig)?;
        Ok(TypeIndex(self.m.obj.type_index(&p, &r)))
    }

    fn func_table_slot(&mut self, name: Name<'_>) -> Result<u64> {
        Ok(self.symbol(name)?.0 as u64)
    }

    fn symbol_value(&mut self, name: Name<'_>) -> Result<SymbolValue> {
        let s = self.symbol(name)?;
        if self.m.obj.is_function(s) {
            return Err(Error::Unsupported(format!(
                "symbol_value of function {name}"
            )));
        }
        Ok(SymbolValue::Symbol(s.0))
    }
}

impl Module for WasmObjectModule {
    fn isa(&self) -> &dyn TargetIsa {
        &*self.isa
    }

    fn declarations(&self) -> &ModuleDeclarations {
        &self.declarations
    }

    fn declare_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        signature: &ir::Signature,
    ) -> ModuleResult<FuncId> {
        let (id, _) = self
            .declarations
            .declare_function(name, linkage, signature)?;
        self.func_symbol(id)?;
        Ok(id)
    }

    fn declare_anonymous_function(&mut self, signature: &ir::Signature) -> ModuleResult<FuncId> {
        let id = self.declarations.declare_anonymous_function(signature)?;
        self.func_symbol(id)?;
        Ok(id)
    }

    fn declare_data(
        &mut self,
        name: &str,
        linkage: Linkage,
        writable: bool,
        tls: bool,
    ) -> ModuleResult<DataId> {
        let (id, _) = self
            .declarations
            .declare_data(name, linkage, writable, tls)?;
        self.data_symbol(id);
        Ok(id)
    }

    fn declare_anonymous_data(&mut self, writable: bool, tls: bool) -> ModuleResult<DataId> {
        let id = self.declarations.declare_anonymous_data(writable, tls)?;
        self.data_symbol(id);
        Ok(id)
    }

    fn define_function_with_control_plane(
        &mut self,
        func_id: FuncId,
        ctx: &mut Context,
        ctrl_plane: &mut ControlPlane,
    ) -> ModuleResult<()> {
        let decl = self.declarations.get_function_decl(func_id);
        let name = decl.linkage_name(func_id).into_owned();
        if !decl.linkage.is_definable() {
            return Err(ModuleError::InvalidImportDefinition(name));
        }
        if self.defined_funcs[func_id] {
            return Err(ModuleError::DuplicateDefinition(name));
        }
        self.defined_funcs[func_id] = true;
        let sym = self.func_symbol(func_id)?;

        let flags = self.isa.flags();
        if flags.enable_verifier() {
            ctx.verify(&*self.isa)
                .map_err(|e| ModuleError::Compilation(CodegenError::Verifier(e)))?;
        }
        if flags.opt_level() != settings::OptLevel::None {
            ctx.optimize(&*self.isa, ctrl_plane)
                .map_err(ModuleError::Compilation)?;
        }
        let func = &mut ctx.func;
        if !self.threads {
            // Without threads there is one copy of thread-local data.
            for b in func.layout.blocks().collect::<Vec<_>>() {
                for inst in func.layout.block_insts(b).collect::<Vec<_>>() {
                    if let InstructionData::UnaryGlobalValue { opcode, .. } =
                        &mut func.dfg.insts[inst]
                        && *opcode == Opcode::TlsValue
                    {
                        *opcode = Opcode::SymbolValue;
                    }
                }
            }
        }
        // Declare the libcalls the function uses, with their signatures.
        for ext in func.dfg.ext_funcs.values() {
            if let ExternalName::LibCall(lc) = ext.name {
                let sig = func.dfg.signatures[ext.signature].clone();
                self.libcall_symbol(lc, &sig)?;
            }
        }

        let config = self.config.clone();
        let compiled = compile_function(func, &config, &mut ObjEnv { m: self });
        let compiled = match compiled {
            Ok(c) => c,
            Err(e) if self.stub_unsupported => {
                self.warnings.push(format!("{name}: {e}"));
                let (params, results) = self.wasm_sig(
                    &self
                        .declarations
                        .get_function_decl(func_id)
                        .signature
                        .clone(),
                )?;
                CompiledFunction {
                    params,
                    results,
                    locals: Vec::new(),
                    body: vec![0x00, 0x0b], // unreachable; end
                    relocs: Vec::new(),
                }
            }
            Err(e) => return Err(unsupported_err(format!("{name}: {e}"))),
        };
        self.obj.define_function(sym, compiled);
        Ok(())
    }

    fn define_function_bytes(
        &mut self,
        _func_id: FuncId,
        _alignment: u64,
        _bytes: &[u8],
        _relocs: &[ModuleReloc],
    ) -> ModuleResult<()> {
        Err(unsupported_err("define_function_bytes for wasm"))
    }

    fn define_data(&mut self, data_id: DataId, data: &DataDescription) -> ModuleResult<()> {
        let decl = self.declarations.get_data_decl(data_id);
        let name = decl.linkage_name(data_id).into_owned();
        if !decl.linkage.is_definable() {
            return Err(ModuleError::InvalidImportDefinition(name));
        }
        if self.defined_data[data_id] {
            return Err(ModuleError::DuplicateDefinition(name));
        }
        if let Some(section) = &data.custom_section {
            return Err(unsupported_err(format!(
                "{name}: data in section {section}"
            )));
        }
        self.defined_data[data_id] = true;
        let writable = decl.writable;
        let bytes = match &data.init {
            Init::Uninitialized => panic!("data {name} is not initialized"),
            Init::Zeros { size } => vec![0; *size],
            Init::Bytes { contents } => contents.to_vec(),
        };
        let mut relocs = Vec::new();
        for &(offset, fref) in &data.function_relocs {
            let symbol = self.reloc_target(&data.function_decls[fref])?;
            relocs.push(DataReloc {
                offset,
                symbol,
                addend: 0,
            });
        }
        for &(offset, gv, addend) in &data.data_relocs {
            let symbol = self.reloc_target(&data.data_decls[gv])?;
            relocs.push(DataReloc {
                offset,
                symbol,
                addend,
            });
        }
        let sym = self.data_symbol(data_id);
        self.used_data[data_id] = data.used;
        let align = data.align.unwrap_or(1).max(1) as u32;
        self.obj.define_data(
            sym,
            DataDef {
                bytes,
                align,
                writable,
                relocs,
            },
        );
        Ok(())
    }
}
