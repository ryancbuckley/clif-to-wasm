//! Encoding of a function body: the instruction byte stream, and the locals
//! created during emission.

use wasm_encoder::{Encode, Instruction as I, ValType};

/// The encoded instructions of one function body.
///
/// Has one peephole: `local.set x; local.get x` becomes `local.tee x`. A
/// `local.set` is held back until the next instruction shows whether it can
/// be fused.
pub(crate) struct Body {
    bytes: Vec<u8>,
    pending_set: Option<u32>,
    use_tee: bool,
}

impl Body {
    pub fn new(use_tee: bool) -> Self {
        Body {
            bytes: Vec::new(),
            pending_set: None,
            use_tee,
        }
    }

    pub fn ins(&mut self, i: I<'_>) {
        if let Some(l) = self.pending_set.take() {
            if self.use_tee && matches!(i, I::LocalGet(g) if g == l) {
                I::LocalTee(l).encode(&mut self.bytes);
                return;
            }
            I::LocalSet(l).encode(&mut self.bytes);
        }
        if let I::LocalSet(l) = i {
            self.pending_set = Some(l);
            return;
        }
        i.encode(&mut self.bytes);
    }

    pub fn finish(mut self) -> Vec<u8> {
        if let Some(l) = self.pending_set.take() {
            I::LocalSet(l).encode(&mut self.bytes);
        }
        self.bytes
    }
}

/// Locals created during emission, after the ones assigned to values:
/// short-lived scratch locals used inside a single instruction's lowering,
/// and function-wide ones such as the frame pointer.
pub(crate) struct ExtraLocals {
    base: u32,
    types: Vec<ValType>,
    free_scratch: Vec<(ValType, u32)>,
}

impl ExtraLocals {
    /// `base` is the first local index not used by parameters or values.
    pub fn new(base: u32) -> Self {
        ExtraLocals {
            base,
            types: Vec::new(),
            free_scratch: Vec::new(),
        }
    }

    /// A new local, for the rest of the function.
    pub fn fresh(&mut self, ty: ValType) -> u32 {
        self.types.push(ty);
        self.base + self.types.len() as u32 - 1
    }

    /// A scratch local. It must be [released](Self::release) once its value
    /// is dead; nested lowerings get distinct scratch locals.
    pub fn scratch(&mut self, ty: ValType) -> u32 {
        match self.free_scratch.iter().position(|&(t, _)| t == ty) {
            Some(i) => self.free_scratch.swap_remove(i).1,
            None => self.fresh(ty),
        }
    }

    pub fn release(&mut self, ty: ValType, l: u32) {
        self.free_scratch.push((ty, l));
    }

    pub fn types(&self) -> &[ValType] {
        &self.types
    }
}
