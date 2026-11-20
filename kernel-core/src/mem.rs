//! Access to physical memory. This is the one module of `kernel-core` that
//! contains `unsafe`: it turns a physical address taken from a capability
//! into a reference to the kernel object stored there.
//!
//! The invariant every caller relies on: an address reaches these functions
//! only from a live capability or from a link inside a live object, and an
//! object's memory is reused (by `Retype`) only after every capability to it
//! and every link to it is gone. `cap.rs` maintains that; the reference-model
//! tests and Miri check it.
#![allow(unsafe_code)]

use core::cell::Cell;

/// A physical address.
pub type Paddr = u64;

/// The window through which the kernel sees physical memory: physical
/// address `p` is at `base + p`.
pub struct Phys {
    base: *mut u8,
}

impl Phys {
    /// # Safety
    /// For every physical address `p` the kernel is given in a boot region,
    /// `base + p` must be valid for reads and writes for as long as this
    /// value lives, and nothing else may use that memory.
    pub const unsafe fn new(base: *mut u8) -> Phys {
        Phys { base }
    }

    /// The address of physical byte `p`.
    pub fn ptr(&self, p: Paddr) -> *mut u8 {
        self.base.wrapping_add(p as usize)
    }

    /// The kernel object at `p`. Objects consist only of `Cell`s of plain
    /// integers, so any bit pattern is a valid object.
    pub(crate) fn obj<T: KObj>(&self, p: Paddr) -> &T {
        debug_assert!(p != 0 && p.is_multiple_of(core::mem::align_of::<T>() as u64));
        // SAFETY: by the module invariant `p` is the address of live memory
        // owned by the kernel, aligned for `T`; `T` is made of `Cell<u64>`
        // (valid for any bits) and is only ever accessed through shared
        // references like this one.
        unsafe { &*self.ptr(p).cast::<T>() }
    }

    /// One 64-bit word of a page table or a user frame.
    pub(crate) fn word(&self, p: Paddr) -> &Cell<u64> {
        debug_assert!(p != 0 && p.is_multiple_of(8));
        // SAFETY: as for `obj`; a `Cell<u64>` is valid for any bits.
        unsafe { &*self.ptr(p).cast::<Cell<u64>>() }
    }

    /// Fill `[p, p + len)` with zeros.
    pub fn zero(&self, p: Paddr, len: u64) {
        // SAFETY: the range is owned by the kernel (module invariant) and no
        // reference to an object inside it exists while it is being retyped.
        unsafe { core::ptr::write_bytes(self.ptr(p), 0, len as usize) }
    }

    /// Copy `data` to `p`.
    pub fn write_bytes(&self, p: Paddr, data: &[u8]) {
        // SAFETY: as for `zero`; `data` is ordinary kernel memory and cannot
        // overlap memory the kernel hands out.
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr(p), data.len()) }
    }

    /// Store a plain value at `p`.
    pub fn write_val<T: Copy>(&self, p: Paddr, v: &T) {
        debug_assert!(p.is_multiple_of(core::mem::align_of::<T>() as u64));
        // SAFETY: as for `write_bytes`.
        unsafe { core::ptr::copy_nonoverlapping(core::ptr::from_ref(v), self.ptr(p).cast::<T>(), 1) }
    }

    /// The bytes `[p, p + len)` of memory the boot loader filled and nothing
    /// modifies (the ramdisk, the command line, firmware tables).
    ///
    /// # Safety
    /// The range must be inside a boot region that is not handed out as
    /// writable memory while the returned slice is in use.
    pub unsafe fn bytes(&self, p: Paddr, len: u64) -> &[u8] {
        // SAFETY: guaranteed by the caller.
        unsafe { core::slice::from_raw_parts(self.ptr(p), len as usize) }
    }
}

/// Marker for types that may be placed in retyped memory: `repr(C)`, made of
/// `Cell<u64>` only, so all-zero memory and any other bit pattern are valid.
pub(crate) trait KObj {}
