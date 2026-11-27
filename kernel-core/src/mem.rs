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
///
/// In the kernel the base is the constant [`crate::boot::PHYS_MAP_BASE`],
/// where every loader maps all of memory: then an object's address is its
/// physical address plus an immediate. Held in the kernel's static, the
/// base was loaded again after every store (any `Cell` might have changed
/// it, for all the compiler could tell): 15 times in a context switch.
/// Tests, the exhaustive check and the proofs run on a host and give a
/// buffer of their own.
pub struct Phys {
    #[cfg(not(target_os = "none"))]
    base: *mut u8,
}

impl Phys {
    /// The kernel's window: all of memory at [`crate::boot::PHYS_MAP_BASE`].
    ///
    /// # Safety
    /// For every physical address `p` the kernel is given in a boot region,
    /// `PHYS_MAP_BASE + p` must be valid for reads and writes for as long
    /// as this value lives, and nothing else may use that memory.
    #[cfg(target_os = "none")]
    pub const unsafe fn direct_map() -> Phys {
        Phys {}
    }

    /// # Safety
    /// For every physical address `p` the kernel is given in a boot region,
    /// `base + p` must be valid for reads and writes for as long as this
    /// value lives, and nothing else may use that memory.
    #[cfg(not(target_os = "none"))]
    pub const unsafe fn new(base: *mut u8) -> Phys {
        Phys { base }
    }

    /// The address of physical byte `p`.
    #[cfg(target_os = "none")]
    #[inline(always)]
    pub fn ptr(&self, p: Paddr) -> *mut u8 {
        (crate::boot::PHYS_MAP_BASE as usize).wrapping_add(p as usize) as *mut u8
    }

    /// The address of physical byte `p`.
    #[cfg(not(target_os = "none"))]
    #[inline(always)]
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

    /// The 32-bit word at `p` of a frame a device writes (`IrqWatch`),
    /// read once as the device left it.
    pub(crate) fn device_word(&self, p: Paddr) -> u32 {
        debug_assert!(p != 0 && p.is_multiple_of(4));
        // SAFETY: `p` is in a live frame (module invariant: a watch ends when
        // its frame is destroyed), aligned; a device may write it at any
        // time, so it is read with one volatile load and the value is
        // only a fact about that moment.
        unsafe { self.ptr(p).cast::<u32>().read_volatile() }
    }

    /// Set the bits `set` and clear the bits `clear` of the 32-bit word at
    /// `p` of a frame a device reads (`IrqQuiet`), in one atomic step each:
    /// the program sharing the word with the device may store to its other
    /// bits at the same time, and keeps what it stored.
    pub(crate) fn device_word_update(&self, p: Paddr, set: u32, clear: u32) {
        debug_assert!(p != 0 && p.is_multiple_of(4));
        // SAFETY: `p` is in a live frame of RAM (module invariant: a watch
        // ends when its frames are destroyed), aligned for a u32; others
        // write it only with atomic or plain stores the hardware keeps
        // apart from this read-modify-write, which is what an atomic is.
        let w = unsafe { &*self.ptr(p).cast::<core::sync::atomic::AtomicU32>() };
        if set != 0 {
            w.fetch_or(set, core::sync::atomic::Ordering::SeqCst);
        }
        if clear != 0 {
            w.fetch_and(!clear, core::sync::atomic::Ordering::SeqCst);
        }
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
