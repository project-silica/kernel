//! Address spaces: four-level page tables whose table pages come from the
//! AddressSpace object itself, and the list of frames mapped into each.
//!
//! An AddressSpace object of `2^bits` bytes is a header page, the root
//! table, and `2^(bits-12) - 2` pages that become page tables on demand.
//! When they run out, `Map` fails with `NoMemory`.

use crate::cap::{FLAG_DEVICE, MAPPED};
use crate::hal::Hal;
use crate::kernel::Kernel;
use crate::mem::{KObj, Paddr};
use core::cell::Cell;
use silica_abi::{Error, LARGE_BITS, PAGE_BITS, Rights};

/// Header page of an AddressSpace object.
#[repr(C, align(64))]
pub struct AsHeader {
    /// Size of the whole object in bits.
    pub(crate) bits: Cell<u64>,
    /// Table pages handed out so far.
    pub(crate) used: Cell<u64>,
    /// First slot in the list of frame capabilities mapped here.
    pub(crate) maps: Cell<Paddr>,
}
impl KObj for AsHeader {}

const PAGE: u64 = 1 << PAGE_BITS;

impl<H: Hal> Kernel<H> {
    pub(crate) fn aspace(&self, a: Paddr) -> &AsHeader {
        self.phys.obj(a)
    }

    /// The entry for `vaddr` at `level` (0: 4 KiB pages, 1: 2 MiB pages).
    /// With `alloc`, missing tables are created.
    fn pte(
        &self,
        a: Paddr,
        vaddr: u64,
        level: u8,
        alloc: bool,
    ) -> Result<Option<&Cell<u64>>, Error> {
        let h = self.aspace(a);
        let mut table = a + PAGE;
        for l in (level + 1..4).rev() {
            let e = self
                .phys
                .word(table + ((vaddr >> (12 + 9 * l as u64)) & 511) * 8);
            if H::pte_present(e.get()) {
                if H::pte_is_leaf(e.get(), l) {
                    return Err(Error::Overlap);
                }
                table = H::pte_addr(e.get());
            } else if alloc {
                let used = h.used.get();
                if used + 2 >= 1u64 << (h.bits.get() - PAGE_BITS as u64) {
                    return Err(Error::NoMemory);
                }
                h.used.set(used + 1);
                let page = a + (used + 2) * PAGE;
                self.phys.zero(page, PAGE);
                e.set(H::pte_table(page));
                table = page;
            } else {
                return Ok(None);
            }
        }
        Ok(Some(self.phys.word(
            table + ((vaddr >> (12 + 9 * level as u64)) & 511) * 8,
        )))
    }

    /// Map the frame in slot `fs` at `vaddr` in address space `a`.
    pub(crate) fn map(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        let slot = self.slot(fs);
        let cap = slot.cap().ok_or(Error::BadCap)?;
        if slot.badge.get() & MAPPED != 0 {
            return Err(Error::Mapped);
        }
        if !cap.rights.has(rights)
            || !rights.has(Rights::READ)
            || rights.has(Rights(Rights::WRITE.0 | Rights::EXEC.0))
        {
            // W^X: no mapping is both writable and executable.
            return Err(if cap.rights.has(rights) {
                Error::BadArg
            } else {
                Error::NoRights
            });
        }
        let (level, step) = if cap.bits >= LARGE_BITS {
            (1u8, 1u64 << LARGE_BITS)
        } else {
            (0u8, PAGE)
        };
        let size = 1u64 << cap.bits;
        if !vaddr.is_multiple_of(step) || vaddr >= H::USER_TOP || size > H::USER_TOP - vaddr {
            return Err(Error::BadArg);
        }
        let count = size / step;
        for i in 0..count {
            let e = self
                .pte(a, vaddr + i * step, level, true)?
                .ok_or(Error::NoMemory)?;
            if H::pte_present(e.get()) {
                return Err(Error::Overlap);
            }
        }
        let device = cap.flags & FLAG_DEVICE != 0;
        for i in 0..count {
            if let Ok(Some(e)) = self.pte(a, vaddr + i * step, level, false) {
                e.set(H::pte_leaf(cap.obj + i * step, rights, level, device));
            }
        }
        let h = self.aspace(a);
        slot.badge.set(vaddr | MAPPED);
        slot.aux.set(a);
        slot.mprev.set(0);
        slot.mnext.set(h.maps.get());
        if h.maps.get() != 0 {
            self.slot(h.maps.get()).mprev.set(fs);
        }
        h.maps.set(fs);
        Ok(())
    }

    /// Remove the mapping made through the frame capability in slot `fs`.
    pub(crate) fn unmap_slot(&self, fs: Paddr) {
        let slot = self.slot(fs);
        let Some(cap) = slot.cap() else { return };
        if slot.badge.get() & MAPPED == 0 {
            return;
        }
        let (a, vaddr) = (slot.aux.get(), slot.badge.get() & !(PAGE - 1));
        let (level, step) = if cap.bits >= LARGE_BITS {
            (1u8, 1u64 << LARGE_BITS)
        } else {
            (0u8, PAGE)
        };
        let size = 1u64 << cap.bits;
        for i in 0..size / step {
            if let Ok(Some(e)) = self.pte(a, vaddr + i * step, level, false) {
                e.set(0);
            }
        }
        self.hal.tlb_flush(a + PAGE, vaddr, size);
        let (p, n) = (slot.mprev.get(), slot.mnext.get());
        if p != 0 {
            self.slot(p).mnext.set(n);
        } else {
            self.aspace(a).maps.set(n);
        }
        if n != 0 {
            self.slot(n).mprev.set(p);
        }
        slot.badge.set(0);
        slot.aux.set(0);
        slot.mprev.set(0);
        slot.mnext.set(0);
    }

    /// The last capability to address space `a` is gone: its frames become unmapped.
    pub(crate) fn finalise_aspace(&self, a: Paddr) {
        let h = self.aspace(a);
        let mut s = h.maps.replace(0);
        while s != 0 {
            let slot = self.slot(s);
            let n = slot.mnext.get();
            slot.badge.set(0);
            slot.aux.set(0);
            slot.mprev.set(0);
            slot.mnext.set(0);
            s = n;
        }
        self.hal.leave_aspace(a + PAGE);
        if self.loaded_root.get() == a + PAGE {
            self.loaded_root.set(0);
        }
    }

    /// The physical address `vaddr` maps to in `a`, for tests and the loader of the root task.
    pub fn translate(&self, a: Paddr, vaddr: u64) -> Option<Paddr> {
        for level in [0u8, 1] {
            if let Ok(Some(e)) = self.pte(a, vaddr, level, false)
                && H::pte_present(e.get())
                && H::pte_is_leaf(e.get(), level)
            {
                let mask = (1u64 << (12 + 9 * level as u64)) - 1;
                return Some(H::pte_addr(e.get()) + (vaddr & mask));
            }
        }
        None
    }
}
