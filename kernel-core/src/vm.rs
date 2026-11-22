//! Address spaces: four-level page tables whose table pages come from the
//! AddressSpace object itself, and the list of frames mapped into each.
//!
//! An AddressSpace object of `2^bits` bytes is a header page, the root
//! table, and `2^(bits-12) - 2` pages that become page tables on demand.
//! When they run out, `Map` fails with `NoMemory`.
//!
//! An address space made for devices (`ASPACE_IO`) is the same object with
//! tables in the IOMMU's format: the frames mapped in it are what the
//! devices attached to it reach by DMA, at the addresses they are mapped
//! at. Its first table page is put under the root's entry 0 when it is
//! made, so an IOMMU that walks one level less can be pointed at it.

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
    /// Not 0: an address space for devices.
    pub(crate) io: Cell<u64>,
    /// Devices attached to it (a device address space).
    pub(crate) attached: Cell<u64>,
}
impl KObj for AsHeader {}

const PAGE: u64 = 1 << PAGE_BITS;

impl<H: Hal> Kernel<H> {
    pub(crate) fn aspace(&self, a: Paddr) -> &AsHeader {
        self.phys.obj(a)
    }

    /// Make the new AddressSpace object at `a` one for devices.
    pub(crate) fn init_io_space(&self, a: Paddr) {
        let h = self.aspace(a);
        h.io.set(1);
        h.used.set(1);
        self.phys.word(a + PAGE).set(H::io_pte_table(a + 2 * PAGE));
    }

    /// The level frames of `2^bits` bytes are mapped at in an address
    /// space (`io`: one for devices), and the bytes one entry there maps.
    fn leaf(&self, bits: u8, io: bool) -> (u8, u64) {
        if bits >= LARGE_BITS && (!io || self.hal.io_large()) { (1, 1 << LARGE_BITS) } else { (0, PAGE) }
    }

    /// The device with identity `id` reaches memory through the device
    /// address space `a` from now on (0: reaches nothing).
    pub(crate) fn attach(&self, id: u64, a: Paddr) -> Result<(), Error> {
        let old = self.hal.io_attach(&self.phys, id, if a == 0 { 0 } else { a + PAGE })?;
        if old != 0 {
            let h = self.aspace(old - PAGE);
            h.attached.set(h.attached.get() - 1);
        }
        if a != 0 {
            let h = self.aspace(a);
            h.attached.set(h.attached.get() + 1);
        }
        Ok(())
    }

    /// Walk to the requested level, stopping at a missing entry or a leaf.
    fn walk(&self, a: Paddr, va: u64, level: u8) -> (&Cell<u64>, u8) {
        let mut table = a + PAGE;
        let mut at = 3;
        loop {
            let entry = self.phys.word(table + ((va >> (PAGE_BITS + 9 * at)) & 511) * 8);
            let value = entry.get();
            if at == level || !H::pte_present(value) || H::pte_is_leaf(value, at) {
                return (entry, at);
            }
            table = H::pte_addr(value);
            at -= 1;
        }
    }

    /// Check both overlap and table capacity without changing the address
    /// space. Each missing subtree is counted once, even for a frame that
    /// crosses several table boundaries.
    fn check_map(&self, a: Paddr, mut va: u64, end: u64, level: u8) -> Result<(), Error> {
        let h = self.aspace(a);
        let mut available = (1 << (h.bits.get() - PAGE_BITS as u64)) - 2 - h.used.get();
        while va < end {
            let (entry, at) = self.walk(a, va, level);
            if H::pte_present(entry.get()) {
                return Err(Error::Overlap);
            }
            let span = 1 << (PAGE_BITS + 9 * at);
            let stop = end.min((va | (span - 1)) + 1);
            for parent in level + 1..=at {
                let shift = PAGE_BITS + 9 * parent;
                let tables = ((stop - 1) >> shift) - (va >> shift) + 1;
                available = available.checked_sub(tables).ok_or(Error::NoMemory)?;
            }
            va = stop;
        }
        Ok(())
    }

    /// Map the frame in slot `fs` at `vaddr` in address space `a`.
    /// Holds the structure lock unless the caller holds every lock or
    /// there is one processor (see `smp.rs`).
    pub(crate) fn map(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        let shared = self.ncpu.get() > 1 && !self.global.get();
        if shared {
            self.structure.lock();
        }
        let r = self.map_locked(a, fs, vaddr, rights);
        if shared {
            self.structure.unlock();
        }
        r
    }

    fn map_locked(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        let slot = self.slot(fs);
        let cap = slot.cap().ok_or(Error::BadCap)?;
        if slot.badge.get() & MAPPED != 0 {
            return Err(Error::Mapped);
        }
        if !cap.rights.has(rights) || !rights.has(Rights::READ) || rights.has(Rights(Rights::WRITE.0 | Rights::EXEC.0)) {
            // W^X: no mapping is both writable and executable.
            return Err(if cap.rights.has(rights) { Error::BadArg } else { Error::NoRights });
        }
        let h = self.aspace(a);
        let io = h.io.get() != 0;
        let (level, step) = self.leaf(cap.bits, io);
        let size = 1u64 << cap.bits;
        let top = if io { self.hal.io_top() } else { H::USER_TOP };
        // Executing means nothing to a device.
        if !vaddr.is_multiple_of(step) || vaddr >= top || size > top - vaddr || (io && rights.has(Rights::EXEC)) {
            return Err(Error::BadArg);
        }
        self.check_map(a, vaddr, vaddr + size, level)?;
        let device = cap.flags & FLAG_DEVICE != 0;
        for i in 0..size / step {
            let va = vaddr + i * step;
            let (mut entry, at) = self.walk(a, va, level);
            for child in (level..at).rev() {
                let used = h.used.get();
                let page = a + (used + 2) * PAGE;
                self.phys.zero(page, PAGE);
                h.used.set(used + 1);
                entry.set(if io { H::io_pte_table(page) } else { H::pte_table(page) });
                entry = self.phys.word(page + ((va >> (PAGE_BITS + 9 * child)) & 511) * 8);
            }
            let frame = cap.obj + i * step;
            entry.set(if io { H::io_pte_leaf(frame, rights, level) } else { H::pte_leaf(frame, rights, level, device) });
        }
        if io && h.attached.get() != 0 {
            self.hal.io_mapped(&self.phys, a + PAGE);
        }
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
        let h = self.aspace(a);
        let io = h.io.get() != 0;
        let (level, step) = self.leaf(cap.bits, io);
        let size = 1u64 << cap.bits;
        for i in 0..size / step {
            let (entry, at) = self.walk(a, vaddr + i * step, level);
            debug_assert_eq!(at, level);
            entry.set(0);
        }
        if io {
            // Nothing is cached of an address space no device uses.
            if h.attached.get() != 0 {
                self.hal.io_flush(&self.phys, a + PAGE);
            }
        } else {
            self.hal.tlb_flush(a + PAGE, vaddr, size);
            self.shoot_down(a + PAGE);
        }
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
        if h.io.get() != 0 {
            // Its devices reach nothing from here on: the memory of its
            // tables and of the frames that were mapped can be reused.
            if h.attached.replace(0) != 0 {
                self.hal.io_release(&self.phys, a + PAGE);
            }
            return;
        }
        self.hal.leave_aspace(a + PAGE);
        let me = self.me();
        if me.loaded_root.get() == a + PAGE {
            me.loaded_root.set(0);
            me.dirty.set(true);
        }
        self.shoot_down(a + PAGE);
    }

    /// Mappings of address space `root` were removed: no other processor
    /// may go on using a cached translation. One that has it loaded is
    /// brought into the kernel and starts over with nothing cached; the
    /// others forget what they had.
    fn shoot_down(&self, root: Paddr) {
        if self.ncpu.get() == 1 {
            return;
        }
        for (i, cpu) in self.others() {
            if cpu.loaded_root.get() == root {
                cpu.flush.set(true);
                cpu.dirty.set(true);
                self.hal.kick(i, true);
            } else {
                self.hal.tlb_forget(root, i);
            }
        }
    }

    /// How many devices are attached to the device address space `a`, for tests.
    pub fn attached_to(&self, a: Paddr) -> u64 {
        self.aspace(a).attached.get()
    }

    /// The physical address `vaddr` maps to in `a`, for tests and the loader of the root task.
    pub fn translate(&self, a: Paddr, vaddr: u64) -> Option<Paddr> {
        let (entry, level) = self.walk(a, vaddr, 0);
        let value = entry.get();
        if H::pte_present(value) && H::pte_is_leaf(value, level) {
            let mask = (1u64 << (PAGE_BITS + 9 * level)) - 1;
            Some(H::pte_addr(value) + (vaddr & mask))
        } else {
            None
        }
    }
}
