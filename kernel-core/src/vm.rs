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
    /// Table pages given back (their mappings all gone), linked through
    /// their first word, and how many: taken before new ones.
    pub(crate) free: Cell<Paddr>,
    pub(crate) nfree: Cell<u64>,
    /// A device address space: accesses its devices were refused (by the
    /// IOMMU), and a capability to the notification told of each, given
    /// by whoever attached a device (`DeviceAttach`).
    pub(crate) faults: Cell<u64>,
    pub(crate) fault_note: crate::cap::Slot,
}
impl KObj for AsHeader {}
// The page's upper half is the architecture's (processors' identifiers, the
// IOMMU's domain).
const _: () = assert!(core::mem::size_of::<AsHeader>() <= 2048);

impl AsHeader {
    /// Where its fault notification's slot is, in the object at `a`.
    pub(crate) fn fault_note_slot(a: Paddr) -> Paddr {
        a + core::mem::offset_of!(AsHeader, fault_note) as u64
    }
}

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
    /// space (`io`: one for devices), and the bytes one entry there maps (a
    /// power of two). Inlined: out of line, the page fault paid for a
    /// division by the step its caller could no longer see.
    #[inline(always)]
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
        let (p, at) = self.walk_to(a, va, level);
        (self.phys.word(p), at)
    }

    /// [`Kernel::walk`], giving where the entry is.
    #[inline(always)]
    fn walk_to(&self, a: Paddr, va: u64, level: u8) -> (Paddr, u8) {
        let mut table = a + PAGE;
        let mut at = 3;
        loop {
            let p = table + ((va >> (PAGE_BITS + 9 * at)) & 511) * 8;
            let value = self.phys.word(p).get();
            if at == level || !H::pte_present(value) || H::pte_is_leaf(value, at) {
                return (p, at);
            }
            table = H::pte_addr(value);
            at -= 1;
        }
    }

    /// Check both overlap and table capacity without changing the address
    /// space. Each missing subtree is counted once, even for a frame that
    /// crosses several table boundaries. Inlined into `map`, its one
    /// caller: the call measured 23 instructions on the page fault's path.
    /// Returns the walk to `va` (where its entry is, and at what level the
    /// walk stopped), which nothing changes before `map` uses it: `map`
    /// does not walk to its first address twice. `va < end` (a frame is
    /// never empty).
    #[inline(always)]
    fn check_map(&self, a: Paddr, mut va: u64, end: u64, level: u8) -> Result<(Paddr, u8), Error> {
        let h = self.aspace(a);
        let mut available = (1 << (h.bits.get() - PAGE_BITS as u64)) - 2 - h.used.get() + h.nfree.get();
        let first = self.walk_to(a, va, level);
        let (mut p, mut at) = first;
        loop {
            if H::pte_present(self.phys.word(p).get()) {
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
            if va >= end {
                return Ok(first);
            }
            (p, at) = self.walk_to(a, va, level);
        }
    }

    /// Map the frame in slot `fs` at `vaddr` in address space `a`.
    /// Holds the structure lock unless the caller holds every lock or
    /// there is one processor (see `smp.rs`).
    pub(crate) fn map(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        // Mapping for a device is a copy of its own, reached by a tail call
        // before anything else is live: a page fault's cure pays one test
        // for it (measured: dispatching after taking the lock kept values
        // alive across the call, 20 instructions a fault).
        if self.aspace(a).io.get() != 0 {
            return self.map_io(a, fs, vaddr, rights);
        }
        self.map_with_lock::<false>(a, fs, vaddr, rights)
    }

    #[cold]
    #[inline(never)]
    fn map_io(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        self.map_with_lock::<true>(a, fs, vaddr, rights)
    }

    #[inline(always)]
    fn map_with_lock<const IO: bool>(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        let shared = self.ncpu() > 1 && !self.global.get();
        if shared {
            self.structure.lock();
        }
        let r = self.map_locked::<IO>(a, fs, vaddr, rights);
        if shared {
            self.structure.unlock();
        }
        r
    }

    #[inline(always)]
    fn map_locked<const IO: bool>(&self, a: Paddr, fs: Paddr, vaddr: u64, rights: Rights) -> Result<(), Error> {
        let slot = self.slot(fs);
        let cap = slot.cap().ok_or(Error::BadCap)?;
        if slot.badge.get() & MAPPED != 0 {
            return Err(Error::Mapped);
        }
        if !cap.rights.has(rights) || !rights.has(Rights::READ) || rights.has(Rights(Rights::WRITE.0 | Rights::EXEC.0)) {
            // W^X: no mapping is both writable and executable.
            return Err(if cap.rights.has(rights) { Error::BadArg } else { Error::NoRights });
        }
        let (level, step) = self.leaf(cap.bits, IO);
        let size = 1u64 << cap.bits;
        let top = if IO { self.hal.io_top() } else { H::USER_TOP };
        // Executing means nothing to a device. No mapping writes more
        // than MAP_MAX_ENTRIES entries (nor its unmapping clears more).
        if vaddr & (step - 1) != 0
            || vaddr >= top
            || size > top - vaddr
            || (IO && rights.has(Rights::EXEC))
            || size / step > silica_abi::MAP_MAX_ENTRIES
        {
            return Err(Error::BadArg);
        }
        let first = self.check_map(a, vaddr, vaddr + size, level)?;
        let device = cap.flags & FLAG_DEVICE != 0;
        // Taken only now: live across the check, it cost the page fault
        // spills (measured).
        let h = self.aspace(a);
        for i in 0..size / step {
            let va = vaddr + i * step;
            // The first entry's walk was made by the check: not again.
            let (p, at) = if i == 0 { first } else { self.walk_to(a, va, level) };
            let mut entry = self.phys.word(p);
            for child in (level..at).rev() {
                // Zero already: a new one was zeroed with the object (Retype),
                // one given back was empty but for its link (`take_table`).
                let page = self.take_table(a, h);
                entry.set(if IO { H::io_pte_table(page) } else { H::pte_table(page) });
                entry = self.phys.word(page + ((va >> (PAGE_BITS + 9 * child)) & 511) * 8);
            }
            let frame = cap.obj + i * step;
            entry.set(if IO { H::io_pte_leaf(frame, rights, level) } else { H::pte_leaf(frame, rights, level, device) });
        }
        if IO && h.attached.get() != 0 {
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
        let n = size / step;
        let mut last = 0;
        for i in 0..n {
            let (p, at) = self.walk_to(a, vaddr + i * step, level);
            debug_assert_eq!(at, level);
            self.phys.word(p).set(0);
            last = p;
        }
        // Tables left with nothing in them go back to the space's pool; the
        // flush below drops what any processor or device cached of them
        // (every one lies on the walk of the addresses flushed). When the
        // cleared entries lie in one table (consecutive, ending at `last`),
        // that table is looked at first, from beside them (a live entry
        // there, the usual case, is one read), and the tables above only
        // when it is empty: nothing above a live table can be.
        let in_table = last & (PAGE - 1);
        let still_used = if (n - 1) * 8 <= in_table {
            let table = last - in_table;
            let first = last - (n - 1) * 8;
            let live = |p: Paddr| self.phys.word(p).get() != 0;
            (in_table != PAGE - 8 && live(last + 8)) || (first != table && live(first - 8)) || !self.page_zero(table)
        } else {
            false
        };
        if !still_used {
            self.prune(a, vaddr, vaddr + size, level, io);
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

    /// The IOMMU refused accesses of the devices named by `sids` (source
    /// identifiers): each one's device address space counts it (`Inspect`)
    /// and its fault notification, if its attacher gave one, is signalled.
    /// Called by the architecture's fault interrupt holding this
    /// processor's lock; `space_of` finds a device's space as
    /// [`Hal::io_attach`] named it, looked up again holding every lock (it
    /// may have been destroyed since the fault was recorded).
    pub fn device_faults(&self, sids: &[u16], space_of: impl Fn(u16) -> Paddr) {
        if sids.is_empty() {
            return;
        }
        let global = self.ncpu() > 1;
        if global {
            self.go_global();
        }
        for &sid in sids {
            let root = space_of(sid);
            if root == 0 {
                continue;
            }
            let a = root - PAGE;
            let h = self.aspace(a);
            h.faults.set(h.faults.get() + 1);
            if let Some(n) = self.slot(AsHeader::fault_note_slot(a)).cap() {
                self.signal(n.obj, n.badge.max(1));
            }
        }
        if global {
            self.go_local();
        }
    }

    /// Remove every mapping of address space `a` that overlaps `start` to
    /// `end`, each whole, whoever holds its frame; returns how many. One
    /// pass over the space's mapped frames.
    pub(crate) fn unmap_range(&self, a: Paddr, start: u64, end: u64) -> u64 {
        let (mut s, mut n) = (self.aspace(a).maps.get(), 0);
        while s != 0 {
            let slot = self.slot(s);
            let next = slot.mnext.get();
            let va = slot.badge.get() & !(PAGE - 1);
            let bits = slot.cap().map_or(PAGE_BITS, |c| c.bits);
            if va < end && va.saturating_add(1 << bits) > start {
                self.unmap_slot(s);
                n += 1;
            }
            s = next;
        }
        n
    }

    /// A table page for address space `a`: one given back, else a new one
    /// from its pool (the caller has checked there is one, `check_map`).
    #[inline(always)]
    fn take_table(&self, a: Paddr, h: &AsHeader) -> Paddr {
        let f = h.free.get();
        if f != 0 {
            h.free.set(self.phys.word(f).get());
            self.phys.word(f).set(0);
            h.nfree.set(h.nfree.get() - 1);
            return f;
        }
        let used = h.used.get();
        h.used.set(used + 1);
        a + (used + 2) * PAGE
    }

    /// Entries of `level` in `[start, end)` of address space `a` were just
    /// cleared: each table on their walks below the root that holds nothing
    /// any more is unlinked from its parent and given back, lowest first so
    /// that its parent may be empty in turn. A device space's first table,
    /// under the root's entry 0 from its making, stays (`init_io_space`).
    #[cold]
    #[inline(never)]
    fn prune(&self, a: Paddr, start: u64, end: u64, level: u8, io: bool) {
        let h = self.aspace(a);
        // One table of leaves at a time: the walk to it gives every table
        // above it, and the first that still holds something ends the look
        // upward (nothing above a live table is empty).
        let span = 1u64 << (PAGE_BITS as u32 + 9 * (level as u32 + 1));
        let mut v = start;
        while v < end {
            let chunk_end = end.min((v & !(span - 1)).saturating_add(span));
            // The entries naming each table on the walk, root down: at
            // levels 3, 2 and 1 (only those above the leaves are used).
            let mut parents = [None; 4];
            let mut table = a + PAGE;
            for at in (level + 1..=3).rev() {
                let e = self.phys.word(table + ((v >> (PAGE_BITS as u32 + 9 * at as u32)) & 511) * 8);
                let ev = e.get();
                if !H::pte_present(ev) || H::pte_is_leaf(ev, at) {
                    break;
                }
                parents[at as usize] = Some(e);
                table = H::pte_addr(ev);
            }
            for l in level..3 {
                let Some(parent) = parents[l as usize + 1] else { break };
                let table = H::pte_addr(parent.get());
                // The entries of this table in the range: cleared at the
                // leaves' level, maybe not above it.
                let shift = PAGE_BITS as u32 + 9 * l as u32;
                let (lo, hi) = ((v >> shift) & 511, ((chunk_end - 1) >> shift) & 511);
                if (io && table == a + 2 * PAGE) || !self.table_empty(table, lo, hi, l == level) {
                    break;
                }
                parent.set(0);
                // The link is written into the freed table, which a
                // processor of this space running in user mode may still
                // walk until the flush that follows (a device until its
                // invalidation): it reads as an entry that is not there in
                // every format (a page's address: bit 0, present or valid,
                // clear), so it can only fault, as the empty entry would.
                let link = h.free.get();
                debug_assert!(!H::pte_present(link) && link & (PAGE - 1) == 0);
                self.phys.word(table).set(link);
                h.free.set(table);
                h.nfree.set(h.nfree.get() + 1);
            }
            v = chunk_end;
        }
    }

    /// Whether the page at `p` is all zero: eight words at a time, so a
    /// table that empties costs a pass of plain loads, once per table.
    #[cold]
    #[inline(never)]
    fn page_zero(&self, p: Paddr) -> bool {
        (0..PAGE / 64).all(|c| (0..8).fold(0, |acc, i| acc | self.phys.word(p + c * 64 + i * 8).get()) == 0)
    }

    /// Whether the table at `table` holds no entry, its entries `lo..=hi`
    /// being the ones just looked at (`cleared`: known to be 0). Looked at
    /// outward from those, alternately above and below: a table being
    /// unmapped in order, either way, has a live entry beside the cleared
    /// ones until it is empty, so this costs a read or two per unmapping
    /// and a whole table only when it really is empty.
    fn table_empty(&self, table: Paddr, lo: u64, hi: u64, cleared: bool) -> bool {
        let live = |i: u64| self.phys.word(table + i * 8).get() != 0;
        if !cleared && (lo..=hi).any(live) {
            return false;
        }
        for d in 1..512 {
            let (up, down) = (hi + d, lo.checked_sub(d));
            if up < 512 && live(up) {
                return false;
            }
            if let Some(i) = down
                && live(i)
            {
                return false;
            }
            if up >= 512 && down.is_none() {
                break;
            }
        }
        true
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
            // Whoever was told of its devices' faults is not any more.
            self.delete_slot_shallow(AsHeader::fault_note_slot(a));
            // Its devices reach nothing from here on: the memory of its
            // tables and of the frames that were mapped can be reused.
            if h.attached.replace(0) != 0 {
                self.hal.io_release(&self.phys, a + PAGE);
            }
            return;
        }
        self.hal.leave_aspace(a + PAGE);
        let me = self.me();
        if me.runs_on(a + PAGE) {
            me.forget_root();
            me.dirty.set(true);
        }
        self.evict(a + PAGE);
    }

    /// Address space `root` is destroyed and its memory may be reused as
    /// soon as this returns: no other processor may still run on it. Where a
    /// root maps the kernel too ([`Hal::KERNEL_IN_ROOT`]), one that has it
    /// loaded runs the kernel through it out of user mode as well (idle, or
    /// waiting for its lock, which this entry holds), so each such processor
    /// is asked to leave it, kicked (out of user mode, out of a halt, out of
    /// its watch: into waiting for its lock), and answers there; this waits
    /// for every answer. Elsewhere the shoot-down is enough.
    fn evict(&self, root: Paddr) {
        if !H::KERNEL_IN_ROOT || self.ncpu() == 1 {
            self.shoot_down(root);
            return;
        }
        for (i, cpu) in self.others() {
            if cpu.runs_on(root) {
                cpu.flush.set(true);
                cpu.dirty.set(true);
                cpu.leave.ask(root);
                self.hal.kick(i, false);
            } else {
                self.hal.tlb_forget(root, i);
            }
        }
        // All asked before any is waited for: they leave at once.
        for (i, cpu) in self.others() {
            if cpu.runs_on(root) {
                self.hal.wait_left(i, &cpu.leave);
                cpu.forget_root();
            }
        }
    }

    /// Mappings of address space `root` were removed: no other processor
    /// may go on using a cached translation. One that has it loaded is
    /// brought into the kernel and starts over with nothing cached; the
    /// others forget what they had.
    fn shoot_down(&self, root: Paddr) {
        if self.ncpu() == 1 {
            return;
        }
        for (i, cpu) in self.others() {
            if cpu.runs_on(root) {
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
