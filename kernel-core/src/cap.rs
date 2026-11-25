//! Capability slots, the derivation tree and object destruction.
//!
//! Every capability lives in a 64-byte [`Slot`], either in a CapSpace or
//! inside a Thread object. Slots that derive from one another form a doubly
//! linked list in which a capability's descendants follow it directly and
//! carry a greater depth, so "everything derived from this" is the run of
//! following slots with greater depth. All capabilities to one object are
//! adjacent in that list, which makes "is this the last capability to the
//! object" a look at the two neighbours.
//!
//! A capability never outlives what it was derived from: deleting one
//! deletes its descendants first. That keeps every step O(1) and the tree
//! exact (no descendant is ever re-parented). A holder who wants a second
//! capability that does not depend on the first makes a duplicate, which is
//! a sibling in the tree, or moves the capability.

use crate::hal::Hal;
use crate::kernel::Kernel;
use crate::mem::{KObj, Paddr};
use crate::smp::Word;
use silica_abi::{Error, ObjectType, Rights, SLOT_BITS};

/// Slots a CapSpace being emptied gives up between looks for a pending
/// interrupt (`drain_reap`).
const REAP_STEP: u64 = 256;

/// Capability flag: device memory, mapped with device attributes
/// (uncached; on Arm also no unaligned access). Always also [`FLAG_KEPT`].
pub const FLAG_DEVICE: u8 = 1;
/// Hw capability flag: a range of I/O ports.
pub const FLAG_PORTS: u8 = 2;
/// Hw capability flag: a range of interrupt lines.
pub const FLAG_IRQS: u8 = 4;
/// Hw capability flag: the kernel's console.
pub const FLAG_CONSOLE: u8 = 8;
/// Hw capability flag: turning the machine off.
pub const FLAG_POWER: u8 = 16;
/// Untyped and Frame flag: memory whose contents are kept (device memory,
/// firmware tables, the ramdisk). It is never zeroed, so it becomes only
/// frames and smaller Untyped, never kernel objects.
pub const FLAG_KEPT: u8 = 32;
/// AddressSpace flag: isolated against speculative execution
/// (`ASPACE_ISOLATED`): switching into or out of it flushes the branch
/// predictors (`Hal::isolate`). On the capability, which the scheduler
/// reads anyway on a switch, so other address spaces pay nothing to look.
/// (Bit 1 means a port range on a Hw capability; an AddressSpace is never one.)
pub const FLAG_ISOLATED: u8 = 2;
/// AddressSpace flag: an address space for devices (`ASPACE_IO`). Its
/// tables are the IOMMU's, and no thread runs in it.
pub const FLAG_IO: u8 = 64;
/// Hw capability flag: a range of processors, by index: the authority to
/// give threads more than the default share of them (`SchedConfigure`).
/// (Bit 0 means device memory on a Frame or Untyped; a Hw capability is
/// never memory.)
pub const FLAG_SCHED: u8 = 1;
/// Hw capability flag: a range of device identities (PCI segment, bus,
/// device, function), the authority to say what those devices reach by DMA
/// and which message interrupts they raise.
pub const FLAG_DMA: u8 = 128;

/// `meta` bits above the depth, for `CapNotice`: the capability is watched
/// (`mnext` is the slot of its notice), or the slot is a notice (`mprev` is
/// the slot it watches). Never on a Frame, whose `mprev`/`mnext` link its
/// mappings. A notice is never copied or watched itself, so ending a watch
/// deletes a leaf and never recurses further.
pub(crate) const META_WATCHED: u64 = 1 << 48;
pub(crate) const META_NOTICE: u64 = 1 << 49;

/// One capability slot.
///
/// Field use by type: `badge` is the endpoint or notification badge, the
/// free offset of an Untyped, the end of a Hw range, or the mapped address
/// (plus `MAPPED` and rights) of a Frame; `aux` is the cleared offset of an
/// Untyped or the address space a Frame is mapped in.
#[repr(C, align(64))]
pub struct Slot {
    pub(crate) obj: Word,
    /// Bits 0..8 type (0: empty), 8..16 rights, 16..24 size bits, 24..32 flags, 32..48 depth.
    pub(crate) meta: Word,
    pub(crate) badge: Word,
    pub(crate) aux: Word,
    pub(crate) prev: Word,
    pub(crate) next: Word,
    /// Links in the list of frames mapped into one address space.
    pub(crate) mprev: Word,
    pub(crate) mnext: Word,
}
impl KObj for Slot {}

/// Low bit of a Frame slot's `badge`: the frame is mapped.
pub(crate) const MAPPED: u64 = 1;

/// A decoded capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cap {
    /// What it refers to.
    pub ty: ObjectType,
    /// What it permits.
    pub rights: Rights,
    /// Size of the object in bits; for a CapSpace, the number of slots in bits.
    pub bits: u8,
    /// `FLAG_` bits.
    pub flags: u8,
    /// Address of the object (start of the range for Hw).
    pub obj: Paddr,
    /// Type-specific word, see [`Slot`].
    pub badge: u64,
}

impl Slot {
    /// The capability in this slot.
    pub fn cap(&self) -> Option<Cap> {
        let m = self.meta.get();
        Some(Cap {
            ty: ObjectType::from_u64(m & 0xFF)?,
            rights: Rights((m >> 8) as u8),
            bits: (m >> 16) as u8,
            flags: (m >> 24) as u8,
            obj: self.obj.get(),
            badge: self.badge.get(),
        })
    }
    /// Whether the slot holds no capability.
    pub fn is_empty(&self) -> bool {
        self.meta.get() & 0xFF == 0
    }
    pub(crate) fn depth(&self) -> u16 {
        (self.meta.get() >> 32) as u16
    }
    pub(crate) fn set(&self, cap: &Cap, depth: u16) {
        self.obj.set(cap.obj);
        self.meta
            .set(cap.ty as u64 | (cap.rights.0 as u64) << 8 | (cap.bits as u64) << 16 | (cap.flags as u64) << 24 | (depth as u64) << 32);
        self.badge.set(cap.badge);
        self.aux.set(0);
    }
    fn wipe(&self) {
        for c in [&self.obj, &self.meta, &self.badge, &self.aux, &self.prev, &self.next, &self.mprev, &self.mnext] {
            c.set(0);
        }
    }
}

impl<H: Hal> Kernel<H> {
    pub(crate) fn slot(&self, p: Paddr) -> &Slot {
        self.phys.obj(p)
    }

    /// The address of slot `index` of the CapSpace `cs` refers to. Inlined:
    /// a few instructions, on every capability operation (out of line it
    /// cost the page fault 16).
    #[inline(always)]
    pub(crate) fn slot_in(&self, cs: &Cap, index: u64) -> Result<Paddr, Error> {
        if cs.ty != ObjectType::CapSpace {
            return Err(Error::WrongType);
        }
        if index == 0 || index >= 1u64 << cs.bits {
            return Err(Error::BadCap);
        }
        Ok(cs.obj + (index << SLOT_BITS))
    }

    /// Whether any capability derives from the one in slot `s`.
    pub(crate) fn has_children(&self, s: Paddr) -> bool {
        let n = self.slot(s).next.get();
        n != 0 && self.slot(n).depth() > self.slot(s).depth()
    }

    /// Put `cap` into the empty slot `child` as a child of the capability in `parent`.
    pub(crate) fn insert_child(&self, parent: Paddr, child: Paddr, cap: &Cap) -> Result<(), Error> {
        let (p, c) = (self.slot(parent), self.slot(child));
        let depth = p.depth().checked_add(1).ok_or(Error::TooDeep)?;
        c.set(cap, depth);
        c.prev.set(parent);
        c.next.set(p.next.get());
        if p.next.get() != 0 {
            self.slot(p.next.get()).prev.set(child);
        }
        p.next.set(child);
        Ok(())
    }

    /// Put `cap` into the empty slot `new` as a sibling of the capability in
    /// `of`: same parent, same depth, placed just before it.
    pub(crate) fn insert_sibling(&self, of: Paddr, new: Paddr, cap: &Cap) {
        let (o, n) = (self.slot(of), self.slot(new));
        n.set(cap, o.depth());
        n.prev.set(o.prev.get());
        n.next.set(of);
        if o.prev.get() != 0 {
            self.slot(o.prev.get()).next.set(new);
        }
        o.prev.set(new);
    }

    /// Put `cap` into the empty slot `s` with no ancestor (boot only).
    pub(crate) fn insert_root(&self, s: Paddr, cap: &Cap) {
        self.slot(s).set(cap, 0);
    }

    fn same_obj(&self, s: Paddr, cap: &Cap) -> bool {
        s != 0 && self.slot(s).cap().is_some_and(|c| c.ty == cap.ty && c.obj == cap.obj && c.ty != ObjectType::Hw)
    }

    /// Empty slot `s`. Destroys the object when this was its last
    /// capability; objects that hold slots of their own are queued on the
    /// reap list instead of being emptied here, so this never recurses.
    fn clear_slot(&self, s: Paddr) {
        let slot = self.slot(s);
        let Some(cap) = slot.cap() else { return };
        if slot.meta.get() & (META_WATCHED | META_NOTICE) != 0 {
            self.end_watch(s);
        }
        if cap.ty == ObjectType::Frame {
            self.unmap_slot(s);
        }
        if cap.ty == ObjectType::AddressSpace && cap.flags & FLAG_IO == 0 {
            // It may be a running thread's own, on any processor: each
            // looks again at what it should have loaded.
            self.me().dirty.set(true);
            if self.ncpu() > 1 {
                for (i, cpu) in self.others() {
                    cpu.dirty.set(true);
                    self.hal.kick(i, true);
                }
            }
        }
        let last = !self.same_obj(slot.prev.get(), &cap) && !self.same_obj(slot.next.get(), &cap);
        let (p, n) = (slot.prev.get(), slot.next.get());
        if p != 0 {
            self.slot(p).next.set(n);
        }
        if n != 0 {
            self.slot(n).prev.set(p);
        }
        slot.wipe();
        if cap.ty == ObjectType::Notification {
            // Whoever listens through this capability (through any, when
            // the notification itself goes) stops waiting.
            self.unlisten(cap.obj, if last { 0 } else { s });
        }
        if !last {
            if matches!(cap.ty, ObjectType::Endpoint | ObjectType::Notification) {
                self.revoke_waiters(s, cap.obj);
            }
            return;
        }
        match cap.ty {
            ObjectType::CapSpace => {
                // Slot 0 of a CapSpace is never used for a capability: it
                // carries the reap link and the size while the space is emptied.
                let s0 = self.slot(cap.obj);
                s0.next.set(self.reap.get());
                s0.badge.set(cap.bits as u64);
                self.reap.set(cap.obj);
                // A watch names its frames by slots of a CapSpace (`watch.rs`).
                if self.nwatches.get() != 0 {
                    self.unwatch_space(cap.obj, cap.obj + ((1u64 << cap.bits) << silica_abi::SLOT_BITS));
                }
            }
            ObjectType::Thread => {
                self.finalise_thread(cap.obj);
                self.thread(cap.obj).reap_next.set(self.reap.get());
                self.reap.set(cap.obj | 1);
            }
            ObjectType::AddressSpace => self.finalise_aspace(cap.obj),
            ObjectType::Endpoint | ObjectType::Notification => self.finalise_queue(cap.obj),
            ObjectType::SchedContext => self.finalise_sched(cap.obj),
            _ => {}
        }
    }

    /// `CapNotice`: watch the capability in slot `watched`, signalling the
    /// notification in slot `n` with `bits` when it is deleted; the notice,
    /// a capability derived from `n` whose badge is `bits`, goes in the
    /// empty slot `notice`.
    pub(crate) fn notice(&self, watched: Paddr, n: Paddr, notice: Paddr, bits: u64) -> Result<(), Error> {
        let w = self.slot(watched);
        let cap = w.cap().ok_or(Error::BadCap)?;
        if cap.ty == ObjectType::Frame {
            return Err(Error::WrongType);
        }
        if bits == 0 || watched == notice || n == notice {
            return Err(Error::BadArg);
        }
        if w.meta.get() & (META_WATCHED | META_NOTICE) != 0 {
            return Err(Error::InUse);
        }
        // A real capability: it goes when the notification does, which
        // ends the watch (`end_watch`) and never leaves it pointing nowhere.
        self.copy(n, notice, Rights::WRITE, bits, false)?;
        let ns = self.slot(notice);
        ns.meta.set(ns.meta.get() | META_NOTICE);
        ns.mprev.set(watched);
        w.meta.set(w.meta.get() | META_WATCHED);
        w.mnext.set(notice);
        Ok(())
    }

    /// Slot `s`, about to be emptied, is watched or a notice. A notice ends
    /// its watch; a watched capability's notification is signalled and its
    /// notice deleted (a leaf: never copied, so without children, and never
    /// watched, so this goes no deeper).
    fn end_watch(&self, s: Paddr) {
        let slot = self.slot(s);
        let m = slot.meta.get();
        slot.meta.set(m & !(META_WATCHED | META_NOTICE));
        if m & META_NOTICE != 0 {
            let w = self.slot(slot.mprev.get());
            w.meta.set(w.meta.get() & !META_WATCHED);
            w.mnext.set(0);
            slot.mprev.set(0);
        }
        if m & META_WATCHED != 0 {
            let at = slot.mnext.get();
            slot.mnext.set(0);
            let ns = self.slot(at);
            ns.meta.set(ns.meta.get() & !META_NOTICE);
            ns.mprev.set(0);
            if let Some(nc) = ns.cap() {
                self.signal(nc.obj, nc.badge);
            }
            self.clear_slot(at);
        }
    }

    /// Empty what is waiting to be emptied. A CapSpace has up to 2^32
    /// slots, and this runs holding every lock: it stops when an interrupt
    /// is pending, every [`REAP_STEP`] slots, and goes on where it stopped
    /// the next time (the CapSpace being emptied and the next slot are kept
    /// in `reaping`, its slot 0). Returns whether nothing is left. What is
    /// left is unreachable (no capability names it) but its memory is not
    /// free: every system call that could reuse memory takes every lock and
    /// finishes this first ([`Kernel::reap_pending`]).
    pub(crate) fn drain_reap(&self) -> bool {
        loop {
            let e = self.reaping.get();
            if e != 0 {
                // A CapSpace part emptied: its size in slot 0's badge, the
                // next slot in its aux.
                let s0 = self.slot(e);
                let n = 1u64 << s0.badge.get();
                let mut i = s0.aux.get().max(1);
                while i < n {
                    self.delete_shallow(e + (i << SLOT_BITS));
                    i += 1;
                    if i.is_multiple_of(REAP_STEP) && i < n && self.hal.irq_pending() {
                        s0.aux.set(i);
                        return false;
                    }
                }
                s0.wipe();
                self.reaping.set(0);
                continue;
            }
            let e = self.reap.get();
            if e == 0 {
                return true;
            }
            if e & 1 == 1 {
                let t = e & !1;
                self.reap.set(self.thread(t).reap_next.get());
                for s in crate::thread::Thread::embedded_slots(t) {
                    self.delete_shallow(s);
                }
            } else {
                // Off the list before it is emptied: what it holds may put
                // more on the list.
                let s0 = self.slot(e);
                self.reap.set(s0.next.get());
                s0.next.set(0);
                s0.aux.set(1);
                self.reaping.set(e);
            }
        }
    }

    /// Whether emptying is left over from an earlier entry: the caller
    /// (holding every lock) finishes it before anything that could reuse
    /// memory, and is made again when it cannot yet.
    #[inline(always)]
    pub(crate) fn reap_pending(&self) -> bool {
        self.reaping.get() != 0 || self.reap.get() != 0
    }

    /// Delete the capability in `s` and everything derived from it.
    fn delete_shallow(&self, s: Paddr) {
        while self.has_children(s) {
            self.clear_slot(self.slot(s).next.get());
        }
        self.clear_slot(s);
    }

    /// Delete the capability in slot `s` (no effect on an empty slot).
    /// What that leaves to empty (a large CapSpace) may be left for later
    /// ([`Kernel::drain_reap`]).
    pub(crate) fn delete(&self, s: Paddr) {
        self.delete_shallow(s);
        self.drain_reap();
    }

    /// Delete every capability derived from the one in `s`. Returns `false`
    /// when an interrupt is pending and work remains; the caller restarts.
    pub(crate) fn revoke(&self, s: Paddr) -> bool {
        while self.has_children(s) {
            self.clear_slot(self.slot(s).next.get());
            if !self.drain_reap() || (self.hal.irq_pending() && self.has_children(s)) {
                return false;
            }
        }
        let slot = self.slot(s);
        if slot.cap().is_some_and(|c| c.ty == ObjectType::Untyped) {
            slot.badge.set(0);
            slot.aux.set(0);
        }
        true
    }

    /// Move the capability in `src` to the empty slot `dst`, keeping its
    /// place in the derivation tree and its mapping.
    pub(crate) fn move_slot(&self, src: Paddr, dst: Paddr) {
        let (s, d) = (self.slot(src), self.slot(dst));
        for (a, b) in [(&s.obj, &d.obj), (&s.meta, &d.meta), (&s.badge, &d.badge), (&s.aux, &d.aux)] {
            b.set(a.get());
        }
        let (p, n, mp, mn) = (s.prev.get(), s.next.get(), s.mprev.get(), s.mnext.get());
        d.prev.set(p);
        d.next.set(n);
        d.mprev.set(mp);
        d.mnext.set(mn);
        if p != 0 {
            self.slot(p).next.set(dst);
        }
        if n != 0 {
            self.slot(n).prev.set(dst);
        }
        // A watch follows its slots (`CapNotice`): each names the other.
        let m = s.meta.get();
        if m & META_WATCHED != 0 {
            self.slot(mn).mprev.set(dst);
        }
        if m & META_NOTICE != 0 {
            self.slot(mp).mnext.set(dst);
        }
        if s.cap().is_some_and(|c| c.ty == ObjectType::Frame) && s.badge.get() & MAPPED != 0 {
            if mp != 0 {
                self.slot(mp).mnext.set(dst);
            } else {
                self.aspace(s.aux.get()).maps.set(dst);
            }
            if mn != 0 {
                self.slot(mn).mprev.set(dst);
            }
        }
        // Whoever waits through the capability waits through this slot,
        // which no longer holds it.
        if let Some(c) = s.cap().filter(|c| matches!(c.ty, ObjectType::Endpoint | ObjectType::Notification)) {
            if c.ty == ObjectType::Notification {
                self.unlisten(c.obj, src);
            }
            self.revoke_waiters(src, c.obj);
        }
        s.wipe();
    }
}
