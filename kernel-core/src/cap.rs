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
use core::cell::Cell;
use silica_abi::{Error, ObjectType, Rights, SLOT_BITS};

/// Capability flag: device memory (not zeroed, mapped uncached).
pub const FLAG_DEVICE: u8 = 1;
/// Hw capability flag: a range of I/O ports.
pub const FLAG_PORTS: u8 = 2;
/// Hw capability flag: a range of interrupt lines.
pub const FLAG_IRQS: u8 = 4;

/// One capability slot.
///
/// Field use by type: `badge` is the endpoint or notification badge, the
/// free offset of an Untyped, the end of a Hw range, or the mapped address
/// (plus [`MAPPED`] and rights) of a Frame; `aux` is the cleared offset of an
/// Untyped or the address space a Frame is mapped in.
#[repr(C, align(64))]
pub struct Slot {
    pub(crate) obj: Cell<u64>,
    /// Bits 0..8 type (0: empty), 8..16 rights, 16..24 size bits, 24..32 flags, 32..48 depth.
    pub(crate) meta: Cell<u64>,
    pub(crate) badge: Cell<u64>,
    pub(crate) aux: Cell<u64>,
    pub(crate) prev: Cell<Paddr>,
    pub(crate) next: Cell<Paddr>,
    /// Links in the list of frames mapped into one address space.
    pub(crate) mprev: Cell<Paddr>,
    pub(crate) mnext: Cell<Paddr>,
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

    /// The address of slot `index` of the CapSpace `cs` refers to.
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
        if cap.ty == ObjectType::Frame {
            self.unmap_slot(s);
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
        if !last {
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
            }
            ObjectType::Thread => {
                self.finalise_thread(cap.obj);
                self.thread(cap.obj).reap_next.set(self.reap.get());
                self.reap.set(cap.obj | 1);
            }
            ObjectType::AddressSpace => self.finalise_aspace(cap.obj),
            ObjectType::Endpoint | ObjectType::Notification => self.finalise_queue(cap.obj),
            _ => {}
        }
    }

    fn drain_reap(&self) {
        loop {
            let e = self.reap.get();
            if e == 0 {
                return;
            }
            if e & 1 == 1 {
                let t = e & !1;
                self.reap.set(self.thread(t).reap_next.get());
                for s in crate::thread::Thread::embedded_slots(t) {
                    self.delete_shallow(s);
                }
            } else {
                let s0 = self.slot(e);
                self.reap.set(s0.next.get());
                let bits = s0.badge.get();
                s0.wipe();
                for i in 1..1u64 << bits {
                    self.delete_shallow(e + (i << SLOT_BITS));
                }
            }
        }
    }

    /// Delete the capability in `s` and everything derived from it.
    fn delete_shallow(&self, s: Paddr) {
        while self.has_children(s) {
            self.clear_slot(self.slot(s).next.get());
        }
        self.clear_slot(s);
    }

    /// Delete the capability in slot `s` (no effect on an empty slot).
    pub(crate) fn delete(&self, s: Paddr) {
        self.delete_shallow(s);
        self.drain_reap();
    }

    /// Delete every capability derived from the one in `s`. Returns `false`
    /// when an interrupt is pending and work remains; the caller restarts.
    pub(crate) fn revoke(&self, s: Paddr) -> bool {
        while self.has_children(s) {
            self.clear_slot(self.slot(s).next.get());
            self.drain_reap();
            if self.hal.irq_pending() && self.has_children(s) {
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
        s.wipe();
    }
}
