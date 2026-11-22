//! Turning Untyped memory into objects.
//!
//! An Untyped is a power-of-two range with a watermark: objects are carved
//! upward from it and the range is reused only after every object made from
//! it is gone (`CapRevoke` on the Untyped). Memory is zeroed before it
//! becomes an object; the zeroing is done in pieces and remembered, so an
//! interrupt can preempt a large `Retype` without repeating finished work.

use crate::cap::{Cap, FLAG_IO, FLAG_KEPT};
use crate::hal::Hal;
use crate::kernel::{Exit, Kernel};
use crate::mem::Paddr;
use crate::vm::AsHeader;
use silica_abi::{ASPACE_IO, ASPACE_MIN_BITS, Error, ObjectType, PAGE_BITS, RETYPE_MAX, Rights};

/// Bytes zeroed between checks for a pending interrupt.
const ZERO_CHUNK: u64 = 1 << 18;

impl<H: Hal> Kernel<H> {
    /// Create `count` objects of type `ty` from the Untyped in slot `us`,
    /// placing their capabilities in `dst` slots `index..index + count`.
    pub(crate) fn retype(&self, us: Paddr, ty: ObjectType, bits: u64, dst: &Cap, index: u64, count: u64) -> Result<Exit, Error> {
        let uslot = self.slot(us);
        let ut = uslot.cap().ok_or(Error::BadCap)?;
        if ut.ty != ObjectType::Untyped {
            return Err(Error::WrongType);
        }
        let kept = ut.flags & FLAG_KEPT != 0;
        let obits = ty.storage_bits(bits)? as u64;
        let cbits = if ty == ObjectType::CapSpace { bits } else { obits };
        if kept && !matches!(ty, ObjectType::Untyped | ObjectType::Frame) {
            return Err(Error::BadArg);
        }
        if count == 0 || count > RETYPE_MAX {
            return Err(Error::BadArg);
        }
        // An address space for devices: one more page than the smallest,
        // for the table it starts with, and only where an IOMMU exists.
        let io = ty == ObjectType::AddressSpace && bits & ASPACE_IO != 0;
        if io && self.hal.io_top() == 0 {
            return Err(Error::Unsupported);
        }
        if io && obits == ASPACE_MIN_BITS as u64 {
            return Err(Error::BadArg);
        }
        // A child Untyped is strictly smaller than its parent, which bounds
        // how deeply Untypeds nest.
        if obits > ut.bits as u64 || (ty == ObjectType::Untyped && obits == ut.bits as u64) {
            return Err(Error::NoMemory);
        }
        for i in 0..count {
            let s = self.slot_in(dst, index.checked_add(i).ok_or(Error::BadCap)?)?;
            if !self.slot(s).is_empty() {
                return Err(Error::SlotFull);
            }
        }
        if !self.has_children(us) {
            uslot.badge.set(0);
            uslot.aux.set(0);
        }
        let size = 1u64 << obits;
        let start = (uslot.badge.get() + size - 1) & !(size - 1);
        let end = start.checked_add(count << obits).ok_or(Error::NoMemory)?;
        if end > 1u64 << ut.bits {
            return Err(Error::NoMemory);
        }
        let zeroed = ty != ObjectType::Untyped && !kept;
        if zeroed {
            // Memory in [free, cleared) is zero. Extend that to `end`.
            let mut cleared = uslot.aux.get().max(start);
            while cleared < end {
                let n = ZERO_CHUNK.min(end - cleared);
                self.phys.zero(ut.obj + cleared, n);
                cleared += n;
                uslot.aux.set(cleared);
                if cleared < end && self.hal.irq_pending() {
                    return Ok(Exit::Preempted);
                }
            }
        }
        let cleared = uslot.aux.get();
        for i in 0..count {
            let obj = ut.obj + start + (i << obits);
            let mut cap = Cap { ty, rights: Rights::ALL, bits: cbits as u8, flags: ut.flags, obj, badge: 0 };
            match ty {
                ObjectType::Thread => {
                    self.hal.init_thread(&self.phys, obj);
                    // A thread starts on the processor that made it.
                    self.thread(obj).cpu.set(self.hal.cpu() as u64);
                }
                ObjectType::SchedContext => self.init_sched(obj),
                ObjectType::AddressSpace => {
                    self.phys.obj::<AsHeader>(obj).bits.set(obits);
                    if io {
                        cap.flags |= FLAG_IO;
                        self.init_io_space(obj);
                    } else {
                        self.hal.init_root(&self.phys, obj + (1 << PAGE_BITS));
                    }
                }
                _ => {}
            }
            if ty == ObjectType::Frame {
                cap.rights = Rights(Rights::READ.0 | Rights::WRITE.0 | Rights::EXEC.0);
            }
            let s = self.slot_in(dst, index + i)?;
            self.insert_child(us, s, &cap)?;
            if ty == ObjectType::Untyped {
                // The part of the child the parent had already zeroed.
                let off = start + (i << obits);
                self.slot(s).aux.set(cleared.saturating_sub(off).min(size));
            }
        }
        uslot.badge.set(end);
        Ok(Exit::Done)
    }
}
