//! Turning Untyped memory into objects.
//!
//! An Untyped is a power-of-two range with a watermark: objects are carved
//! upward from it and the range is reused only after every object made from
//! it is gone (`CapRevoke` on the Untyped). Memory is zeroed before it
//! becomes an object; the zeroing is done in pieces and remembered, so an
//! interrupt can preempt a large `Retype` without repeating finished work.

use crate::cap::{Cap, FLAG_DEVICE};
use crate::hal::Hal;
use crate::kernel::{Exit, Kernel};
use crate::mem::Paddr;
use crate::vm::AsHeader;
use silica_abi::{
    ASPACE_MIN_BITS, ENDPOINT_BITS, Error, ObjectType, PAGE_BITS, RETYPE_MAX, Rights, SLOT_BITS,
    THREAD_BITS, UNTYPED_MAX_BITS,
};

/// Bytes zeroed between checks for a pending interrupt.
const ZERO_CHUNK: u64 = 1 << 18;

impl<H: Hal> Kernel<H> {
    /// Create `count` objects of type `ty` from the Untyped in slot `us`,
    /// placing their capabilities in `dst` slots `index..index + count`.
    pub(crate) fn retype(
        &self,
        us: Paddr,
        ty: ObjectType,
        bits: u64,
        dst: &Cap,
        index: u64,
        count: u64,
    ) -> Result<Exit, Error> {
        let uslot = self.slot(us);
        let ut = uslot.cap().ok_or(Error::BadCap)?;
        if ut.ty != ObjectType::Untyped {
            return Err(Error::WrongType);
        }
        let device = ut.flags & FLAG_DEVICE != 0;
        // (object size in bits, the `bits` field of the new capability)
        let (obits, cbits) = match ty {
            ObjectType::Untyped if (SLOT_BITS as u64..=UNTYPED_MAX_BITS as u64).contains(&bits) => {
                (bits, bits)
            }
            ObjectType::Frame if (PAGE_BITS as u64..=UNTYPED_MAX_BITS as u64).contains(&bits) => {
                (bits, bits)
            }
            ObjectType::CapSpace if (1..=32).contains(&bits) => (bits + SLOT_BITS as u64, bits),
            ObjectType::AddressSpace if (ASPACE_MIN_BITS as u64..=30).contains(&bits) => {
                (bits, bits)
            }
            ObjectType::Thread => (THREAD_BITS as u64, THREAD_BITS as u64),
            ObjectType::Endpoint | ObjectType::Notification => {
                (ENDPOINT_BITS as u64, ENDPOINT_BITS as u64)
            }
            ObjectType::SchedContext | ObjectType::Vcpu => return Err(Error::Unsupported),
            _ => return Err(Error::BadArg),
        };
        if device && !matches!(ty, ObjectType::Untyped | ObjectType::Frame) {
            return Err(Error::BadArg);
        }
        if count == 0 || count > RETYPE_MAX {
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
        let zeroed = ty != ObjectType::Untyped && !device;
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
            let mut cap = Cap {
                ty,
                rights: Rights::ALL,
                bits: cbits as u8,
                flags: ut.flags,
                obj,
                badge: 0,
            };
            match ty {
                ObjectType::Thread => self.hal.init_thread(&self.phys, obj),
                ObjectType::AddressSpace => {
                    self.phys.obj::<AsHeader>(obj).bits.set(obits);
                    self.hal.init_root(&self.phys, obj + (1 << PAGE_BITS));
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
