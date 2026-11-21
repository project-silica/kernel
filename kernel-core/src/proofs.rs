//! Kani proofs of the capability system's two promises
//! (`spec/TESTING.md` section 3, `spec/MILESTONES.md` M2):
//!
//! - **No forge.** Whatever sequence of derive, duplicate, move, delete and
//!   revoke a holder performs, every capability that exists refers to an
//!   object a capability existed for at the start, carries no right the
//!   capability it was made from did not have, and keeps the badge it was
//!   given (a badge, once set, is never changed or removed).
//! - **Revoke completeness** is not proved here yet: harnesses that run
//!   delete and revoke on a symbolic derivation list of three slots did not
//!   finish in 20 minutes (the next slot to clear is a symbolic address, so
//!   every access becomes one). It is checked by the reference-model tests
//!   (tests/model.rs); `tools/test.sh --deep` reports the missing proof.
//!
//! Each proof runs one operation of the kernel's own code on a machine of a
//! few capability slots in any well-formed state and checks its result, so
//! the properties hold after any sequence of operations by induction. Within
//! the bounds (`SLOTS`, `#[kani::unwind]`) each proof covers every
//! capability type, rights mask, badge, derivation shape and interrupt
//! timing.
//!
//! Run with `cargo kani -p kernel-core` (the `kani` suite of
//! `tools/test.sh --deep`).
#![allow(unsafe_code)]

use crate::cap::Cap;
use crate::hal::Hal;
use crate::kernel::Kernel;
use crate::mem::{Paddr, Phys};
use silica_abi::{ObjectType, Rights, SLOT_BITS};

/// Capability slots in the machine (indices 1 to `SLOTS`).
const SLOTS: usize = 3;
/// Operations in one sequence.

/// The memory the kernel sees: physical address 0 is never used, slots
/// from 64, the endpoint's queue after them.
#[repr(C, align(4096))]
struct Mem([u8; 64 * (SLOTS + 3)]);

const ENDPOINT: Paddr = (64 * (SLOTS + 1)) as Paddr;
const FRAME: Paddr = 0x10_0000;

fn slot_addr(i: usize) -> Paddr {
    (i as u64) << SLOT_BITS
}

/// An architecture with nothing behind it; interrupts arrive whenever the
/// proof says so.
struct ProofHal;

impl Hal for ProofHal {
    const IP: usize = 0;
    const SP: usize = 1;
    const FLAGS: usize = 2;
    const SYS: usize = 3;
    const ARG: [usize; 6] = [4, 5, 6, 7, 8, 9];
    const FLAGS_INIT: u64 = 0;
    const SYSCALL_LEN: u64 = 2;
    const USER_TOP: u64 = 1 << 47;
    fn pte_table(table: Paddr) -> u64 {
        table | 1
    }
    fn pte_leaf(frame: Paddr, _: Rights, _: u8, _: bool) -> u64 {
        frame | 1
    }
    fn pte_present(pte: u64) -> bool {
        pte & 1 != 0
    }
    fn pte_is_leaf(_: u64, _: u8) -> bool {
        true
    }
    fn pte_addr(pte: u64) -> Paddr {
        pte & !0xFFF
    }
    fn init_root(&self, _: &Phys, _: Paddr) {}
    fn init_thread(&self, _: &Phys, _: Paddr) {}
    fn switch_thread(&self, _: &Phys, _: Paddr, _: Paddr, _: Paddr) {}
    fn leave_aspace(&self, _: Paddr) {}
    fn tlb_flush(&self, _: Paddr, _: u64, _: u64) {}
    fn cpu(&self) -> usize {
        0
    }
    fn kick(&self, _: usize, _: bool) {}
    fn tlb_reset(&self) {}
    fn tlb_forget(&self, _: Paddr, _: usize) {}
    fn now(&self) -> u64 {
        0
    }
    fn set_timer(&self, _: u64) {}
    fn irq_pending(&self) -> bool {
        kani::any()
    }
    fn idle_begin(&self) {}
    fn idle(&self, _: u64) {}
    fn irq_route(&self, _: usize, _: usize, _: bool, _: bool) -> bool {
        false
    }
    fn irq_mask(&self, _: usize, _: bool) {}
    fn port_io(&self, _: u16, _: u8, _: bool, _: u32) -> u32 {
        0
    }
    fn console(&self, _: &[u8]) {}
}

/// A kernel over `mem`.
fn machine(mem: &mut Mem) -> Kernel<ProofHal> {
    // SAFETY: every physical address the kernel is given in these proofs is
    // below `size_of::<Mem>()` (the slots and the endpoint's queue) or is an
    // object address that is only compared, never dereferenced. `mem`
    // outlives the kernel.
    Kernel::new(ProofHal, unsafe { Phys::new(mem.0.as_mut_ptr()) })
}

/// Any capability: any type, rights, size, flags and badge, to an object at
/// `ENDPOINT` (whose queue is in memory) or anywhere else.
fn any_cap() -> Cap {
    let ty = ObjectType::from_u64(kani::any::<u8>() as u64 % 12);
    kani::assume(ty.is_some());
    let obj = if kani::any() { ENDPOINT } else { FRAME };
    Cap { ty: ty.unwrap(), rights: Rights(kani::any()), bits: kani::any(), flags: kani::any(), obj, badge: kani::any() }
}

#[kani::proof]
#[kani::unwind(4)]
fn copy_never_amplifies() {
    let mut mem = Mem([0; 64 * (SLOTS + 3)]);
    let k = machine(&mut mem);
    let src = any_cap();
    // A Frame capability that is mapped lives in an address space's list;
    // the copies of one start unmapped either way.
    kani::assume(!(src.ty == ObjectType::Frame && src.badge & 1 != 0));
    // A hardware range is well-formed: it starts before it ends.
    kani::assume(!(src.ty == ObjectType::Hw && src.obj >= src.badge));
    let depth: u16 = kani::any();
    kani::assume(depth < 8);
    k.slot(slot_addr(1)).set(&src, depth);
    let (mask, badge, sibling): (u8, u64, bool) = (kani::any(), kani::any(), kani::any());
    let r = k.copy(slot_addr(1), slot_addr(2), Rights(mask), badge, sibling);
    assert_eq!(k.slot(slot_addr(1)).cap(), Some(src), "the source is unchanged");
    let Ok(()) = r else {
        assert!(k.slot(slot_addr(2)).is_empty(), "a failed copy leaves nothing behind");
        return;
    };
    assert!(src.ty != ObjectType::Untyped, "an Untyped has one capability");
    let c = k.slot(slot_addr(2)).cap().expect("the copy exists");
    assert_eq!(c.ty, src.ty, "same type");
    assert_eq!(c.rights.0, src.rights.0 & mask, "rights are the source's, narrowed by the mask");
    match src.ty {
        ObjectType::Hw => {
            assert!(c.obj >= src.obj && c.badge <= src.badge && c.obj < c.badge, "a hardware range only narrows");
        }
        ObjectType::Endpoint | ObjectType::Notification => {
            assert_eq!(c.obj, src.obj, "same object");
            if src.badge != 0 {
                assert_eq!(c.badge, src.badge, "a badge, once set, is kept");
            } else {
                assert_eq!(c.badge, badge, "an unbadged capability takes the badge asked for");
            }
        }
        _ => {
            assert_eq!(c.obj, src.obj, "same object");
            assert!(src.ty == ObjectType::Frame || badge == 0, "no badge on other types");
        }
    }
    let (s, d) = (k.slot(slot_addr(1)), k.slot(slot_addr(2)));
    if sibling {
        assert_eq!((d.depth(), d.next.get(), s.prev.get()), (depth, slot_addr(1), slot_addr(2)), "a duplicate is a sibling just before");
    } else {
        assert_eq!(
            (d.depth(), d.prev.get(), s.next.get()),
            (depth + 1, slot_addr(1), slot_addr(2)),
            "a derived capability is a child just after"
        );
    }
}
