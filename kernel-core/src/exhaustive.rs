//! Revoke completeness, by exhaustion (`spec/MILESTONES.md` M2, `spec/TESTING.md`
//! section 3).
//!
//! The kernel's own `revoke` runs on every derivation list of up to
//! [`SLOTS`] slots that the capability system can produce: every assignment
//! of depths in which each slot is at most one deeper than the one before
//! it, every mix of capabilities whose destruction differs (a frame; an
//! endpoint, sharing its object with others or not, so that deleting it may
//! or may not destroy a queue), every slot to revoke, and every pattern of
//! interrupts arriving at the points where revoke looks for them, each
//! interrupted revoke restarted as the system call is. After every call:
//!
//! - when it reports done, nothing derived from the revoked capability is
//!   left (the slot after it is the end of the list or no deeper);
//! - only its descendants are gone, a prefix of them when interrupted, and
//!   every other capability is unchanged, its own included;
//! - the list is well formed: every link has its reverse, it ends.
//!
//! Exhaustion over these bounds is the proof a model checker would give
//! over the same bounds; Kani did not finish one harness for three slots in
//! 30 minutes (every slot address read back from memory is symbolic), while
//! this covers five slots in seconds. The list's slots are linked in address
//! order: revoke compares addresses only with 0 and follows links, so the
//! order stands for every placement.
#![allow(unsafe_code)]

extern crate std;

use crate::cap::Cap;
use crate::hal::Hal;
use crate::kernel::Kernel;
use crate::mem::{Paddr, Phys};
use core::cell::Cell;
use silica_abi::{ObjectType, Rights, SLOT_BITS};
use std::vec::Vec;
use std::{assert, assert_eq, vec};

/// The longest derivation list.
const SLOTS: usize = 5;
/// Physical memory: address 0 unused, slots from 64, two endpoint queues after them.
#[repr(C, align(4096))]
struct Mem([u8; 64 * (SLOTS + 3)]);
const QUEUE: [Paddr; 2] = [(64 * (SLOTS + 1)) as Paddr, (64 * (SLOTS + 2)) as Paddr];
/// A frame's object address (never dereferenced: the frames are not mapped).
const FRAME: Paddr = 0x10_0000;

std::thread_local! {
    /// Which of revoke's interrupt polls see an interrupt: bit `i` for the
    /// `i`-th poll since the schedule was set.
    static SCHEDULE: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
}

struct Hal0;

impl Hal for Hal0 {
    const IP: usize = 0;
    const SP: usize = 1;
    const FLAGS: usize = 2;
    const SYS: usize = 3;
    const ARG: [usize; 6] = [4, 5, 6, 7, 8, 9];
    const BY_SYSCALL: usize = 10;
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
        SCHEDULE.with(|s| {
            let (bits, n) = s.get();
            s.set((bits, n + 1));
            n < 32 && bits >> n & 1 != 0
        })
    }
    fn idle_begin(&self) {}
    fn idle(&self, _: u64) {}
    fn timer_armed(&self) -> bool {
        false
    }
    fn irq_route(&self, _: usize, _: usize, _: bool, _: bool) -> bool {
        false
    }
    fn irq_mask(&self, _: usize, _: bool) {}
    fn port_io(&self, _: u16, _: u8, _: bool, _: u32) -> u32 {
        0
    }
    fn console(&self, _: &[u8]) {}
    fn power_off(&self, _: u64) {}
}

fn slot_addr(i: usize) -> Paddr {
    (i as u64) << SLOT_BITS
}

/// What slot `i` of a list holds: 0 a frame, 1 an endpoint to the first
/// queue, 2 an endpoint to the second.
fn cap_of(kind: u32) -> Cap {
    let (ty, obj) = match kind {
        0 => (ObjectType::Frame, FRAME),
        k => (ObjectType::Endpoint, QUEUE[k as usize - 1]),
    };
    Cap { ty, rights: Rights::ALL, bits: if ty == ObjectType::Frame { 12 } else { 6 }, flags: 0, obj, badge: 0 }
}

/// Every depth sequence of length `n`: the first at most 2, each next at most one deeper.
fn depth_sequences(n: usize) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    let mut d = vec![0u16; n];
    fn go(i: usize, d: &mut Vec<u16>, out: &mut Vec<Vec<u16>>) {
        if i == d.len() {
            out.push(d.clone());
            return;
        }
        let top = if i == 0 { 2 } else { d[i - 1] + 1 };
        for v in 0..=top {
            d[i] = v;
            go(i + 1, d, out);
        }
    }
    go(0, &mut d, &mut out);
    out
}

/// Check the list after one call of revoke on slot `s`; returns whether it is done.
fn check(k: &Kernel<Hal0>, n: usize, caps: &[Option<Cap>], depth: &[u16], s: usize, end: usize, done: bool) {
    let mut gone_seen_present = false;
    for (i, &want) in caps.iter().enumerate().take(n + 1).skip(1) {
        let now = k.slot(slot_addr(i)).cap();
        if i <= s || i >= end {
            assert_eq!(now, want, "slot {i} is not derived from slot {s} and changed (depths {depth:?})");
        } else if done {
            assert!(now.is_none(), "after a revoke of slot {s} that is done, slot {i} derived from it remains (depths {depth:?})");
        } else {
            // Interrupted: what is gone is a prefix of the descendants.
            if now.is_some() {
                gone_seen_present = true;
            } else {
                assert!(!gone_seen_present, "an interrupted revoke left a hole in slot {s}'s descendants (depths {depth:?})");
            }
        }
    }
    let next = k.slot(slot_addr(s)).next.get();
    if done {
        assert!(next == 0 || k.slot(next).depth() <= depth[s - 1], "slot {s} still has a child after revoke (depths {depth:?})");
    }
    let (mut at, mut steps, mut prev) = (slot_addr(1), 0, 0);
    while at != 0 {
        assert!(steps <= n, "the list does not end");
        assert_eq!(k.slot(at).prev.get(), prev, "a link without its reverse");
        prev = at;
        at = k.slot(at).next.get();
        steps += 1;
    }
}

#[test]
fn revoke_is_complete_on_every_list_of_up_to_five_slots() {
    let mut mem = Mem([0; 64 * (SLOTS + 3)]);
    // SAFETY: every address the kernel is given here is inside `mem` (the
    // slots and the two queues) or is a frame address that is only compared;
    // `mem` outlives the kernel.
    let k = Kernel::new(Hal0, unsafe { Phys::new(mem.0.as_mut_ptr()) });
    let mut cases = 0u64;
    // Under Miri (which checks this test's use of memory, not the kernel's
    // logic, and runs it a thousand times slower) lists of up to three.
    let longest = if cfg!(miri) { 3 } else { SLOTS };
    for n in 1..=longest {
        for depth in depth_sequences(n) {
            for kinds in 0..3u32.pow(n as u32) {
                for s in 1..=n {
                    let end = (s + 1..=n).find(|&j| depth[j - 1] <= depth[s - 1]).unwrap_or(n + 1);
                    // Each poll can see an interrupt or not; revoke polls once
                    // per descendant removed, so `end - s - 1` bits cover them all.
                    for schedule in 0..1u32 << (end - s - 1) {
                        // A fresh list in memory: the slots and the queues
                        // (through the kernel's own view of it, the only one
                        // in use while it lives).
                        k.phys.zero(0, (64 * (SLOTS + 3)) as u64);
                        let mut caps = [None; SLOTS + 1];
                        let mut kk = kinds;
                        for i in 1..=n {
                            let c = cap_of(kk % 3);
                            kk /= 3;
                            k.slot(slot_addr(i)).set(&c, depth[i - 1]);
                            if i > 1 {
                                k.slot(slot_addr(i)).prev.set(slot_addr(i - 1));
                                k.slot(slot_addr(i - 1)).next.set(slot_addr(i));
                            }
                            caps[i] = Some(c);
                        }
                        SCHEDULE.with(|c| c.set((schedule, 0)));
                        let mut calls = 0;
                        loop {
                            let done = k.revoke(slot_addr(s));
                            check(&k, n, &caps, &depth, s, end, done);
                            calls += 1;
                            assert!(calls <= n + 1, "revoke made no progress");
                            if done {
                                break;
                            }
                        }
                        assert_eq!(k.reap.get(), 0, "something was left on the reap list");
                        cases += 1;
                    }
                }
            }
        }
    }
    // Every case above ran; the count is here so a change to the bounds is seen.
    std::println!("{cases} cases");
    assert!(cases > if cfg!(miri) { 100 } else { 100_000 }, "only {cases} cases");
}
