//! Scenario tests of the kernel through its syscall interface, on the host machine.
mod support;

use kernel_core::elf::Elf;
use kernel_core::hal::Hal as _;
use kernel_core::thread::{INACTIVE, RECEIVING, REPLY, RUNNABLE, SENDING};
use silica_abi::root::*;
use silica_abi::{Error, FAULT_LABEL, FAULT_RESUME, Fault, KEEP, MsgInfo, NONBLOCK, ObjectType as T, Rights, Syscall as S};
use support::*;

const FREE: u64 = 2000; // first slot the tests use

#[test]
fn root_task_starts_with_documented_layout() {
    let m = Machine::new(16);
    let i = m.info();
    assert_eq!(i.magic, ROOT_MAGIC);
    assert_eq!(i.capspace_slots, 4096);
    assert_eq!(&i.cmdline[..i.cmdline_len as usize], b"test=host");
    assert_eq!(i.entropy, [7; 32]);
    assert_eq!(m.cap(SLOT_THREAD).unwrap().ty, T::Thread);
    assert_eq!(m.cap(SLOT_THREAD).unwrap().obj, m.root);
    assert_eq!(m.cap(SLOT_CAPSPACE).unwrap().ty, T::CapSpace);
    assert_eq!(m.cap(SLOT_ASPACE).unwrap().ty, T::AddressSpace);
    assert_eq!(m.cap(SLOT_PORTS).unwrap().ty, T::Hw);
    assert_eq!(m.cap(SLOT_IPCBUF).unwrap().ty, T::Frame);
    assert_eq!(m.cap(SLOT_BOOT_UNTYPED).unwrap().ty, T::Untyped);
    assert!(m.cap(i.first_free_slot).is_none());
    assert!(m.cap(i.first_free_slot - 1).is_some());
    // Every byte of memory is covered exactly once by the boot Untyped, the
    // listed Untypeds, and nothing else; page 0 never appears as ordinary memory.
    let mut covered = 1u64 << m.cap(SLOT_BOOT_UNTYPED).unwrap().bits;
    for (slot, d) in m.untyped() {
        let c = m.cap(slot).unwrap();
        assert_eq!((c.ty, c.obj, c.bits), (T::Untyped, d.base, d.bits));
        assert!(d.base != 0 || d.kind != KIND_RAM);
        assert_eq!(d.base % (1 << d.bits), 0);
        if d.base >= RAM_BASE {
            covered += 1 << d.bits;
        }
    }
    // The low device window has two holes reserved by the platform, one
    // described by overlapping records. Every other page is granted once.
    let untyped = m.untyped();
    for page in (0..RAM_BASE).step_by(4096) {
        let owners = untyped.iter().filter(|(_, d)| d.base <= page && page < d.base + (1 << d.bits)).count();
        let reserved = (0x3000..0x8000).contains(&page) || (0xA000..0xD000).contains(&page);
        assert_eq!(owners, usize::from(!reserved), "wrong coverage at {page:#x}");
    }
    assert_eq!(covered, 16 << 20);
    assert!(m.untyped().iter().any(|(_, d)| d.kind == KIND_RAMDISK && d.bits == 14));
    // The image is mapped: its entry point translates.
    let asp = m.cap(SLOT_ASPACE).unwrap().obj;
    assert!(m.k.translate(asp, 0x40_0000).is_some());
    assert!(m.k.translate(asp, 0x60_2000).is_some());
    assert!(m.k.translate(asp, 0x60_3000).is_none());
}

#[test]
fn bad_syscalls_and_capabilities_are_errors() {
    let m = Machine::new(16);
    let r = m.raw(999, [0; 6]);
    assert_eq!(Error::result(r.status), Err(Error::BadSyscall));
    assert_eq!(m.err(S::Call, [0, 0, 0, 0, 0, 0]), Error::BadCap);
    assert_eq!(m.err(S::Call, [5000, 0, 0, 0, 0, 0]), Error::BadCap);
    assert_eq!(m.err(S::Call, [FREE, 0, 0, 0, 0, 0]), Error::BadCap);
    assert_eq!(m.err(S::Call, [SLOT_THREAD, 0, 0, 0, 0, 0]), Error::WrongType);
    assert_eq!(m.err(S::Map, [SLOT_ASPACE, SLOT_THREAD, 0, 1, 0, 0]), Error::WrongType);
    assert_eq!(m.err(S::ReplyRecv, [0; 6]), Error::NoCaller);
}

#[test]
fn retype_checks_and_untyped_balances() {
    let m = Machine::new(16);
    let (ut, d) = m.big_untyped();
    let cs = SLOT_CAPSPACE;
    assert_eq!(m.err(S::Retype, [ut, 99, 0, cs, FREE, 1]), Error::BadArg);
    assert_eq!(m.err(S::Retype, [ut, T::Hw as u64, 0, cs, FREE, 1]), Error::BadArg);
    assert_eq!(m.err(S::Retype, [ut, T::Vcpu as u64, 0, cs, FREE, 1]), Error::Unsupported);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, 11, cs, FREE, 1]), Error::BadArg);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, 12, cs, FREE, 0]), Error::BadArg);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, 12, cs, FREE, 257]), Error::BadArg);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, 12, cs, 4095, 2]), Error::BadCap);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, d.bits as u64 + 1, cs, FREE, 1]), Error::NoMemory);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, 48, cs, FREE, 1]), Error::BadArg);
    assert_eq!(m.err(S::Retype, [ut, T::Frame as u64, 12, cs, SLOT_THREAD, 1]), Error::SlotFull);
    assert_eq!(m.err(S::Retype, [SLOT_THREAD, T::Frame as u64, 12, cs, FREE, 1]), Error::WrongType);
    // Fill it exactly, then one more fails and changes nothing.
    m.retype(ut, T::Frame, d.bits as u64 - 1, FREE, 2);
    assert_eq!(m.err(S::Retype, [ut, T::Endpoint as u64, 0, cs, FREE + 2, 1]), Error::NoMemory);
    assert!(m.cap(FREE + 2).is_none());
    let (a, b) = (m.cap(FREE).unwrap(), m.cap(FREE + 1).unwrap());
    assert_eq!((a.obj, b.obj), (d.base, d.base + (1 << (d.bits - 1))));
    // Objects are zeroed: the machine's memory starts as 0xA5.
    assert!(m.is_zero(a.obj, 4096));
    // Deleting one child does not free the Untyped; revoking does.
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(m.err(S::Retype, [ut, T::Endpoint as u64, 0, cs, FREE, 1]), Error::NoMemory);
    m.ok(S::CapRevoke, [ut, 0, 0, 0, 0, 0]);
    assert!(m.cap(FREE + 1).is_none());
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    assert_eq!(m.cap(FREE).unwrap().obj, d.base);
    // An Untyped cannot be copied, and a child Untyped is strictly smaller.
    assert_eq!(m.err(S::CapCopy, [cs, FREE + 5, ut, 15, 0, 0]), Error::InUse);
    assert_eq!(m.err(S::Retype, [ut, T::Untyped as u64, d.bits as u64, cs, FREE + 5, 1]), Error::NoMemory);
}

#[test]
fn large_retype_is_preempted_and_restarted_without_repeating_work() {
    let m = Machine::new(16);
    let (ut, d) = m.big_untyped();
    assert!(d.bits >= 22);
    m.k.hal.pending.set(5);
    m.retype(ut, T::Frame, 22, FREE, 1);
    assert_eq!(m.k.hal.pending.get(), 0, "the retype polled for interrupts while zeroing");
    let f = m.cap(FREE).unwrap();
    assert!(m.is_zero(f.obj, 1 << 22));
}

#[test]
fn copy_narrows_rights_and_revoke_deletes_descendants() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let cs = SLOT_CAPSPACE;
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [cs, FREE + 1, FREE, 3, 42, 0]);
    let c = m.cap(FREE + 1).unwrap();
    assert_eq!((c.rights, c.badge), (Rights(3), 42));
    // A badged capability cannot be re-badged; rights never grow.
    assert_eq!(m.err(S::CapCopy, [cs, FREE + 2, FREE + 1, 15, 43, 0]), Error::BadArg);
    m.ok(S::CapCopy, [cs, FREE + 2, FREE + 1, 15, 0, 0]);
    assert_eq!(m.cap(FREE + 2).unwrap().rights, Rights(3));
    assert_eq!(m.err(S::CapCopy, [cs, FREE + 2, FREE, 15, 0, 0]), Error::SlotFull);
    assert_eq!(m.err(S::CapCopy, [cs, FREE + 3, FREE + 9, 15, 0, 0]), Error::BadCap);
    // Move keeps the derivation: revoking the original still reaches it.
    m.ok(S::CapMove, [cs, FREE + 7, FREE + 2, 0, 0, 0]);
    assert!(m.cap(FREE + 2).is_none());
    assert_eq!(m.err(S::CapMove, [cs, FREE + 7, FREE + 1, 0, 0, 0]), Error::SlotFull);
    assert_eq!(m.err(S::CapMove, [cs, FREE + 8, FREE + 2, 0, 0, 0]), Error::BadCap);
    m.ok(S::CapRevoke, [FREE, 0, 0, 0, 0, 0]);
    assert!(m.cap(FREE).is_some());
    assert!(m.cap(FREE + 1).is_none() && m.cap(FREE + 7).is_none());
    assert_eq!(m.err(S::CapRevoke, [FREE + 1, 0, 0, 0, 0, 0]), Error::BadCap);
    // Hw ranges narrow and never widen.
    m.ok(S::CapCopy, [cs, FREE + 1, SLOT_PORTS, 15, 0x3F8 | 0x400 << 32, 0]);
    assert_eq!(m.err(S::CapCopy, [cs, FREE + 2, FREE + 1, 15, 0x3F0 | 0x400 << 32, 0]), Error::BadArg);
    assert_eq!(m.err(S::PortIo, [FREE + 1, 0x3F7, 1, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::PortIo, [FREE + 1, 0x3FF, 2, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::PortIo, [FREE + 1, 0x3F8, 3, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::PortIo, [SLOT_IRQS, 0x3F8, 1, 0, 0, 0]), Error::WrongType);
    m.ok(S::PortIo, [FREE + 1, 0x3F8, 1, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0x3F8);
}

#[test]
fn deleting_a_capspace_deletes_what_it_holds_without_recursion() {
    let m = Machine::new(64);
    let (ut, _) = m.big_untyped();
    // A chain of 300 CapSpaces, each holding the only capability to the next.
    m.retype(ut, T::CapSpace, 1, FREE, 1);
    for _ in 0..if cfg!(miri) { 30 } else { 300 } {
        m.retype(ut, T::CapSpace, 1, FREE + 1, 1);
        m.ok(S::CapMove, [FREE + 1, 1, FREE, 0, 0, 0]);
        m.ok(S::CapMove, [SLOT_CAPSPACE, FREE, FREE + 1, 0, 0, 0]);
    }
    m.retype(ut, T::Endpoint, 0, FREE + 2, 1);
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    // Everything is gone, so the Untyped has only the endpoint left.
    m.ok(S::CapDelete, [FREE + 2, 0, 0, 0, 0, 0]);
    let v = m.k.view(m.k.slot_addr(&m.cs, ut).unwrap()).unwrap();
    assert_eq!(v.next, 0, "no capability derived from the Untyped remains");
}

#[test]
fn map_enforces_rights_alignment_and_w_xor_x() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let (cs, asp) = (SLOT_CAPSPACE, SLOT_ASPACE);
    m.retype(ut, T::Frame, 12, FREE, 2);
    m.retype(ut, T::Frame, 21, FREE + 2, 1);
    let a = m.cap(asp).unwrap().obj;
    assert_eq!(m.err(S::Map, [asp, FREE, 0x1000_0000, 7, 0, 0]), Error::BadArg, "writable and executable");
    assert_eq!(m.err(S::Map, [asp, FREE, 0x1000_0000, 2, 0, 0]), Error::BadArg, "write without read");
    assert_eq!(m.err(S::Map, [asp, FREE, 0x1000_0800, 3, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::Map, [asp, FREE, 0x7FFF_FFFF_F000, 3, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::Map, [asp, FREE, 0xFFFF_8000_0000_0000, 3, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::Map, [asp, FREE + 2, 0x1000_1000, 3, 0, 0]), Error::BadArg, "large frame alignment");
    m.ok(S::Map, [asp, FREE, 0x1000_0000, 3, 0, 0]);
    assert_eq!(m.k.translate(a, 0x1000_0123), Some(m.cap(FREE).unwrap().obj + 0x123));
    assert_eq!(m.err(S::Map, [asp, FREE, 0x1000_1000, 3, 0, 0]), Error::Mapped);
    assert_eq!(m.err(S::Map, [asp, FREE + 1, 0x1000_0000, 3, 0, 0]), Error::Overlap);
    assert_eq!(m.err(S::Map, [asp, FREE + 2, 0x1000_0000, 3, 0, 0]), Error::Overlap, "large page over a table");
    m.ok(S::Map, [asp, FREE + 2, 0x4000_0000, 1, 0, 0]);
    assert_eq!(m.k.translate(a, 0x4010_0000), Some(m.cap(FREE + 2).unwrap().obj + 0x10_0000));
    assert_eq!(m.err(S::Map, [asp, FREE + 1, 0x4000_1000, 3, 0, 0]), Error::Overlap, "page under a large page");
    // A read-only copy cannot be mapped writable.
    m.ok(S::CapCopy, [cs, FREE + 3, FREE + 1, 1, 0, 0]);
    assert_eq!(m.err(S::Map, [asp, FREE + 3, 0x1000_2000, 3, 0, 0]), Error::NoRights);
    m.ok(S::Map, [asp, FREE + 3, 0x1000_2000, 1, 0, 0]);
    // Moving a mapped capability keeps the mapping; deleting it unmaps.
    m.ok(S::CapMove, [cs, FREE + 4, FREE + 3, 0, 0, 0]);
    assert!(m.k.translate(a, 0x1000_2000).is_some());
    let flushes = m.k.hal.flushes.get();
    m.ok(S::CapDelete, [FREE + 4, 0, 0, 0, 0, 0]);
    assert!(m.k.translate(a, 0x1000_2000).is_none());
    assert_eq!(m.k.hal.flushes.get(), flushes + 1, "unmapping flushes the TLB");
    m.ok(S::Unmap, [FREE, 0, 0, 0, 0, 0]);
    assert!(m.k.translate(a, 0x1000_0000).is_none());
    m.ok(S::Unmap, [FREE, 0, 0, 0, 0, 0]);
    // Revoking the Untyped a mapped frame came from unmaps it.
    m.ok(S::CapRevoke, [ut, 0, 0, 0, 0, 0]);
    assert!(m.k.translate(a, 0x4010_0000).is_none());
}

#[test]
fn address_space_table_pool_runs_out_cleanly_and_destroying_it_unmaps_frames() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    // Header, root and two tables: one mapping needs three tables.
    m.retype(ut, T::AddressSpace, 14, FREE, 1);
    m.retype(ut, T::Frame, 12, FREE + 1, 2);
    assert_eq!(m.err(S::Map, [FREE, FREE + 1, 0x1000_0000, 3, 0, 0]), Error::NoMemory);
    // A refused small mapping must leave the two tables available for a
    // large page at a different root index.
    m.retype(ut, T::Frame, 21, FREE + 4, 1);
    m.ok(S::Map, [FREE, FREE + 4, 1 << 39, 3, 0, 0]);
    assert_eq!(m.k.translate(m.cap(FREE).unwrap().obj, (1 << 39) + 15), Some(m.cap(FREE + 4).unwrap().obj + 15));
    m.retype(ut, T::AddressSpace, 15, FREE + 3, 1);
    m.ok(S::Map, [FREE + 3, FREE + 1, 0x1000_0000, 3, 0, 0]);
    m.ok(S::Map, [FREE + 3, FREE + 2, 0x1000_1000, 1, 0, 0]);
    // Cross a table boundary into the first existing page. Both its
    // translation and the missing page immediately below must survive.
    m.retype(ut, T::Frame, 13, FREE + 5, 1);
    let space = m.cap(FREE + 3).unwrap().obj;
    assert_eq!(m.err(S::Map, [FREE + 3, FREE + 5, 0x0FFF_F000, 3, 0, 0]), Error::Overlap);
    assert_eq!(m.k.translate(space, 0x0FFF_F000), None);
    assert_eq!(m.k.translate(space, 0x1000_0000), Some(m.cap(FREE + 1).unwrap().obj));
    m.ok(S::Map, [FREE + 3, FREE + 5, 1 << 39, 3, 0, 0]);
    let root = m.cap(FREE + 3).unwrap().obj + 4096;
    m.ok(S::CapDelete, [FREE + 3, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.left.get(), root, "the processor is told to leave the dead address space");
    // The frames are unmapped again and can be mapped elsewhere.
    m.ok(S::Map, [SLOT_ASPACE, FREE + 1, 0x2000_0000, 3, 0, 0]);
}

/// Root plus one thread sharing its CapSpace; an endpoint at FREE.
fn pair() -> (Machine, u64) {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 77, 0]);
    let t = m.spawn(ut, FREE + 2);
    (m, t)
}

#[test]
fn call_and_reply_carry_message_badge_and_buffer_words() {
    let (m, t) = pair();
    let r = m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]);
    assert!(r.blocked);
    assert_eq!(m.k.thread_view(t).state, RECEIVING);
    let r = m.sys(S::Call, [FREE + 1, MsgInfo::new(9, 3).0, 11, 22, 33, 44]);
    assert!(r.blocked);
    assert_eq!(m.k.thread_view(m.root).state, REPLY);
    assert_eq!(m.k.current(), t);
    assert_eq!(m.status(t), 0);
    assert_eq!((m.arg(t, 0), MsgInfo(m.arg(t, 1)).label(), MsgInfo(m.arg(t, 1)).words()), (77, 9, 3));
    assert_eq!([m.arg(t, 2), m.arg(t, 3), m.arg(t, 4)], [11, 22, 33]);
    let r = m.sys_as(t, S::ReplyRecv, [FREE, MsgInfo::new(5, 1).0, 99, 0, 0, 0]);
    assert!(r.blocked, "the server waits for the next request");
    assert_eq!(m.k.thread_view(m.root).state, RUNNABLE);
    assert_eq!((m.status(m.root), MsgInfo(m.arg(m.root, 1)).label(), m.arg(m.root, 2)), (0, 5, 99));
    // A message longer than the registers needs both sides to have a buffer;
    // the spawned thread has none, so the message is cut to four words.
    m.sys(S::Call, [FREE, MsgInfo::new(1, 10).0, 1, 2, 3, 4]);
    assert_eq!(MsgInfo(m.arg(t, 1)).words(), 4);
    // The descriptor's fields do not run into each other at their limits.
    let full = MsgInfo::with_caps(0xABCD, 500, 4);
    assert_eq!((full.label(), full.words(), full.caps()), (0xABCD, 500, 4));
    assert_eq!(MsgInfo::with_caps(1, 1023, 7).words(), 500);
    assert_eq!(MsgInfo::with_caps(1, 1023, 7).caps(), 4);
}

#[test]
fn sender_queues_until_receiver_arrives_and_nonblocking_send_refuses() {
    let (m, t) = pair();
    assert_eq!(m.err(S::Send, [FREE | NONBLOCK, 0, 0, 0, 0, 0]), Error::WouldBlock);
    assert_eq!(m.err(S::Recv, [FREE, 1, 0, 0, 0, 0]), Error::WouldBlock);
    let r = m.sys(S::Send, [FREE + 1, MsgInfo::new(3, 1).0, 5, 0, 0, 0]);
    assert!(r.blocked);
    assert_eq!(m.k.thread_view(m.root).state, SENDING);
    let r = m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]);
    assert!(!r.blocked && r.status == 0);
    assert_eq!((m.arg(t, 0), m.arg(t, 2)), (77, 5));
    assert_eq!(m.k.thread_view(m.root).state, RUNNABLE);
    assert_eq!(m.status(m.root), 0);
    // Rights are checked on both sides.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 3, FREE, 1, 0, 0]);
    assert_eq!(m.err(S::Send, [FREE + 3, 0, 0, 0, 0, 0]), Error::NoRights);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 4, FREE, 2, 0, 0]);
    assert_eq!(m.err(S::Recv, [FREE + 4, 0, 0, 0, 0, 0]), Error::NoRights);
}

#[test]
fn destroying_an_endpoint_or_a_server_fails_waiters_with_gone() {
    let (m, t) = pair();
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::CapRevoke, [FREE, 0, 0, 0, 0, 0]);
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    assert_eq!(Error::result(m.status(t)), Err(Error::Gone));

    let (m, t) = pair();
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE, 0, 0, 0, 0, 0]).blocked);
    // The server dies holding the root's call. Its capability is in the
    // shared CapSpace, so the server itself can delete it.
    let r = m.sys_as(t, S::CapDelete, [FREE + 2, 0, 0, 0, 0, 0]);
    assert!(r.blocked);
    assert_eq!(m.k.thread_view(m.root).state, RUNNABLE);
    assert_eq!(Error::result(m.status(m.root)), Err(Error::Gone));
    assert_eq!(m.k.current(), m.root);
}

#[test]
fn a_second_call_received_before_replying_fails_the_first_caller() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    let t2 = m.spawn(ut, FREE + 5);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE, 0, 1, 0, 0, 0]).blocked);
    assert!(m.sys_as(t2, S::Call, [FREE, 0, 2, 0, 0, 0]).blocked);
    let r = m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]);
    assert!(!r.blocked);
    assert_eq!(m.arg(t, 2), 2);
    assert_eq!(Error::result(m.status(m.root)), Err(Error::Gone));
    assert_eq!(m.k.thread_view(t).caller, t2);
}

#[test]
fn notification_accumulates_badges_and_wait_honours_deadlines() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 0b01, 0]);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, FREE, 15, 0b10, 0]);
    let t = m.spawn(ut, FREE + 3);
    m.ok(S::Send, [FREE + 1, 0, 0, 0, 0, 0]);
    m.ok(S::Send, [FREE + 2, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0b11);
    assert_eq!(m.err(S::Recv, [FREE, 1, 0, 0, 0, 0]), Error::WouldBlock);
    m.ok(S::Send, [FREE, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1, "a capability without a badge signals bit 0");
    assert_eq!(m.err(S::Recv, [FREE, 500, 0, 0, 0, 0]), Error::Timeout, "a deadline in the past");
    // A waiter is woken by a signal with that signal's badge.
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Send, [FREE + 2, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t).state, m.status(t), m.arg(t, 0)), (RUNNABLE, 0, 0b10));
    // A deadline wakes the waiter with Timeout; the kernel arms the timer for it.
    let at = m.k.hal.now.get() + 5_000_000;
    assert!(m.sys_as(t, S::Recv, [FREE, at, 0, 0, 0, 0]).blocked);
    assert!(m.k.hal.timer() <= at);
    m.k.hal.now.set(at);
    m.k.on_timer();
    m.k.schedule();
    assert_eq!(Error::result(m.status(t)), Err(Error::Timeout));
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    // With every thread blocked the kernel idles until the deadline.
    let at = m.k.hal.now.get() + 1_000_000;
    assert!(m.sys_as(t, S::Recv, [FREE, at + 7, 0, 0, 0, 0]).blocked);
    let r = m.sys(S::Recv, [FREE, at, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.now.get(), at);
    assert_eq!(Error::result(m.status(r.thread)), Err(Error::Timeout));
}

#[test]
fn time_slices_rotate_runnable_threads() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let t = m.spawn(ut, FREE);
    assert_eq!(m.k.current(), m.root);
    assert!(m.k.hal.timer() != u64::MAX, "a second runnable thread arms the slice timer");
    m.k.hal.now.set(m.k.hal.timer());
    m.k.on_timer();
    m.k.schedule();
    assert_eq!(m.k.current(), t);
    m.sys_as(t, S::Yield, [0; 6]);
    assert_eq!(m.k.current(), m.root);
    m.ok(S::ThreadControl, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, INACTIVE);
    m.ok(S::ThreadControl, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    assert_eq!(m.err(S::ThreadControl, [FREE, 9, 0, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::ThreadControl, [FREE, 2, 1, 0, 0, 0]), Error::BadArg, "there is no processor 1");
    assert_eq!(m.err(S::ThreadStart, [FREE, 0xFFFF_8000_0000_0000, 0, 0, 1, 0]), Error::BadArg);
}

#[test]
fn fault_is_delivered_as_a_call_and_the_reply_decides() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 5, 0]);
    let t = m.spawn(ut, FREE + 2);
    m.ok(S::ThreadConfigure, [FREE + 2, KEEP, KEEP, FREE + 1, KEEP, 0]);
    m.sys_as(t, S::Yield, [0; 6]);
    // `t` is current again after the root's turn; make it fault.
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.k.regs(t)[0].set(0x1234);
    m.k.fault(Fault::Page, 0xdead_b000, 6);
    m.k.schedule();
    assert_eq!(m.k.thread_view(t).state, SENDING);
    let r = m.sys(S::Recv, [FREE, 0, 0, 0, 0, 0]);
    assert!(!r.blocked);
    let info = MsgInfo(m.arg(m.root, 1));
    assert_eq!((m.arg(m.root, 0), info.label(), info.words()), (5, FAULT_LABEL + Fault::Page as u64, 4));
    assert_eq!([m.arg(m.root, 2), m.arg(m.root, 3), m.arg(m.root, 4)], [0x1234, 0xdead_b000, 6]);
    assert_eq!(m.k.thread_view(t).state, REPLY);
    // Reply "resume": the thread runs again with its registers untouched.
    m.ok(S::ReplyRecv, [0, 0, FAULT_RESUME, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    assert_eq!(m.k.regs(t)[0].get(), 0x1234);
    // Fault again; any other reply leaves it stopped.
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.k.fault(Fault::Instruction, 0, 0);
    m.k.schedule();
    m.ok(S::Recv, [FREE, 0, 0, 0, 0, 0]);
    m.ok(S::ReplyRecv, [0, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, INACTIVE);
}

#[test]
fn fault_without_an_endpoint_stops_the_thread_and_is_logged() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let t = m.spawn(ut, FREE);
    m.sys_as(t, S::Yield, [0; 6]);
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.k.fault(Fault::Exception, 0, 13);
    m.k.schedule();
    assert_eq!(m.k.thread_view(t).state, INACTIVE);
    assert_eq!(m.k.current(), m.root);
    let log = String::from_utf8_lossy(&m.k.hal.console.borrow()).into_owned();
    assert!(log.contains("Exception fault") && log.contains("no fault endpoint"), "{log}");
}

#[test]
fn thread_configure_validates_before_changing_anything() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Thread, 0, FREE, 1);
    m.retype(ut, T::Frame, 13, FREE + 1, 1);
    let t = m.cap(FREE).unwrap().obj;
    assert_eq!(m.err(S::ThreadConfigure, [FREE, SLOT_CAPSPACE, SLOT_THREAD, 0, 0, 0]), Error::WrongType);
    assert!(m.k.view(m.k.thread_view(t).slots[0]).is_none(), "nothing was configured");
    assert_eq!(m.err(S::ThreadConfigure, [FREE, KEEP, KEEP, KEEP, FREE + 1, 0]), Error::BadArg, "IPC buffer is one page");
    m.ok(S::ThreadConfigure, [FREE, SLOT_CAPSPACE, SLOT_ASPACE, KEEP, SLOT_IPCBUF, 0]);
    assert_eq!(m.k.view(m.k.thread_view(t).slots[3]).unwrap().cap.ty, T::Frame);
    m.ok(S::ThreadConfigure, [FREE, 0, KEEP, KEEP, KEEP, 0]);
    assert!(m.k.view(m.k.thread_view(t).slots[0]).is_none());
    // A thread without a CapSpace gets BadCap for everything.
    m.ok(S::ThreadStart, [FREE, 0x40_0000, 0, 0, 1, 0]);
    let r = m.sys_as(t, S::Send, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(Error::result(r.status), Err(Error::BadCap));
}

#[test]
fn elf_reader_rejects_malformed_images() {
    use kernel_core::elf::ElfError as E;
    let good = tiny_elf(&[1, 2, 3], 10);
    let e = Elf::parse(&good, 62).unwrap();
    assert_eq!(e.span(), Some((0x40_0000, 0x60_1000)));
    assert_eq!(e.segments().count(), 2);
    assert_eq!(Elf::parse(&good, 183).err(), Some(E::WrongMachine));
    assert_eq!(Elf::parse(&good[..40], 62).err(), Some(E::NotElf));
    let mut bad = good.clone();
    bad[64 + 32..64 + 40].copy_from_slice(&1000u64.to_le_bytes()); // filesz > memsz
    assert_eq!(Elf::parse(&bad, 62).err(), Some(E::BadSegment));
    let mut bad = good.clone();
    bad[64 + 8..64 + 16].copy_from_slice(&u64::MAX.to_le_bytes()); // offset
    assert_eq!(Elf::parse(&bad, 62).err(), Some(E::Truncated));
    let mut bad = good.clone();
    bad[56..58].copy_from_slice(&500u16.to_le_bytes()); // phnum
    assert_eq!(Elf::parse(&bad, 62).err(), Some(E::Truncated));
    // No prefix of a valid file makes the reader panic.
    for n in 0..good.len() {
        let _ = Elf::parse(&good[..n], 62).map(|e| (e.span(), e.segments().count(), e.relative_relocs().count()));
    }
}

#[test]
fn command_line_lookup() {
    let c = kernel_core::boot::Cmdline(b"test=boot.x  quiet panic=1\n");
    assert_eq!(c.get("test"), Some(&b"boot.x"[..]));
    assert_eq!(c.get("quiet"), Some(&b""[..]));
    assert_eq!(c.get("panic"), Some(&b"1"[..]));
    assert_eq!(c.get("tes"), None);
    assert_eq!(kernel_core::boot::Cmdline(b"").get("x"), None);
}

#[test]
fn destroying_a_capspace_deletes_capabilities_derived_from_its_contents() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::CapSpace, 3, FREE, 1);
    m.retype(ut, T::Notification, 0, FREE + 1, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, FREE + 1, 1, 0, 0]);
    m.ok(S::CapMove, [FREE, 1, FREE + 1, 0, 0, 0]);
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert!(m.cap(FREE + 2).is_none(), "a capability outlived the one it was derived from");
}

/// The fuzzer's interpreter on fixed pseudo-random inputs, so every test run
/// (and Miri) exercises it; `tools/fuzz.sh` runs it coverage-guided.
#[test]
fn random_syscall_sequences_keep_the_kernel_invariants() {
    random_syscall_sequences(0x5EED);
}

// The same in three tests of their own, so they run at once (Miri runs one
// test at a time per process).
#[test]
fn random_syscall_sequences_keep_the_kernel_invariants_2() {
    random_syscall_sequences(0x5EED_0002);
}

#[test]
fn random_syscall_sequences_keep_the_kernel_invariants_3() {
    random_syscall_sequences(0x5EED_0003);
}

fn random_syscall_sequences(seed: u64) {
    let mut r = Rng(seed);
    for _ in 0..if cfg!(miri) { 1 } else { 100 } {
        let len = 16 + r.below(if cfg!(miri) { 200 } else { 3000 }) as usize;
        let data: Vec<u8> = (0..len).map(|_| r.next() as u8).collect();
        support::fuzz::run(&data);
    }
}

/// The timer is programmed lazily while threads run, but exactly before
/// the processor idles: with every thread blocked for ever it is off, so an
/// idle system makes no wakeups. The host machine panics when asked to idle
/// with no timer armed, which is the proof here.
#[test]
#[should_panic(expected = "idle with no timer armed")]
fn an_idle_system_has_no_timer_armed() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    let t = m.spawn(ut, FREE + 1);
    // Two runnable threads: a slice timer is armed.
    assert!(m.k.hal.timer() != u64::MAX);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(!m.k.can_run());
    m.k.schedule();
}

#[test]
fn a_call_hands_the_processor_to_the_receiver_and_the_reply_hands_it_back() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    // A third runnable thread that must not get in between.
    let other = m.spawn(ut, FREE + 6);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    while m.k.current() != m.root {
        m.raw(S::Yield as u64, [0; 6]);
    }
    let switches = m.k.hal.switches.get();
    assert!(m.raw(S::Call as u64, [FREE, 0, 1, 0, 0, 0]).blocked);
    assert_eq!(m.k.current(), t, "the receiver runs at once, ahead of the queue");
    assert!(m.raw(S::ReplyRecv as u64, [FREE, 0, 2, 0, 0, 0]).blocked);
    assert_eq!(m.k.current(), m.root, "the caller runs at once after the reply");
    assert_eq!(m.k.hal.switches.get(), switches + 2);
    assert_eq!(m.k.thread_view(other).state, RUNNABLE);
    assert_eq!(m.arg(m.root, 2), 2);
}

/// Found by the `elf` fuzz target: a dynamic segment whose file offset is
/// near the top of the address range made the relocation reader overflow.
#[test]
fn elf_reader_survives_a_dynamic_segment_at_the_end_of_the_address_space() {
    let mut f = tiny_elf(&[1, 2, 3], 10);
    // Turn the second program header into PT_DYNAMIC with a wild offset and size.
    let o = 64 + 56;
    f[o..o + 4].copy_from_slice(&2u32.to_le_bytes());
    f[o + 8..o + 16].copy_from_slice(&(u64::MAX - 7).to_le_bytes());
    f[o + 32..o + 40].copy_from_slice(&u64::MAX.to_le_bytes());
    let e = Elf::parse(&f, 62).unwrap();
    assert_eq!(e.relative_relocs().count(), 0);
    let crash = include_bytes!("../fuzz/corpus/elf/dynamic-offset-overflow");
    for machine in [62, 183, 243] {
        if let Ok(e) = Elf::parse(crash, machine) {
            let _ = e.relative_relocs().count();
        }
    }
}

#[test]
fn a_fault_reply_can_map_the_missing_page_and_resume_in_one_call() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.retype(ut, T::Frame, 12, FREE + 1, 2);
    let t = m.spawn(ut, FREE + 3);
    m.ok(S::ThreadConfigure, [FREE + 3, KEEP, KEEP, FREE, KEEP, 0]);
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.k.fault(Fault::Page, 0x5000_0123, 6);
    m.k.schedule();
    m.ok(S::Recv, [FREE, 0, 0, 0, 0, 0]);
    let asp = m.cap(SLOT_ASPACE).unwrap().obj;
    // A mapping that fails leaves the fault pending and tells the pager why.
    assert_eq!(m.err(S::ReplyRecv, [0, 0, silica_abi::FAULT_MAP, SLOT_ASPACE, FREE + 1, 0x5000_0000 | 7]), Error::BadArg);
    assert_eq!(m.err(S::ReplyRecv, [0, 0, silica_abi::FAULT_MAP, SLOT_THREAD, FREE + 1, 0x5000_0000 | 3]), Error::WrongType);
    assert_eq!(m.k.thread_view(t).state, REPLY);
    assert_eq!(m.k.thread_view(m.root).caller, t);
    m.ok(S::ReplyRecv, [0, 0, silica_abi::FAULT_MAP, SLOT_ASPACE, FREE + 1, 0x5000_0000 | 3]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    assert_eq!(m.k.translate(asp, 0x5000_0123), Some(m.cap(FREE + 1).unwrap().obj + 0x123));
    assert_eq!(m.err(S::ReplyRecv, [0; 6]), Error::NoCaller);
}

// ---- more than one processor ------------------------------------------------
//
// The host machine's processors take turns (`Machine::on`), so these tests
// check what each processor does and asks of the others; that the locks make
// real concurrency equivalent to some such order is checked by the loom
// model of the lock protocol and by the in-OS tests.

/// Root on processor 0; a thread `t` moved to processor 1 and running
/// there; an endpoint at FREE with a badged copy at FREE + 1.
fn two_processors() -> (Machine, u64) {
    let m = Machine::with_cpus(16, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 77, 0]);
    let t = m.spawn(ut, FREE + 2);
    m.ok(S::ThreadControl, [FREE + 2, 2, 1, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE, "a moved runnable thread stays runnable");
    assert_eq!(m.k.hal.kicks.borrow().last(), Some(&(1, false)), "the other processor is told it has work");
    m.settle(1);
    assert_eq!(m.k.current(), t, "the thread runs on processor 1");
    m.on(0);
    assert_eq!(m.k.current(), m.root);
    (m, t)
}

#[test]
fn a_thread_moved_while_it_waits_for_a_reply_is_answered_on_its_new_processor() {
    let (m, t) = two_processors();
    // t, on processor 1, calls the root on processor 0; the call is delivered.
    m.on(0);
    assert!(m.raw(S::Recv as u64, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.on(1);
    assert!(m.raw(S::Call as u64, [FREE + 1, MsgInfo::new(3, 1).0, 10, 0, 0, 0]).blocked);
    m.settle(0);
    assert_eq!((m.k.current(), m.arg(m.root, 2)), (m.root, 10));
    // Moved while it waits: still waiting, now on processor 0.
    m.ok(S::ThreadControl, [FREE + 2, 2, 0, 0, 0, 0]);
    let v = m.k.thread_view(t);
    assert_eq!((v.state, v.cpu), (REPLY, 0), "the call survives the move");
    // The answer reaches it there.
    assert!(m.raw(S::ReplyRecv as u64, [FREE, MsgInfo::new(0, 1).0, 11, 0, 0, 0]).blocked);
    m.settle(0);
    assert_eq!(m.k.current(), t);
    assert_eq!((m.status(t), m.arg(t, 2)), (0, 11));
    assert_eq!(m.k.check_runqueues(), Ok(()));
}

#[test]
fn a_thread_moved_while_it_waits_to_receive_makes_its_receive_again_there() {
    let (m, t) = two_processors();
    m.on(1);
    assert!(m.raw(S::Recv as u64, [FREE, 0, 0, 0, 0, 0]).blocked);
    let ip = m.k.regs(t)[0].get();
    m.on(0);
    m.ok(S::ThreadControl, [FREE + 2, 2, 0, 0, 0, 0]);
    let v = m.k.thread_view(t);
    assert_eq!((v.state, v.cpu, v.blocked_on), (RUNNABLE, 0, 0), "off the endpoint, runnable on its new processor");
    assert_eq!(m.k.regs(t)[0].get(), ip - 2, "set to make the same system call again");
    assert_eq!(m.k.check_runqueues(), Ok(()));
    // Nothing was delivered to it: a call now finds no receiver.
    assert_eq!(m.err(S::Send, [(FREE + 1) | NONBLOCK, MsgInfo::new(1, 0).0, 0, 0, 0, 0]), Error::WouldBlock);
}

#[test]
fn a_server_stopped_in_reply_recv_after_replying_only_receives_when_it_runs_again() {
    let (m, t) = two_processors();
    m.on(1);
    assert!(m.raw(S::Recv as u64, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    assert!(m.raw(S::Call as u64, [FREE + 1, MsgInfo::new(3, 1).0, 10, 0, 0, 0]).blocked);
    m.settle(1);
    assert_eq!(m.k.current(), t);
    // It answers and waits for the next call.
    assert!(m.raw(S::ReplyRecv as u64, [FREE, MsgInfo::new(0, 1).0, 11, 0, 0, 0]).blocked);
    let ip = m.k.regs(t)[0].get();
    m.settle(0);
    assert_eq!((m.k.current(), m.status(m.root), m.arg(m.root, 2)), (m.root, 0, 11));
    // Stopped there (here: moved), the reply is not made again: what runs
    // again is the receive.
    m.ok(S::ThreadControl, [FREE + 2, 2, 0, 0, 0, 0]);
    let r = m.k.regs(t);
    assert_eq!(r[0].get(), ip - 2);
    assert_eq!((r[3].get(), m.arg(t, 0), m.arg(t, 1), m.arg(t, 2)), (S::Recv as u64, FREE, 0, 0));
}

#[test]
fn a_signal_recv_made_again_only_receives() {
    let (m, t) = two_processors();
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE + 3, 1);
    // t, on processor 1, signals the notification and waits on the endpoint.
    m.on(1);
    assert!(m.raw(S::SignalRecv as u64, [FREE, 0, 0, FREE + 3, 0, 0]).blocked);
    let ip = m.k.regs(t)[0].get();
    m.on(0);
    m.ok(S::Recv, [FREE + 3, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1, "signalled once");
    // Moved while it waits, its call is made again: as a receive alone.
    m.ok(S::ThreadControl, [FREE + 2, 2, 0, 0, 0, 0]);
    let r = m.k.regs(t);
    assert_eq!((r[0].get(), r[3].get(), m.arg(t, 0)), (ip - 2, S::Recv as u64, FREE));
    assert_eq!(m.arg(t, 3), FREE + 3, "the notification is still in its registers");
    // Running again, the call it makes from its registers waits on the
    // endpoint and signals nothing.
    for _ in 0..1000 {
        if m.k.current() == t {
            break;
        }
        m.k.regs(m.k.current())[TestHal::SYS].set(S::Yield as u64);
        m.k.enter();
        m.k.syscall();
        m.k.schedule();
        m.k.leave();
    }
    assert_eq!(m.k.current(), t);
    m.k.enter();
    m.k.syscall();
    m.k.leave();
    assert_ne!(m.k.thread_view(t).blocked_on, 0, "waiting on the endpoint again");
    m.on(0);
    assert_eq!(m.err(S::Recv, [FREE + 3, 1, 0, 0, 0, 0]), Error::WouldBlock, "not signalled a second time");
}

#[test]
fn a_system_call_overtaken_while_it_waits_for_a_lock_is_made_again() {
    // Thread t, on processor 1, has entered a system call and waits for its
    // lock, which processor 0 holds to stop it: the call has not happened.
    let (m, t) = two_processors();
    let r = m.k.regs(t);
    r[0].set(0x40_0002);
    r[3].set(S::Recv as u64);
    r[10].set(1);
    m.on(0);
    m.ok(S::ThreadControl, [FREE + 2, 0, 0, 0, 0, 0]);
    assert_eq!((r[0].get(), r[3].get()), (0x40_0000, S::Recv as u64), "set to make the call again");
    // Interrupted in user mode instead, it goes on from where it was.
    m.ok(S::ThreadControl, [FREE + 2, 1, 0, 0, 0, 0]);
    m.settle(1);
    assert_eq!(m.k.current(), t);
    r[0].set(0x40_1234);
    r[10].set(0);
    m.on(0);
    m.ok(S::ThreadControl, [FREE + 2, 0, 0, 0, 0, 0]);
    assert_eq!(r[0].get(), 0x40_1234);
}

#[test]
fn a_suspended_thread_moved_stays_suspended_and_a_thread_may_move_itself() {
    let (m, t) = two_processors();
    m.on(0);
    m.ok(S::ThreadControl, [FREE + 2, 0, 0, 0, 0, 0]);
    m.ok(S::ThreadControl, [FREE + 2, 2, 0, 0, 0, 0]);
    let v = m.k.thread_view(t);
    assert_eq!((v.state, v.cpu), (INACTIVE, 0));
    // The root moves itself to processor 1 and carries on there.
    let r = m.raw(S::ThreadControl as u64, [SLOT_THREAD, 2, 1, 0, 0, 0]);
    assert_eq!(r.status, 0);
    assert_ne!(m.k.current(), m.root, "processor 0 no longer runs it");
    let v = m.k.thread_view(m.root);
    assert_eq!((v.state, v.cpu), (RUNNABLE, 1));
    m.settle(1);
    assert_eq!((m.k.current(), m.status(m.root)), (m.root, 0));
    assert_eq!(m.k.check_runqueues(), Ok(()));
}

#[test]
fn a_call_across_processors_wakes_the_server_there_and_the_reply_comes_back() {
    let (m, t) = two_processors();
    m.on(1);
    assert!(m.raw(S::Recv as u64, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    let kicks = m.k.hal.kicks.borrow().len();
    let (globals, pairs) = (m.k.global_entries(), m.k.pair_entries());
    assert!(m.raw(S::Call as u64, [FREE + 1, MsgInfo::new(3, 2).0, 10, 20, 0, 0]).blocked);
    assert_eq!(m.k.hal.kicks.borrow()[kicks..], [(1, false)], "processor 1 is kicked to run the server");
    assert_eq!(m.k.thread_view(m.root).state, REPLY);
    m.settle(1);
    assert_eq!(m.k.current(), t);
    assert_eq!((m.arg(t, 0), m.arg(t, 2), m.arg(t, 3)), (77, 10, 20));
    assert!(m.raw(S::ReplyRecv as u64, [FREE, MsgInfo::new(4, 1).0, 30, 0, 0, 0]).blocked);
    m.settle(0);
    assert_eq!(m.k.current(), m.root);
    assert_eq!((m.status(m.root), m.arg(m.root, 2)), (0, 30));
    // The call and the reply each took the two processors' locks, and
    // neither stopped the rest of the machine.
    assert_eq!((m.k.global_entries(), m.k.pair_entries()), (globals, pairs + 2));
}

#[test]
fn a_receive_timing_out_on_a_queue_another_processor_owns_takes_every_lock() {
    let (m, t) = two_processors();
    // A second thread on each processor, so neither idles (which, here,
    // would move the clock to the next deadline).
    let (ut, _) = m.big_untyped();
    m.spawn(ut, FREE + 3);
    m.spawn(ut, FREE + 4);
    m.ok(S::ThreadControl, [FREE + 4, 2, 1, 0, 0, 0]);
    m.ok(S::ThreadControl, [FREE + 4, 1, 0, 0, 0, 0]);
    m.settle(1);
    assert_eq!(m.k.current(), t);
    m.on(0);
    assert_eq!(m.k.current(), m.root);
    let at = m.k.hal.now.get() + 1_000_000;
    // Processor 1's thread waits first, then the root on processor 0: the
    // queue now belongs to processor 0 and holds a thread of each.
    m.on(1);
    assert!(m.raw(S::Recv as u64, [FREE, at, 0, 0, 0, 0]).blocked);
    m.on(0);
    assert!(m.raw(S::Recv as u64, [FREE, at + 9_000_000, 0, 0, 0, 0]).blocked);
    let globals = m.k.global_entries();
    // Processor 1's deadline passes. Its waiter is unlinked from a queue
    // whose links are processor 0's, so not under processor 1's lock alone.
    m.k.hal.now.set(at);
    m.on(1);
    m.tick();
    assert_eq!(m.k.global_entries(), globals + 1);
    assert_eq!((m.k.thread_view(t).state, Error::result(m.status(t))), (RUNNABLE, Err(Error::Timeout)));
    // The queue is whole: a message sent now reaches the root.
    m.sys_as(t, S::Yield, [0; 6]);
    assert!(!m.raw(S::Send as u64, [FREE + 1, MsgInfo::new(1, 1).0, 42, 0, 0, 0]).blocked);
    assert_eq!((m.k.thread_view(m.root).state, m.status(m.root), m.arg(m.root, 0), m.arg(m.root, 2)), (RUNNABLE, 0, 77, 42));
}

#[test]
fn a_call_is_parked_on_a_notification_last_used_from_a_third_processor() {
    // Root on processor 0, servers `a` on processor 1 and `b` on processor 2,
    // an endpoint at FREE (badged copy at FREE + 1), a notification at FREE + 9.
    let m = Machine::with_cpus(16, 3);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 77, 0]);
    m.retype(ut, T::Notification, 0, FREE + 9, 1);
    let place = |slot: u64, cpu: u64| {
        let t = m.spawn(ut, slot);
        m.ok(S::ThreadControl, [slot, 2, cpu, 0, 0, 0]);
        m.ok(S::ThreadControl, [slot, 1, 0, 0, 0, 0]);
        m.settle(cpu as usize);
        assert_eq!(m.k.current(), t);
        m.on(0);
        t
    };
    let (a, b) = (place(FREE + 2, 1), place(FREE + 3, 2));
    let n = m.cap(FREE + 9).unwrap().obj;
    // `b` takes a call parked on the notification and answers it: the
    // notification is now processor 2's.
    m.on(2);
    assert!(m.raw(S::Recv as u64, [FREE, 0, FREE + 9, 0, 0, 0]).blocked);
    m.on(0);
    assert!(m.raw(S::Call as u64, [FREE + 1, MsgInfo::new(0, 1).0, 1, 0, 0, 0]).blocked);
    assert_eq!(m.k.thread_view(m.root).blocked_on, n);
    m.settle(2);
    assert_eq!(m.k.current(), b);
    assert!(!m.raw(S::Send as u64, [FREE + 9, MsgInfo::new(0, 1).0, 2, 0, 0, 0]).blocked);
    m.settle(0);
    assert_eq!((m.k.current(), m.status(m.root), m.arg(m.root, 2)), (m.root, 0, 2));
    // Now `a` on processor 1 asks for its next call to be parked there. The
    // call from processor 0 is parked, although neither processor owned
    // the notification, and `a` answers it.
    m.on(1);
    assert!(m.raw(S::Recv as u64, [FREE, 0, FREE + 9, 0, 0, 0]).blocked);
    m.on(0);
    assert!(m.raw(S::Call as u64, [FREE + 1, MsgInfo::new(0, 1).0, 3, 0, 0, 0]).blocked);
    assert_eq!(m.k.thread_view(m.root).blocked_on, n, "the call is parked");
    assert_eq!(m.k.thread_view(a).caller, 0);
    m.settle(1);
    assert_eq!((m.k.current(), m.arg(a, 2)), (a, 3));
    assert!(!m.raw(S::Send as u64, [FREE + 9, MsgInfo::new(0, 1).0, 4, 0, 0, 0]).blocked);
    m.settle(0);
    assert_eq!((m.k.current(), m.status(m.root), m.arg(m.root, 2)), (m.root, 0, 4));
}

#[test]
fn calls_between_threads_of_one_processor_take_only_that_processors_lock() {
    let (m, t) = two_processors();
    let (ut, _) = m.big_untyped();
    // A second endpoint served and called on processor 1 only.
    m.retype(ut, T::Endpoint, 0, FREE + 5, 1);
    let c = m.spawn(ut, FREE + 6);
    m.ok(S::ThreadControl, [FREE + 6, 2, 1, 0, 0, 0]);
    m.ok(S::ThreadControl, [FREE + 6, 1, 0, 0, 0, 0]);
    m.on(1);
    // The first wait gives the endpoint to processor 1 (a global step).
    assert!(m.sys_as(t, S::Recv, [FREE + 5, 0, 0, 0, 0, 0]).blocked);
    let before = m.k.global_entries();
    for i in 0..100 {
        assert!(m.sys_as(c, S::Call, [FREE + 5, 0, i, 0, 0, 0]).blocked);
        assert_eq!(m.k.current(), t);
        assert!(m.raw(S::ReplyRecv as u64, [FREE + 5, 0, i + 1, 0, 0, 0]).blocked);
        assert_eq!((m.k.current(), m.arg(c, 2)), (c, i + 1));
    }
    assert_eq!(m.k.global_entries(), before, "a call and its reply on one processor never take every lock");
    // The same call from processor 0 takes that processor's lock and
    // processor 1's, and still not every lock.
    let pairs = m.k.pair_entries();
    m.on(0);
    assert!(m.raw(S::Call as u64, [FREE + 5, 0, 5, 0, 0, 0]).blocked);
    assert_eq!((m.k.global_entries(), m.k.pair_entries()), (before, pairs + 1));
}

#[test]
fn unmapping_brings_a_processor_running_in_that_address_space_into_the_kernel() {
    let (m, _t) = two_processors();
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Frame, 12, FREE + 5, 1);
    m.ok(S::Map, [SLOT_ASPACE, FREE + 5, 0x3000_0000, 3, 0, 0]);
    let kicks = m.k.hal.kicks.borrow().len();
    m.ok(S::Unmap, [FREE + 5, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.kicks.borrow()[kicks..], [(1, true)], "processor 1 runs a thread in the root address space");
    assert_eq!(m.k.hal.resets.get(), 0);
    m.settle(1);
    assert_eq!(m.k.hal.resets.get(), 1, "it drops its cached translations before running user code again");
    // An address space loaded nowhere else needs no interrupt, only forgetting.
    m.on(0);
    m.retype(ut, T::AddressSpace, 15, FREE + 6, 1);
    m.ok(S::Map, [FREE + 6, FREE + 5, 0x3000_0000, 3, 0, 0]);
    let (kicks, forgets) = (m.k.hal.kicks.borrow().len(), m.k.hal.forgets.get());
    m.ok(S::Unmap, [FREE + 5, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.kicks.borrow().len(), kicks);
    assert_eq!(m.k.hal.forgets.get(), forgets + 1);
}

#[test]
fn destroying_or_stopping_a_thread_running_on_another_processor_waits_for_it() {
    let (m, t) = two_processors();
    let kicks = m.k.hal.kicks.borrow().len();
    m.ok(S::ThreadControl, [FREE + 2, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.kicks.borrow()[kicks..], [(1, true)], "its registers must be final before the call returns");
    assert_eq!(m.k.thread_view(t).state, INACTIVE);
    m.ok(S::ThreadControl, [FREE + 2, 1, 0, 0, 0, 0]);
    m.settle(1);
    assert_eq!(m.k.current(), t);
    m.on(0);
    m.ok(S::CapDelete, [FREE + 2, 0, 0, 0, 0, 0]);
    assert!(m.k.hal.kicks.borrow()[kicks..].contains(&(1, true)));
    // Processor 1 enters the kernel (the kick) and finds its thread gone:
    // the syscall it might have been about to make is dropped, not run.
    m.on(1);
    assert_eq!(m.k.current(), 0);
    assert!(!m.k.can_run());
}

#[test]
fn a_thread_stopped_while_its_processor_waited_for_the_lock_does_not_run_its_syscall() {
    let (m, t) = two_processors();
    // Processor 1's thread has "made" a syscall (registers saved), but
    // processor 0 gets in first and stops it.
    m.k.regs(t)[TestHal::SYS].set(S::CapDelete as u64);
    m.k.regs(t)[TestHal::ARG[0]].set(FREE);
    m.ok(S::ThreadControl, [FREE + 2, 0, 0, 0, 0, 0]);
    m.on(1);
    m.k.enter();
    m.k.syscall();
    m.k.leave();
    assert!(m.cap(FREE).is_some(), "the stopped thread's syscall must not take effect");
}

// ---- capabilities in messages, calls in progress, parked calls -------------

/// Root and a second thread, each with an IPC buffer; an endpoint at FREE
/// (all rights) and a copy without GRANT at FREE + 1. Returns the machine,
/// the thread, and the two buffers' physical addresses (root's, thread's).
fn pair_with_buffers() -> (Machine, u64, u64, u64) {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 3, 0, 0]);
    let t = m.spawn(ut, FREE + 2);
    m.retype(ut, T::Frame, 12, FREE + 3, 1);
    m.ok(S::ThreadConfigure, [FREE + 2, KEEP, KEEP, KEEP, FREE + 3, 0]);
    let (rb, tb) = (m.cap(SLOT_IPCBUF).unwrap().obj, m.cap(FREE + 3).unwrap().obj);
    (m, t, rb, tb)
}

fn poke(m: &Machine, buffer: u64, offset: usize, index: usize, v: u64) {
    m.k.phys.write_val(buffer + (offset + index * 8) as u64, &v);
}

#[test]
fn a_message_passes_capabilities_derived_duplicated_or_moved() {
    use silica_abi::{IPC_CAP_DUPLICATE, IPC_CAP_MOVE, IPC_CAPS, IPC_RECV};
    let (m, t, rb, tb) = pair_with_buffers();
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE + 10, 3);
    // The receiver (sharing the CapSpace) wants what it is sent in slots 40, 41, 42.
    for i in 0..3 {
        poke(&m, tb, IPC_RECV, i, FREE + 40 + i as u64);
    }
    poke(&m, rb, IPC_CAPS, 0, FREE + 10);
    poke(&m, rb, IPC_CAPS, 1, (FREE + 11) | (IPC_CAP_DUPLICATE << 60));
    poke(&m, rb, IPC_CAPS, 2, (FREE + 12) | (IPC_CAP_MOVE << 60));
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    // Without the GRANT right the message arrives and the capabilities do not.
    m.ok(S::Send, [FREE + 1, MsgInfo::with_caps(1, 1, 3).0, 7, 0, 0, 0]);
    assert_eq!(MsgInfo(m.arg(t, 1)).caps(), 0);
    assert!(m.cap(FREE + 40).is_none());
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Send, [FREE, MsgInfo::with_caps(1, 1, 3).0, 7, 0, 0, 0]);
    assert_eq!((MsgInfo(m.arg(t, 1)).caps(), m.arg(t, 2)), (3, 7));
    let slot = |i| m.k.view(m.k.slot_addr(&m.cs, i).unwrap());
    assert_eq!(slot(FREE + 40).unwrap().depth, slot(FREE + 10).unwrap().depth + 1, "derived: a child of the sender's");
    assert_eq!(slot(FREE + 41).unwrap().depth, slot(FREE + 11).unwrap().depth, "duplicated: beside the sender's");
    assert!(slot(FREE + 12).is_none() && slot(FREE + 42).is_some(), "moved: the sender no longer has it");
    // The sender takes back what it lent, and only that.
    m.ok(S::CapRevoke, [FREE + 10, 0, 0, 0, 0, 0]);
    assert!(slot(FREE + 40).is_none() && slot(FREE + 41).is_some());
    // A full receiving slot ends the transfer there; the count says so.
    poke(&m, rb, IPC_CAPS, 0, FREE + 10);
    poke(&m, rb, IPC_CAPS, 1, FREE + 11);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Send, [FREE, MsgInfo::with_caps(1, 0, 2).0, 0, 0, 0, 0]);
    assert_eq!(MsgInfo(m.arg(t, 1)).caps(), 1, "slot 41 is occupied");
    assert!(slot(FREE + 40).is_some());
    // A reply may carry capabilities whatever the caller's rights were.
    m.ok(S::CapDelete, [FREE + 40, 0, 0, 0, 0, 0]);
    poke(&m, rb, IPC_RECV, 0, FREE + 50);
    poke(&m, tb, IPC_CAPS, 0, FREE + 42);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 0, 0, 0, 0]).blocked);
    m.sys_as(t, S::ReplyRecv, [0, MsgInfo::with_caps(2, 0, 1).0, 0, 0, 0, 0]);
    assert_eq!(MsgInfo(m.arg(m.root, 1)).caps(), 1);
    assert!(slot(FREE + 50).is_some());
}

#[test]
fn destroying_an_endpoint_fails_the_calls_in_progress_through_it_on_both_sides() {
    let (m, t) = pair();
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 1, 0, 0, 0]).blocked);
    // The server holds the call. The endpoint goes (the server's own
    // capability is in the shared CapSpace, so it can delete it).
    assert_eq!(m.k.thread_view(t).caller, m.root);
    let r = m.sys_as(t, S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(r.status, 0);
    assert_eq!(Error::result(m.status(m.root)), Err(Error::Gone), "the caller's call fails");
    assert_eq!(m.k.thread_view(m.root).state, RUNNABLE);
    assert_eq!(m.k.thread_view(t).caller, 0);
    let r = m.sys_as(t, S::ReplyRecv, [0, 0, 2, 0, 0, 0]);
    assert_eq!(Error::result(r.status), Err(Error::NoCaller), "the server's reply finds no caller");
}

#[test]
fn a_parked_call_is_answered_later_through_its_notification() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    let t2 = m.spawn(ut, FREE + 5);
    m.retype(ut, T::Notification, 0, FREE + 10, 2);
    // The server takes two calls, parking each, before answering either.
    assert!(m.sys_as(t, S::Recv, [FREE, 0, FREE + 10, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 100, 0, 0, 0]).blocked);
    assert_eq!(m.k.thread_view(t).caller, 0, "the call is parked, not held by the server thread");
    assert_eq!(m.arg(t, 2), 100);
    let r = m.sys_as(t, S::Recv, [FREE, 0, FREE + 11, 0, 0, 0]);
    assert!(r.blocked);
    assert!(m.sys_as(t2, S::Call, [FREE + 1, 0, 200, 0, 0, 0]).blocked);
    assert_eq!(m.arg(t, 2), 200);
    // While a call is parked the notification cannot be waited on, and a
    // signal to it does not answer the call.
    let r = m.sys_as(t, S::Recv, [FREE + 10, 1, 0, 0, 0, 0]);
    assert_eq!(Error::result(r.status), Err(Error::InUse));
    // Answers in the other order.
    m.sys_as(t, S::Send, [FREE + 11, MsgInfo::new(9, 1).0, 201, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t2).state, m.status(t2), m.arg(t2, 2)), (RUNNABLE, 0, 201));
    assert_eq!(m.k.thread_view(m.root).state, REPLY);
    m.sys_as(t, S::Send, [FREE + 10, MsgInfo::new(9, 1).0, 101, 0, 0, 0]);
    assert_eq!((m.k.thread_view(m.root).state, m.status(m.root), m.arg(m.root, 2)), (RUNNABLE, 0, 101));
    // The notification is an ordinary notification again.
    m.ok(S::Send, [FREE + 10, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE + 10, 1, 0, 0, 0, 0]);
    // Destroying the notification a call is parked on fails the call.
    assert!(m.sys_as(t, S::Recv, [FREE, 0, FREE + 10, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 1, 0, 0, 0]).blocked);
    let r = m.sys_as(t, S::CapDelete, [FREE + 10, 0, 0, 0, 0, 0]);
    assert_eq!(r.status, 0);
    assert_eq!(Error::result(m.status(m.root)), Err(Error::Gone));
    // And a parked caller that is destroyed leaves the notification free,
    // which the server can see (Inspect) before it answers.
    m.ok(S::Inspect, [FREE + 11, 0, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0, "nothing parked yet");
    assert!(m.sys_as(t, S::Recv, [FREE, 0, FREE + 11, 0, 0, 0]).blocked);
    assert!(m.sys_as(t2, S::Call, [FREE + 1, 0, 1, 0, 0, 0]).blocked);
    m.ok(S::Inspect, [FREE + 11, 0, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1, "the caller waits");
    m.ok(S::CapDelete, [FREE + 5, 0, 0, 0, 0, 0]);
    m.ok(S::Inspect, [FREE + 11, 0, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0, "the caller has gone");
    m.ok(S::Send, [FREE + 11, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE + 11, 1, 0, 0, 0, 0]);
}

#[test]
fn an_interrupt_line_signals_the_notification_it_is_bound_to() {
    use silica_abi::IRQ_LEVEL;
    let m = Machine::with_cpus(16, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 0b100, 0]);
    // Only lines the capability covers, and only lines the machine has.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, SLOT_IRQS, 15, 4 | 6 << 32, 0]);
    assert_eq!(m.err(S::IrqBind, [FREE + 2, 6, FREE + 1, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::IrqBind, [SLOT_PORTS, 4, FREE + 1, 0, 0, 0]), Error::WrongType);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 30, FREE + 1, 0, 0, 0]), Error::Unsupported);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 400, FREE + 1, 0, 0, 0]), Error::BadArg);
    m.ok(S::IrqBind, [FREE + 2, 5, FREE + 1, IRQ_LEVEL, 0, 0]);
    assert_eq!(m.k.hal.irq_log.borrow().last().unwrap(), "route 5 cpu0 level=true low=false");
    // The interrupt is delivered as the capability's badge, to a waiter or pending.
    m.irq(5);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0b100);
    m.ok(S::IrqAck, [FREE + 2, 5, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.irq_log.borrow().last().unwrap(), "unmask 5");
    // A waiter on another processor is woken there.
    let t = m.spawn(ut, FREE + 3);
    m.ok(S::ThreadControl, [FREE + 3, 2, 1, 0, 0, 0]);
    m.ok(S::ThreadControl, [FREE + 3, 1, 0, 0, 0, 0]);
    m.settle(1);
    assert!(m.raw(S::Recv as u64, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    m.irq(5);
    assert_eq!((m.k.thread_view(t).state, m.arg(t, 0)), (RUNNABLE, 0b100));
    // An unbound line is held off; a destroyed notification unbinds its lines.
    m.irq(7);
    assert_eq!(m.k.hal.irq_log.borrow().last().unwrap(), "mask 7");
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.irq_log.borrow().last().unwrap(), "mask 5");
    m.irq(5);
    assert_eq!(m.k.hal.irq_log.borrow().last().unwrap(), "mask 5");
}

#[test]
fn inspect_shows_slots_and_threads_without_changing_them() {
    let (m, t) = pair();
    m.ok(S::Inspect, [SLOT_CAPSPACE, FREE + 1, 0, 0, 0, 0]);
    let ep = m.cap(FREE).unwrap();
    assert_eq!(
        [m.arg(m.root, 0), m.arg(m.root, 1), m.arg(m.root, 3), m.arg(m.root, 4), m.arg(m.root, 5)],
        [T::Endpoint as u64, 15, ep.obj, 77, 2]
    );
    m.ok(S::Inspect, [SLOT_CAPSPACE, FREE + 30, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0, "an empty slot");
    assert_eq!(m.err(S::Inspect, [SLOT_CAPSPACE, 5000, 0, 0, 0, 0]), Error::BadCap);
    assert_eq!(m.err(S::Inspect, [FREE, 0, 0, 0, 0, 0]), Error::WrongType);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Inspect, [FREE + 2, 0, 0, 0, 0, 0]);
    assert_eq!([m.arg(m.root, 0), m.arg(m.root, 1), m.arg(m.root, 4)], [silica_abi::THREAD_RECEIVING, 0, ep.obj]);
}

#[test]
fn revoking_a_capability_fails_what_waits_through_it_and_nothing_else() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    let t2 = m.spawn(ut, FREE + 5);
    let regrant = |m: &Machine| m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 77, 0]);
    // A second client capability, a sibling of the first: its users are untouched.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 20, FREE, 15, 78, 0]);

    // 1. A call still queued (no receiver yet): revoking the caller's
    // capability takes it out of the queue with `Gone`.
    assert!(m.sys(S::Call, [FREE + 1, 0, 1, 0, 0, 0]).blocked);
    assert!(m.sys_as(t2, S::Call, [FREE + 20, 0, 2, 0, 0, 0]).blocked);
    assert_eq!(m.sys_as(t, S::CapRevoke, [FREE, 0, 0, 0, 0, 0]).status, 0);
    for c in [m.root, t2] {
        assert_eq!((m.k.thread_view(c).state, Error::result(m.status(c))), (RUNNABLE, Err(Error::Gone)));
    }
    // The endpoint itself lives on and nothing is left queued on it.
    assert_eq!(Error::result(m.sys_as(t, S::Recv, [FREE, 1, 0, 0, 0, 0]).status), Err(Error::WouldBlock));

    // 2. A call delivered and held by the server: deleting the caller's
    // capability fails the call on both sides. The other caller, queued
    // through a different capability, is delivered next.
    regrant(&m);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 20, FREE, 15, 78, 0]);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 3, 0, 0, 0]).blocked);
    assert!(m.sys_as(t2, S::Call, [FREE + 20, 0, 4, 0, 0, 0]).blocked);
    assert_eq!(m.k.thread_view(t).caller, m.root);
    assert_eq!(m.sys_as(t, S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]).status, 0);
    assert_eq!((m.k.thread_view(m.root).state, Error::result(m.status(m.root))), (RUNNABLE, Err(Error::Gone)));
    assert_eq!(m.k.thread_view(t).caller, 0);
    assert_eq!(Error::result(m.sys_as(t, S::ReplyRecv, [0, 0, 9, 0, 0, 0]).status), Err(Error::NoCaller));
    assert_eq!(m.k.thread_view(t2).state, SENDING, "a call through another capability still waits");
    let r = m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!((r.status, m.arg(t, 0), m.arg(t, 2)), (0, 78, 4));
    m.sys_as(t, S::ReplyRecv, [0, 0, 5, 0, 0, 0]);
    assert_eq!((m.status(t2), m.arg(t2, 2)), (0, 5));

    // 3. The same slot granted again is a working capability again.
    regrant(&m);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 6, 0, 0, 0]).blocked);
    m.sys_as(t, S::ReplyRecv, [0, 0, 7, 0, 0, 0]);
    assert_eq!((m.status(m.root), m.arg(m.root, 2)), (0, 7));

    // 4. A parked call: revoking the capability it came through fails it
    // and frees the notification it was parked on.
    m.retype(ut, T::Notification, 0, FREE + 10, 1);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, FREE + 10, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 8, 0, 0, 0]).blocked);
    assert_eq!(m.sys_as(t, S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]).status, 0);
    assert_eq!((m.k.thread_view(m.root).state, Error::result(m.status(m.root))), (RUNNABLE, Err(Error::Gone)));
    m.ok(S::Send, [FREE + 10, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE + 10, 1, 0, 0, 0, 0]);
    // A parked call also fails when its endpoint is destroyed (step 6).
    regrant(&m);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 30, FREE, 15, 0, 1]);
    m.retype(ut, T::Notification, 0, FREE + 12, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 13, FREE + 12, 15, 0, 0]);
    let t3 = m.spawn(ut, FREE + 40);
    assert!(m.sys_as(t, S::Recv, [FREE, 0, FREE + 10, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, 0, 8, 0, 0, 0]).blocked);

    // 5. A receiver whose capability goes stops waiting; one that waits
    // through another capability to the same endpoint does not.
    assert!(m.sys_as(t, S::Recv, [FREE + 30, 0, 0, 0, 0, 0]).blocked);
    assert!(m.sys_as(t2, S::Recv, [FREE + 13, 0, 0, 0, 0, 0]).blocked);
    assert_eq!(m.sys_as(t3, S::CapDelete, [FREE + 13, 0, 0, 0, 0, 0]).status, 0);
    assert_eq!((m.k.thread_view(t2).state, Error::result(m.status(t2))), (RUNNABLE, Err(Error::Gone)));
    assert_eq!(m.k.thread_view(t).state, RECEIVING);
    assert_eq!(m.sys_as(t3, S::CapDelete, [FREE + 30, 0, 0, 0, 0, 0]).status, 0);
    assert_eq!((m.k.thread_view(t).state, Error::result(m.status(t))), (RUNNABLE, Err(Error::Gone)));

    // 6. Destroying the endpoint (revoking the original capability, then
    // deleting it) fails the parked call and frees the notification.
    assert_eq!(m.k.thread_view(m.root).state, REPLY);
    m.sys_as(t3, S::CapRevoke, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(
        (m.k.thread_view(m.root).state, Error::result(m.status(m.root))),
        (RUNNABLE, Err(Error::Gone)),
        "its capability was revoked"
    );
    m.ok(S::Send, [FREE + 10, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE + 10, 1, 0, 0, 0, 0]);
}

/// Inputs the syscall fuzzer once failed on, kept as seeds and replayed here.
#[test]
fn syscall_sequences_the_fuzzer_found_keep_the_kernel_invariants() {
    support::fuzz::run(include_bytes!("../fuzz/corpus/syscalls/root-moved-to-the-other-processor"));
}

/// Found by the syscall fuzzer: a receive with the deadline `u64::MAX` put
/// its thread on the timer list with the value that means "timer off", so
/// the timer was never set for it and an idle processor spun in the
/// scheduler for ever. That deadline now means "wait for ever", like 0.
#[test]
fn deadline_max_waits_for_ever() {
    support::fuzz::run(include_bytes!("../fuzz/corpus/syscalls/recv-deadline-max-spins-idle"));
}

/// Found by the syscall fuzzer: once the clock reached a deadline just
/// below `u64::MAX`, the end of the next time slice overflowed (in a
/// release build it wrapped to the past and every entry rescheduled).
/// The slice end saturates.
#[test]
fn slice_end_saturates_near_the_end_of_time() {
    support::fuzz::run(include_bytes!("../fuzz/corpus/syscalls/slice-end-overflows-near-time-max"));
}

/// Found by the syscall fuzzer, and a mistake of its own: it went on
/// writing a thread's message words into the frame that had been the
/// thread's IPC buffer after the frame was revoked and its memory made
/// into CapSpaces, which no thread can do (its mapping goes with the
/// frame). The write now goes to the buffer the kernel holds for the
/// thread, if any; the sequence is kept because it revokes buffers and
/// reuses their memory while messages are in flight.
#[test]
fn a_revoked_ipc_buffer_is_not_written_by_its_thread() {
    support::fuzz::run(include_bytes!("../fuzz/corpus/syscalls/buffer-written-after-its-frame-was-revoked"));
}

/// An address space made for devices: frames mapped in it are what the
/// devices attached to it reach, a device reaches nothing before it is
/// attached and nothing once the address space is gone, and removing a
/// mapping drops what the IOMMU cached before the call returns.
#[test]
fn a_device_reaches_exactly_what_is_mapped_in_the_address_space_it_is_attached_to() {
    use silica_abi::ASPACE_IO;
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let (io, frame, large) = (FREE, FREE + 1, FREE + 3);
    // The smallest address space has no room for the table a device one starts with.
    assert_eq!(m.err(S::Retype, [ut, T::AddressSpace as u64, 13 | ASPACE_IO, SLOT_CAPSPACE, io, 1]), Error::BadArg);
    m.retype(ut, T::AddressSpace, 16 | ASPACE_IO, io, 1);
    m.retype(ut, T::Frame, 12, frame, 2);
    m.retype(ut, T::Frame, 21, large, 1);
    let space = m.cap(io).unwrap().obj;
    // Device addresses go up to what the IOMMU translates, not to the top
    // of a program's addresses; rights are read, or read and write.
    assert_eq!(m.err(S::Map, [io, frame, 1 << 48, 3, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::Map, [io, frame, 0x10_0000, 5, 0, 0]), Error::BadArg, "a device executes nothing");
    assert_eq!(m.err(S::Map, [io, large, 0x10_1000, 3, 0, 0]), Error::BadArg, "large frame alignment");
    m.ok(S::Map, [io, frame, 0xFFFF_FFFF_F000, 3, 0, 0]);
    m.ok(S::Unmap, [frame, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.io_flushes.get(), 0, "nothing is cached of an address space no device uses");
    m.ok(S::Map, [io, frame, 0x10_0000, 3, 0, 0]);
    m.ok(S::Map, [io, large, 0x4000_0000, 1, 0, 0]);
    assert_eq!(m.k.translate(space, 0x10_0123), Some(m.cap(frame).unwrap().obj + 0x123));
    assert_eq!(m.k.translate(space, 0x4012_3456), Some(m.cap(large).unwrap().obj + 0x12_3456));
    // No thread runs in it, and a program's address space takes no device.
    assert_eq!(m.err(S::ThreadConfigure, [SLOT_THREAD, KEEP, io, KEEP, KEEP, 0]), Error::BadArg);
    assert_eq!(m.err(S::DeviceAttach, [SLOT_DEVICES, 0x18, SLOT_ASPACE, 0, 0, 0]), Error::WrongType);
    // Attaching takes the authority over that device.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 4, SLOT_DEVICES, 15, 0x18 | 0x19 << 32, 0]);
    assert_eq!(m.err(S::DeviceAttach, [FREE + 4, 0x20, io, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::DeviceAttach, [SLOT_IRQS, 0x18, io, 0, 0, 0]), Error::WrongType);
    assert_eq!(m.err(S::DeviceAttach, [SLOT_DEVICES, 0x1_0018, io, 0, 0, 0]), Error::Unsupported, "a segment the IOMMU does not cover");
    m.ok(S::DeviceAttach, [FREE + 4, 0x18, io, 0, 0, 0]);
    m.ok(S::DeviceAttach, [SLOT_DEVICES, 0x20, io, 0, 0, 0]);
    assert_eq!(m.k.hal.attached.borrow().get(&0x18), Some(&(space + 4096)));
    assert_eq!(m.k.attached_to(space), 2);
    // Removing a mapping reaches the IOMMU before the call returns.
    let flushes = m.k.hal.io_flushes.get();
    m.ok(S::Unmap, [frame, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.io_flushes.get(), flushes + 1);
    assert_eq!(m.k.translate(space, 0x10_0000), None);
    // Moving a device to another address space, and detaching it.
    m.retype(ut, T::AddressSpace, 14 | ASPACE_IO, FREE + 5, 1);
    m.ok(S::DeviceAttach, [SLOT_DEVICES, 0x20, FREE + 5, 0, 0, 0]);
    assert_eq!((m.k.attached_to(space), m.k.attached_to(m.cap(FREE + 5).unwrap().obj)), (1, 1));
    m.ok(S::DeviceAttach, [SLOT_DEVICES, 0x20, 0, 0, 0, 0]);
    assert_eq!(m.k.attached_to(m.cap(FREE + 5).unwrap().obj), 0);
    m.ok(S::DeviceAttach, [SLOT_DEVICES, 0x20, 0, 0, 0, 0]);
    // Destroying the address space (here: with everything else made from
    // the Untyped, as when a driver is ended) leaves its device reaching nothing.
    m.ok(S::CapRevoke, [ut, 0, 0, 0, 0, 0]);
    assert!(m.k.hal.attached.borrow().is_empty());
    // On a machine with no IOMMU there are no such address spaces, a device
    // cannot be confined, and the holder of the device authority is told
    // where a frame is; with one, nobody is.
    m.retype(ut, T::Frame, 12, frame, 1);
    assert_eq!(m.err(S::Inspect, [SLOT_DEVICES, frame, 0, 0, 0, 0]), Error::Unsupported);
    m.k.hal.io_top.set(0);
    assert_eq!(m.err(S::Retype, [ut, T::AddressSpace as u64, 16 | ASPACE_IO, SLOT_CAPSPACE, io, 1]), Error::Unsupported);
    m.ok(S::Inspect, [SLOT_DEVICES, frame, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), m.cap(frame).unwrap().obj);
    assert_eq!(m.err(S::Inspect, [SLOT_DEVICES, SLOT_ASPACE, 0, 0, 0, 0]), Error::WrongType);
}

/// An IOMMU without large pages maps a 2 MiB frame page by page, and the
/// frame is unmapped the same way.
#[test]
fn a_large_frame_is_mapped_for_devices_page_by_page_where_the_iommu_has_no_large_pages() {
    use silica_abi::ASPACE_IO;
    let m = Machine::new(16);
    m.k.hal.io_large.set(false);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 16 | ASPACE_IO, FREE, 1);
    m.retype(ut, T::Frame, 21, FREE + 1, 1);
    let space = m.cap(FREE).unwrap().obj;
    m.ok(S::Map, [FREE, FREE + 1, 0x40_3000, 3, 0, 0]);
    assert_eq!(m.k.translate(space, 0x60_2FFF), Some(m.cap(FREE + 1).unwrap().obj + 0x1F_FFFF));
    m.ok(S::Unmap, [FREE + 1, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.translate(space, 0x40_3000), m.k.translate(space, 0x60_2FFF)), (None, None));
}

/// A message-signalled interrupt line: bound for one device, it returns the
/// message that device's registers are given, and signals like a wire.
#[test]
fn a_message_interrupt_line_is_bound_for_one_device_and_returns_its_message() {
    use silica_abi::{IRQ_LEVEL, IRQ_MSI};
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 70, FREE, 0, SLOT_DEVICES, 0x18]), Error::BadArg, "a message line bound as a wire");
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI | IRQ_LEVEL, SLOT_DEVICES, 0x18]), Error::BadArg);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 5, FREE, IRQ_MSI, SLOT_DEVICES, 0x18]), Error::BadArg, "a wire bound as a message");
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, SLOT_IRQS, 0x18]), Error::WrongType, "no authority over the device");
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, SLOT_DEVICES, 15, 0x18 | 0x19 << 32, 0]);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, FREE + 1, 0x20]), Error::BadArg);
    m.ok(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, FREE + 1, 0x18]);
    assert_eq!((m.arg(m.root, 0), m.arg(m.root, 1)), (0xFEE0_0000, 0x30 + 70));
    assert_eq!(m.k.hal.irq_log.borrow().last().unwrap(), "message 70 cpu0 from 0x18");
    m.irq(70);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1);
    m.ok(S::IrqBind, [SLOT_IRQS, 70, 0, 0, 0, 0]);
    m.irq(70);
    assert_eq!(m.err(S::Recv, [FREE, 1, 0, 0, 0, 0]), Error::WouldBlock);
}

/// A receive on an endpoint that also listens on a notification: `a0` with
/// the notification's slot above bit 32.
fn listening(endpoint: u64, notification: u64) -> u64 {
    endpoint | notification << silica_abi::LISTEN_SHIFT
}

/// One wait for messages and signals: a receive on an endpoint that names a
/// notification hears a signal that came before it or comes while it waits,
/// as a result no message can imitate, and still receives messages.
#[test]
fn a_receive_listens_on_a_notification_while_it_waits_for_messages() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    let (ep, send, n) = (FREE, FREE + 1, FREE + 5);
    m.retype(ut, T::Notification, 0, n, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, n + 1, n, 15, 0b100, 0]);
    let notified = MsgInfo::NOTIFIED.0;
    // A signal already pending is returned at once, and cleared.
    m.ok(S::Send, [n + 1, 0, 0, 0, 0, 0]);
    let r = m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]);
    assert!(!r.blocked && r.status == 0);
    assert_eq!((m.arg(t, 0), m.arg(t, 1)), (0b100, notified));
    // Nothing pending: it waits, on the endpoint.
    assert!(m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    assert_eq!(m.k.thread_view(t).state, RECEIVING);
    // A signal ends the wait, and the thread is off the endpoint's queue:
    // a message sent now finds no receiver.
    m.ok(S::Send, [n + 1, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t).state, m.status(t), m.arg(t, 0), m.arg(t, 1)), (RUNNABLE, 0, 0b100, notified));
    assert_eq!(m.err(S::Send, [send | NONBLOCK, MsgInfo::new(1, 1).0, 5, 0, 0, 0]), Error::WouldBlock);
    assert_eq!(m.k.check_queue(m.cap(ep).unwrap().obj, false), Ok(()));
    // A message is received as ever, and no message can carry the mark of
    // a notification: the kernel clears the bit in what it delivers.
    assert!(m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Send, [send, MsgInfo(MsgInfo::new(9, 1).0 | notified).0, 5, 0, 0, 0]);
    assert_eq!((m.arg(t, 0), m.arg(t, 1), m.arg(t, 2)), (77, MsgInfo::new(9, 1).0, 5));
    // The thread has moved on (it is not receiving): a signal waits for
    // whoever looks next, here a plain wait on the notification, which
    // also ends the thread's standing as its listener.
    m.ok(S::Send, [n + 1, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    m.ok(S::Recv, [n, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0b100);
    // A plain receive after a listening one does not listen.
    assert!(m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Send, [send, MsgInfo::new(1, 1).0, 6, 0, 0, 0]);
    assert!(m.sys_as(t, S::Recv, [ep, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Send, [n + 1, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RECEIVING, "a plain receive is not ended by a signal");
    m.ok(S::Send, [send, MsgInfo::new(1, 1).0, 7, 0, 0, 0]);
    assert_eq!(m.arg(t, 2), 7);
    assert_eq!(m.k.check_queue(m.cap(n).unwrap().obj, true), Ok(()));
    // Listening is for an endpoint's receive, on a notification that can be read.
    assert_eq!(m.err(S::Recv, [listening(n, n), 1, 0, 0, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::Recv, [listening(ep, ep), 1, 0, 0, 0, 0]), Error::WrongType);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, n + 2, n, 2, 0, 0]);
    assert_eq!(m.err(S::Recv, [listening(ep, n + 2), 1, 0, 0, 0, 0]), Error::NoRights);
    assert_eq!(m.err(S::Recv, [listening(ep, n + 3), 1, 0, 0, 0, 0]), Error::BadCap);
}

/// One thread listens on a notification at a time, and nobody waits on it
/// directly meanwhile.
#[test]
fn a_notification_has_one_listener_and_no_waiter_beside_it() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    let (ep, n) = (FREE, FREE + 5);
    m.retype(ut, T::Notification, 0, n, 1);
    assert!(m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    assert_eq!(m.err(S::Recv, [n, 0, 0, 0, 0, 0]), Error::InUse, "a wait beside a listener");
    assert_eq!(m.err(S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]), Error::InUse, "a second listener");
    // And the other way round: nobody listens where a thread waits.
    m.retype(ut, T::Notification, 0, n + 1, 1);
    let t2 = m.spawn(ut, n + 2);
    assert!(m.sys_as(t2, S::Recv, [n + 1, 0, 0, 0, 0, 0]).blocked);
    assert_eq!(m.err(S::Recv, [listening(ep, n + 1), 0, 0, 0, 0, 0]), Error::InUse);
}

/// Taking the notification away ends the listening receive with `Gone`,
/// whether its capability leaves the slot or the notification is destroyed;
/// a destroyed listener leaves the notification usable.
#[test]
fn revoking_or_destroying_the_notification_ends_a_listening_receive() {
    let (m, t) = pair();
    let (ut, _) = m.big_untyped();
    let (ep, n) = (FREE, FREE + 5);
    m.retype(ut, T::Notification, 0, n, 1);
    // Through a derived capability, which is then revoked.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, n + 1, n, 15, 0, 0]);
    assert!(m.sys_as(t, S::Recv, [listening(ep, n + 1), 0, 0, 0, 0, 0]).blocked);
    m.ok(S::CapRevoke, [n, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t).state, m.status(t)), (RUNNABLE, Error::Gone as u64));
    assert_eq!(m.k.check_queue(m.cap(ep).unwrap().obj, false), Ok(()));
    // Moved to another slot: the receive was made through the old one.
    assert!(m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.ok(S::CapMove, [SLOT_CAPSPACE, n + 2, n, 0, 0, 0]);
    assert_eq!(m.status(t), Error::Gone as u64);
    // Destroyed while listened on.
    assert!(m.sys_as(t, S::Recv, [listening(ep, n + 2), 0, 0, 0, 0, 0]).blocked);
    m.ok(S::CapDelete, [n + 2, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t).state, m.status(t)), (RUNNABLE, Error::Gone as u64));
    // The listener destroyed while it listens: signals are kept, and
    // another thread can listen or wait.
    m.retype(ut, T::Notification, 0, n, 1);
    assert!(m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.ok(S::CapDelete, [FREE + 2, 0, 0, 0, 0, 0]);
    m.ok(S::Send, [n, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [listening(ep, n), 1, 0, 0, 0, 0]);
    assert_eq!((m.arg(m.root, 0), m.arg(m.root, 1)), (1, MsgInfo::NOTIFIED.0));
    assert_eq!(m.k.check_queue(m.cap(n).unwrap().obj, true), Ok(()));
}

/// A server's loop: reply, receive and listen in one call. An interrupt
/// ends the wait like any signal, also when it arrives on another
/// processor, and a signal from another processor takes two locks, not all.
#[test]
fn a_server_loop_replies_receives_and_hears_interrupts_in_one_call() {
    let (m, t) = two_processors();
    let (ut, _) = m.big_untyped();
    let (ep, send, n) = (FREE, FREE + 1, FREE + 5);
    m.retype(ut, T::Notification, 0, n, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, n + 1, n, 15, 0b10, 0]);
    // The server is the thread on processor 1; it binds an interrupt line there.
    m.on(1);
    assert_eq!(m.raw(S::IrqBind as u64, [SLOT_IRQS, 5, n + 1, 0, 0, 0]).status, 0);
    assert!(m.raw(S::Recv as u64, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    // The interrupt arrives on the server's processor: local.
    let globals = m.k.global_entries();
    m.irq(5);
    assert_eq!((m.status(t), m.arg(t, 0), m.arg(t, 1)), (0, 0b10, MsgInfo::NOTIFIED.0));
    assert_eq!(m.k.global_entries(), globals, "an interrupt on the listener's processor takes no other lock");
    // A call from processor 0 is served, and the reply goes back to
    // listening: the loop's steady state, on the fast path.
    assert!(m.raw(S::Recv as u64, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    assert!(m.raw(S::Call as u64, [send, MsgInfo::new(3, 1).0, 10, 0, 0, 0]).blocked);
    m.settle(1);
    assert_eq!((m.k.current(), m.arg(t, 2)), (t, 10));
    assert!(m.raw(S::ReplyRecv as u64, [listening(ep, n), MsgInfo::new(0, 1).0, 11, 0, 0, 0]).blocked);
    m.settle(0);
    assert_eq!((m.status(m.root), m.arg(m.root, 2)), (0, 11));
    assert_eq!(m.k.thread_view(t).state, RECEIVING);
    // A signal from processor 0 while the server listens on processor 1.
    let (globals, pairs) = (m.k.global_entries(), m.k.pair_entries());
    m.on(0);
    m.ok(S::Send, [n + 1, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t).state, m.arg(t, 0), m.arg(t, 1)), (RUNNABLE, 0b10, MsgInfo::NOTIFIED.0));
    assert_eq!((m.k.global_entries(), m.k.pair_entries()), (globals, pairs + 1), "a signal across processors takes two locks");
    // An interrupt taken on the other processor reaches the listener too.
    m.on(1);
    m.settle(1);
    assert!(m.raw(S::Recv as u64, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    m.irq(5);
    assert_eq!((m.k.thread_view(t).state, m.arg(t, 1)), (RUNNABLE, MsgInfo::NOTIFIED.0));
    // Moved to processor 0 while it listens, the server makes its receive
    // again there, and still hears the signal.
    m.on(1);
    m.settle(1);
    assert!(m.raw(S::Recv as u64, [listening(ep, n), 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    m.ok(S::ThreadControl, [FREE + 2, 2, 0, 0, 0, 0]);
    m.ok(S::Send, [n + 1, 0, 0, 0, 0, 0]);
    let r = m.sys_as(t, S::Recv, [listening(ep, n), 0, 0, 0, 0, 0]);
    assert!(!r.blocked);
    assert_eq!((m.arg(t, 0), m.arg(t, 1)), (0b10, MsgInfo::NOTIFIED.0));
    assert_eq!(m.k.check_runqueues(), Ok(()));
    assert_eq!(m.k.check_queue(m.cap(n).unwrap().obj, true), Ok(()));
}

/// A receive that signals first: the waiter's wake-up of its peer and its
/// own wait in one entry. The peer waiting on the other notification runs
/// with the signal's badge; the receiver then waits as a plain receive
/// does; a signal that cannot be made fails the receive before it waits.
#[test]
fn a_receive_can_signal_a_notification_first() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    // Two notifications: `mine` the root task waits on, `theirs` the peer does.
    let (mine, theirs) = (FREE, FREE + 1);
    m.retype(ut, T::Notification, 0, mine, 2);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, theirs, 2, 0b100, 0]);
    let peer = m.spawn(ut, FREE + 3);
    assert!(m.sys_as(peer, S::Recv, [theirs, 0, 0, 0, 0, 0]).blocked);
    // Signal `theirs` (badge 0b100), then look at `mine` without waiting.
    assert_eq!(m.err(S::SignalRecv, [mine, 1, 0, FREE + 2, 0, 0]), Error::WouldBlock, "then the receive found nothing");
    assert_eq!((m.k.thread_view(peer).state, m.arg(peer, 0)), (RUNNABLE, 0b100), "the peer was woken first");
    // Waiting: the signal, then the wait, which the peer's own signal ends.
    assert!(m.sys(S::SignalRecv, [mine, 0, 0, FREE + 2, 0, 0]).blocked);
    m.sys_as(peer, S::Send, [mine, 0, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(m.root).state, m.status(m.root), m.arg(m.root, 0)), (RUNNABLE, 0, 1));
    m.ok(S::Recv, [theirs, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0b100, "the second wait signalled the peer's notification too");
    // A signal already pending on `mine` is received at once, after the signal.
    m.ok(S::Send, [mine, 0, 0, 0, 0, 0]);
    m.ok(S::SignalRecv, [mine, 1, 0, FREE + 2, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1);
    m.ok(S::Recv, [theirs, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0b100);
    // The notification to signal must be one, and writable: otherwise
    // nothing is signalled and nothing received.
    m.ok(S::Send, [mine, 0, 0, 0, 0, 0]);
    assert_eq!(m.err(S::SignalRecv, [mine, 1, 0, FREE + 9, 0, 0]), Error::BadCap);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 9, theirs, 1, 0, 0]);
    assert_eq!(m.err(S::SignalRecv, [mine, 1, 0, FREE + 9, 0, 0]), Error::NoRights);
    m.ok(S::Recv, [mine, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1, "the failed receives took nothing");
}
