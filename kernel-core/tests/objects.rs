//! Scenario tests of the kernel through its syscall interface, on the host machine.
mod support;

use kernel_core::elf::Elf;
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
    assert_eq!(m.err(S::Retype, [ut, T::SchedContext as u64, 0, cs, FREE, 1]), Error::Unsupported);
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
    m.retype(ut, T::AddressSpace, 15, FREE + 3, 1);
    m.ok(S::Map, [FREE + 3, FREE + 1, 0x1000_0000, 3, 0, 0]);
    m.ok(S::Map, [FREE + 3, FREE + 2, 0x1000_1000, 1, 0, 0]);
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
    assert!(m.k.hal.timer.get() <= at);
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
    assert!(m.k.hal.timer.get() != u64::MAX, "a second runnable thread arms the slice timer");
    m.k.hal.now.set(m.k.hal.timer.get());
    m.k.on_timer();
    m.k.schedule();
    assert_eq!(m.k.current(), t);
    m.sys_as(t, S::Yield, [0; 6]);
    assert_eq!(m.k.current(), m.root);
    m.ok(S::ThreadControl, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, INACTIVE);
    m.ok(S::ThreadControl, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, RUNNABLE);
    assert_eq!(m.err(S::ThreadControl, [FREE, 2, 0, 0, 0, 0]), Error::BadArg);
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
    let log = String::from_utf8(m.k.hal.console.borrow().clone()).unwrap();
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
    let mut r = Rng(0x5EED);
    for _ in 0..if cfg!(miri) { 3 } else { 300 } {
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
    assert!(m.k.hal.timer.get() != u64::MAX);
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
