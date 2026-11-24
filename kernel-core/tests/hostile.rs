//! A hostile program against the kernel: each test is one attack a
//! program (or a driver, which is a program) can make through the syscall
//! interface, and checks that it gains nothing. Authority is only ever what
//! a capability's rights say; these tests try to get more than that.
mod support;

use kernel_core::hal::Hal as _;
use kernel_core::thread::{REPLY, RUNNABLE};
use silica_abi::root::*;
use silica_abi::{Error, Fault, IRQ_LEVEL, IRQ_MSI, KEEP, MsgInfo, ObjectType as T, Rights, Syscall as S};
use support::*;

const FREE: u64 = 2000;

/// A copy of the capability in `from` into `to` with only `rights`.
fn narrowed(m: &Machine, from: u64, to: u64, rights: Rights) {
    m.ok(S::CapCopy, [SLOT_CAPSPACE, to, from, rights.0 as u64, 0, 0]);
}

/// Running a thread in a CapSpace or an address space is full use of it,
/// and a fault endpoint is sent through: a capability narrowed below that
/// cannot be installed.
#[test]
fn thread_configure_takes_only_capabilities_that_carry_the_use_it_makes() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Thread, 0, FREE, 1);
    m.retype(ut, T::Endpoint, 0, FREE + 1, 1);
    let t = m.cap(FREE).unwrap().obj;
    let rw = Rights(Rights::READ.0 | Rights::WRITE.0);
    narrowed(&m, SLOT_CAPSPACE, FREE + 2, Rights::READ);
    narrowed(&m, SLOT_ASPACE, FREE + 3, Rights::READ);
    narrowed(&m, FREE + 1, FREE + 4, Rights::READ);
    narrowed(&m, SLOT_CAPSPACE, FREE + 5, rw);
    narrowed(&m, SLOT_ASPACE, FREE + 6, rw);
    narrowed(&m, FREE + 1, FREE + 7, Rights::WRITE);
    assert_eq!(m.err(S::ThreadConfigure, [FREE, FREE + 2, KEEP, KEEP, KEEP, 0]), Error::NoRights, "a read-only CapSpace");
    assert_eq!(m.err(S::ThreadConfigure, [FREE, KEEP, FREE + 3, KEEP, KEEP, 0]), Error::NoRights, "a read-only address space");
    assert_eq!(m.err(S::ThreadConfigure, [FREE, KEEP, KEEP, FREE + 4, KEEP, 0]), Error::NoRights, "a fault endpoint it may not send to");
    let slots = m.k.thread_view(t).slots;
    assert!(slots[..3].iter().all(|s| m.k.view(*s).is_none()), "something was installed");
    m.ok(S::ThreadConfigure, [FREE, FREE + 5, FREE + 6, FREE + 7, KEEP, 0]);
    assert_eq!(m.k.view(slots[2]).unwrap().cap.rights, Rights::WRITE);
}

/// Looking at kernel state through a capability is a use of it that needs
/// its read right: a capability narrowed to nothing tells nothing.
#[test]
fn inspect_needs_the_read_right() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::SchedContext, 0, FREE, 1);
    m.retype(ut, T::Notification, 0, FREE + 1, 1);
    narrowed(&m, SLOT_THREAD, FREE + 2, Rights::WRITE);
    narrowed(&m, SLOT_CAPSPACE, FREE + 3, Rights::WRITE);
    narrowed(&m, FREE, FREE + 4, Rights::WRITE);
    narrowed(&m, FREE + 1, FREE + 5, Rights::WRITE);
    narrowed(&m, SLOT_DEVICES, FREE + 6, Rights::WRITE);
    narrowed(&m, SLOT_CAPSPACE, FREE + 7, Rights::READ);
    assert_eq!(m.err(S::Inspect, [FREE + 2, 0, 0, 0, 0, 0]), Error::NoRights, "a thread's registers");
    assert_eq!(m.err(S::Inspect, [FREE + 3, SLOT_THREAD, 0, 0, 0, 0]), Error::NoRights, "a CapSpace's slots");
    assert_eq!(m.err(S::Inspect, [FREE + 3, SLOT_THREAD, 1, 0, 0, 0]), Error::NoRights, "a thread through a CapSpace");
    assert_eq!(m.err(S::Inspect, [FREE + 4, 0, 0, 0, 0, 0]), Error::NoRights, "a scheduling context");
    assert_eq!(m.err(S::Inspect, [FREE + 5, 0, 0, 0, 0, 0]), Error::NoRights, "a notification");
    assert_eq!(m.err(S::Inspect, [FREE + 6, SLOT_IPCBUF, 0, 0, 0, 0]), Error::NoRights, "a frame's physical address");
    // Read is enough.
    m.ok(S::Inspect, [FREE + 7, SLOT_THREAD, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), T::Thread as u64);
}

/// `SignalRecv` signals; it carries no message. A call parked on the
/// notification it signals is not answered by it (its deadline register is
/// not a message descriptor, and no capability moves): the signal
/// accumulates, as any signal to a notification with a parked call does.
#[test]
fn signal_recv_does_not_answer_a_parked_call_with_its_deadline() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 77, 0]);
    let t = m.spawn(ut, FREE + 2);
    let u = m.spawn(ut, FREE + 3);
    m.retype(ut, T::Notification, 0, FREE + 5, 1);
    let n = m.cap(FREE + 5).unwrap().obj;
    assert!(m.sys_as(t, S::Recv, [FREE, 0, FREE + 5, 0, 0, 0]).blocked);
    assert!(m.sys(S::Call, [FREE + 1, MsgInfo::new(1, 1).0, 7, 0, 0, 0]).blocked);
    assert_eq!(m.k.thread_view(m.root).blocked_on, n, "the call is parked");
    // The server signals the notification its caller is parked on, with a
    // "deadline" that reads as a descriptor with a label, words and a
    // capability.
    let forged = MsgInfo::with_caps(0x77, 2, 1).0;
    assert!(m.sys_as(t, S::SignalRecv, [FREE, forged, 0, FREE + 5, 0, 0]).blocked);
    let v = m.k.thread_view(m.root);
    assert_eq!((v.state, v.blocked_on), (REPLY, n), "the signal answered the parked call");
    // The answer is a message, sent: then the signal is still pending.
    assert!(!m.sys_as(u, S::Send, [FREE + 5, MsgInfo::new(3, 1).0, 9, 0, 0, 0]).blocked);
    m.sys_as(u, S::Yield, [0; 6]);
    while m.k.current() != m.root {
        m.sys_as(u, S::Yield, [0; 6]);
    }
    assert_eq!((m.status(m.root), MsgInfo(m.arg(m.root, 1)).label(), m.arg(m.root, 2)), (0, 3, 9));
    m.ok(S::Recv, [FREE + 5, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 1, "the signal was lost");
}

/// A rebind that fails leaves the line as it was: the old binding keeps
/// its interrupts, and the line is not masked.
#[test]
fn a_failed_irq_bind_keeps_the_binding_it_would_have_replaced() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, SLOT_DEVICES, 15, 0x18 | 0x19 << 32, 0]);
    let poll = || match m.sys(S::Recv, [FREE, 1, 0, 0, 0, 0]) {
        r if r.status == 0 => Some(m.arg(m.root, 0)),
        _ => None,
    };
    m.ok(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, FREE + 1, 0x18]);
    m.ok(S::IrqBind, [SLOT_IRQS, 5, FREE, 0, 0, 0]);
    let log = m.k.hal.irq_log.borrow().len();
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI | IRQ_LEVEL, FREE + 1, 0x18]), Error::BadArg);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, FREE + 1, 0x20]), Error::BadArg);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 5, FREE, IRQ_MSI, 0, 0]), Error::BadArg);
    assert_eq!(m.err(S::IrqBind, [SLOT_IRQS, 5, FREE + 1, 0, 0, 0]), Error::WrongType);
    assert_eq!(m.k.hal.irq_log.borrow().len(), log, "a failed bind touched the controller: {:?}", m.k.hal.irq_log.borrow());
    m.irq(70);
    assert_eq!(poll(), Some(1), "the message line lost its binding");
    m.irq(5);
    assert_eq!(poll(), Some(1), "the wire lost its binding");
}

/// `IrqWatch` and `IrqQuiet` name a frame by its slot. Replacing that frame
/// with a smaller one at the same address (delete it, and retype the same
/// memory into the same slot) must not let the kernel read or write the
/// word past the new frame's end.
#[test]
fn a_watch_does_not_reach_past_a_smaller_frame_put_in_its_slot() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, SLOT_DEVICES, 15, 0x18 | 0x19 << 32, 0]);
    m.retype(ut, T::Untyped, 21, FREE + 10, 1);
    m.retype(FREE + 10, T::Frame, 20, FREE + 11, 1);
    let big = m.cap(FREE + 11).unwrap().obj;
    m.ok(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, FREE + 1, 0x18]);
    m.ok(S::IrqWatch, [SLOT_IRQS, 70, FREE + 11, 0xF_FF00, 0xFFFF, 0]);
    m.ok(S::IrqQuiet, [SLOT_IRQS, 70, FREE + 11, 0xF_FF04, 1, 0]);
    m.ok(S::CapDelete, [FREE + 11, 0, 0, 0, 0, 0]);
    m.retype(FREE + 10, T::Frame, 12, FREE + 11, 1);
    assert_eq!(m.cap(FREE + 11).unwrap().obj, big, "the small frame is at the same address");
    // Past the small frame: memory its Untyped may make anything of.
    let past = big + 0xF_FF04;
    m.k.phys.write_val(past, &0u32);
    m.k.hal.spy.set(m.k.phys.ptr(past).cast::<u32>().cast_const());
    m.k.hal.spied.set(0xDEAD);
    assert!(m.sys(S::Recv, [FREE, m.k.hal.now() + 1_000_000, 0, 0, 0, 0]).blocked);
    m.idle();
    assert_eq!(m.k.hal.spied.get(), 0xDEAD, "the kernel watched a word past the frame");
    assert_eq!(u32::from_le_bytes(m.mem(past, 4).try_into().unwrap()), 0, "the kernel wrote past the frame");
    // The watch is inert: every interrupt is news.
    m.irq(70);
    m.irq(70);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
}

/// A program that makes its thread fault again and again, with nobody to
/// tell, does not get a kernel console line per fault: the console is
/// shared, and writing it is slow on real hardware.
#[test]
fn faults_with_nobody_to_tell_are_not_logged_without_limit() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let t = m.spawn(ut, FREE);
    for _ in 0..1000 {
        m.sys_as(t, S::Yield, [0; 6]);
        while m.k.current() != t {
            m.sys(S::Yield, [0; 6]);
        }
        m.fault(Fault::Exception, 0, 13);
        assert_ne!(m.k.thread_view(t).state, RUNNABLE);
        m.ok(S::ThreadStart, [FREE, 0x40_0000, 0, 0, 1, 0]);
    }
    let log = String::from_utf8_lossy(&m.k.hal.console.borrow()).into_owned();
    let lines = log.matches("no fault endpoint").count();
    assert!((1..=20).contains(&lines), "{lines} lines for 1000 faults in no time");
    // Time passing: the next is told, with how many were not.
    m.k.hal.now.set(m.k.hal.now() + 2_000_000_000);
    m.sys_as(t, S::Yield, [0; 6]);
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.fault(Fault::Exception, 0, 13);
    let log = String::from_utf8_lossy(&m.k.hal.console.borrow()).into_owned();
    assert!(log.contains("not logged"), "{log}");
}

/// A thread object reused for a new program (`ThreadStart`) starts it with
/// the initial vector and x87 state, not the previous program's: its save
/// area is reset, and state still live in a processor's registers is not
/// saved over it.
#[test]
fn thread_start_resets_the_extended_state() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    let t = m.spawn(ut, FREE);
    let area = t + kernel_core::thread::FPU_OFFSET as u64;
    let len = 4096 - kernel_core::thread::FPU_OFFSET as u64;
    // The previous program's state, in the save area.
    for i in 0..len / 8 {
        m.k.phys.write_val(area + i * 8, &0x5EC2_E75E_C2E7_5EC2u64);
    }
    // It is running, its state live: it starts a new program in itself.
    m.sys_as(t, S::Yield, [0; 6]);
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.raw(S::ThreadStart as u64, [FREE, 0x40_1000, 0, 0, 1, 0]);
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    let (from, to) = m.k.hal.last_switch.get();
    assert_eq!(to, t);
    assert_eq!(from, 0, "the old program's live state was saved over the new one's");
    assert_eq!(u64::from_le_bytes(m.mem(area, 8).try_into().unwrap()), 0x1F80, "not the initial image");
    assert!(m.is_zero(area + 8, len - 8), "the previous program's state is still in the save area");
}
