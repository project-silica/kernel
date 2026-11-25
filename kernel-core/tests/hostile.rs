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

/// Processor time is authority (`SLOT_SCHED`, narrowed by processor): a
/// scheduling context made from memory alone may lower its threads' share
/// and not raise it, nor reserve a processor. Without this, any holder of
/// an Untyped took 1,024 times a service's share, or 90% of the processor
/// every service runs on.
#[test]
fn processor_time_takes_a_capability_to_processor_time() {
    use silica_abi::{SCHED_DEADLINE, SCHED_DEFAULT_WEIGHT, SCHED_FAIR, SCHED_MAX_WEIGHT};
    const MS: u64 = 1_000_000;
    let m = Machine::with_cpus(16, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::SchedContext, 0, FREE, 2);
    let (sc, other) = (FREE, FREE + 1);
    // Memory alone: down, not up.
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, 0]), Error::NoRights);
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_DEFAULT_WEIGHT + 1, 0, 0, 0]), Error::NoRights);
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_DEADLINE, 9 * MS, 10 * MS, 0, 0]), Error::NoRights);
    m.ok(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_DEFAULT_WEIGHT / 2, 0, 0, 0]);
    m.ok(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, 0]);
    // Not any capability: one to processor time.
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, SLOT_IRQS]), Error::WrongType);
    // Narrowed to processor 1: a reservation there, not on processor 0, and
    // no weight above the default (a fair thread runs wherever it is moved).
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, SLOT_SCHED, 15, 1 | 2 << 32, 0]);
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_DEADLINE, 9 * MS, 10 * MS, 0, FREE + 2]), Error::NoRights);
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, FREE + 2]), Error::NoRights);
    m.ok(S::SchedConfigure, [sc, SCHED_DEADLINE, 9 * MS, 10 * MS, 1, FREE + 2]);
    // The whole machine's: anything.
    m.ok(S::SchedConfigure, [other, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, SLOT_SCHED]);
    m.ok(S::SchedConfigure, [other, SCHED_FAIR, 1, 0, 0, 0]);
    // A capability without its write right is no authority.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 3, SLOT_SCHED, Rights::READ.0 as u64, 0, 0]);
    assert_eq!(m.err(S::SchedConfigure, [other, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, FREE + 3]), Error::NoRights);
}

/// Destroying a CapSpace empties every slot it has (up to 2^32), holding
/// every processor's lock: a program with the memory for a large one must
/// not stop the machine with one call. The emptying stops for a pending
/// interrupt and goes on where it stopped; anything that could reuse
/// memory (a global system call) finishes it first.
#[test]
fn destroying_a_large_capspace_is_preemptible_and_finished_before_anything_else() {
    let m = Machine::new(64);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.retype(ut, T::CapSpace, 16, FREE + 1, 1);
    let n = m.k.slot_addr(&m.cs, FREE).unwrap();
    // Capabilities derived from the root's, in slots across the space.
    for i in [1u64, 1000, 30_000, 65_535] {
        m.ok(S::CapCopy, [FREE + 1, i, FREE, 15, 0, 0]);
    }
    let held = || {
        let v = m.k.view(n).unwrap();
        v.next != 0 && m.k.view(v.next).is_some_and(|c| c.depth > v.depth)
    };
    assert!(held());
    // One kernel entry with an interrupt pending: it stops, to be made again.
    let t = m.root;
    let r = m.k.regs(t);
    r[TestHal::IP].set(0x40_0002);
    r[TestHal::SYS].set(S::CapDelete as u64);
    r[TestHal::ARG[0]].set(FREE + 1);
    m.k.hal.pending.set(1);
    m.k.enter();
    m.k.syscall();
    m.k.leave();
    assert_eq!(r[TestHal::IP].get(), 0x40_0000, "the whole CapSpace was emptied in one go");
    assert!(m.cap(FREE + 1).is_none(), "the capability itself is gone at once");
    assert!(held(), "nothing was left for later");
    // The next system call that needs every lock finishes it first.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, FREE, 15, 0, 0]);
    assert!(m.k.view(m.k.slot_addr(&m.cs, FREE + 2).unwrap()).is_some());
    let v = m.k.view(n).unwrap();
    assert_eq!(v.next, m.k.slot_addr(&m.cs, FREE + 2).unwrap(), "the destroyed CapSpace's capabilities are still there");
    assert!(m.k.view(v.next).unwrap().next == 0 || m.k.view(m.k.view(v.next).unwrap().next).unwrap().depth <= v.depth);
}

/// One `Map` writes at most `MAP_MAX_ENTRIES` page-table entries: a frame
/// that would need more (a huge frame where only small pages are
/// available) is refused, so no mapping or unmapping holds the machine for
/// longer than that.
#[test]
fn a_mapping_too_large_for_one_call_is_refused() {
    let m = Machine::new(1024);
    m.k.hal.io_large.set(false);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 20 | silica_abi::ASPACE_IO, FREE, 1);
    m.retype(ut, T::Frame, 27, FREE + 2, 1);
    m.retype(ut, T::Frame, 28, FREE + 1, 1);
    let rw = (Rights::READ.0 | Rights::WRITE.0) as u64;
    assert_eq!(m.err(S::Map, [FREE, FREE + 1, 0, rw, 0, 0]), Error::BadArg, "256 MiB in 4 KiB pages");
    m.ok(S::Map, [FREE, FREE + 2, 0, rw, 0, 0]);
    m.ok(S::Unmap, [FREE + 2, 0, 0, 0, 0, 0]);
}

/// Table pages come back when what they mapped is unmapped. A program that
/// maps a peer's memory at a fresh address each time (as a service mapping
/// what its clients lend it does) ran its address space out of tables:
/// 120 lend-and-end cycles of a 1 MiB frame in an 18-bit space.
#[test]
fn page_tables_are_given_back_when_their_mappings_go() {
    for (bits, io) in [(14u64, false), (20, false), (22, false), (20, true)] {
        let m = Machine::new(if cfg!(miri) { 16 } else { 64 });
        let (ut, _) = m.big_untyped();
        m.retype(ut, T::AddressSpace, 18 | if io { silica_abi::ASPACE_IO } else { 0 }, FREE + 10, 1);
        m.retype(ut, T::Frame, bits, FREE, 1);
        let space = m.cap(FREE + 10).unwrap().obj;
        let step = 1u64 << bits.max(21);
        let mut at: u64 = if io { 0x40_0000 } else { 0x80_0000_0000 };
        // Under Miri (which checks the memory safety of the same path) a few
        // hundred: past the 120 that ran a 2^20 frame's space out before.
        for cycle in 0..if cfg!(miri) { 300u64 } else { 5000 } {
            let r = m.sys(S::Map, [FREE + 10, FREE, at, 3, 0, 0]);
            assert_eq!(r.status, 0, "2^{bits} (io {io}): {:?} after {cycle} cycles", Error::result(r.status));
            assert!(m.k.translate(space, at).is_some() || io);
            m.ok(S::Unmap, [FREE, 0, 0, 0, 0, 0]);
            assert!(m.k.translate(space, at).is_none());
            // Fresh addresses, spread over the space's reach.
            at = if io { (at + step) % (1 << 38) } else { 0x80_0000_0000 + (at + step * 1031) % (1 << 46) };
            at &= !(step - 1);
        }
    }
}

/// The boot secret (`RootSpec::entropy`, what every seed and key on the
/// machine descends from) reaches the root task in its first thread's
/// registers, never in a frame: the RootInfo frame describes the machine
/// and is lent on to whoever manages its devices (the device manager,
/// which parses hostile device data).
#[test]
fn the_boot_secret_is_in_no_frame_the_root_task_lends() {
    let m = Machine::new(16);
    let secret = support::BOOT_SECRET;
    let info = m.mem(m.info, 4096);
    assert!(!info.windows(8).any(|w| secret.chunks(8).any(|c| c == w)), "the RootInfo frame holds the boot secret");
    let words: Vec<u64> = secret.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(m.start_args[1..5], words[..], "the root thread does not start with the boot secret");
}

/// A level-triggered driver acknowledges its line after every interrupt:
/// that takes its own processor's lock, never every processor's (each
/// acknowledgement made every other processor's kernel entries wait).
#[test]
fn acknowledging_an_interrupt_takes_only_its_processors_lock() {
    let m = Machine::with_cpus(16, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::IrqBind, [SLOT_IRQS, 5, FREE, IRQ_LEVEL, 0, 0]);
    for _ in 0..10 {
        m.irq(5);
        m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
        let before = m.k.global_entries();
        m.ok(S::IrqAck, [SLOT_IRQS, 5, 0, 0, 0, 0]);
        assert_eq!(m.k.global_entries(), before, "an acknowledgement took every lock");
    }
    assert_eq!(m.k.hal.irq_log.borrow().last().map(String::as_str), Some("unmask 5"));
    // A line not bound: nothing to let interrupt.
    assert_eq!(m.err(S::IrqAck, [SLOT_IRQS, 300, 0, 0, 0, 0]), Error::BadArg);
}

/// Processors that never start are not counted, and indices stay
/// contiguous: secondaries take indices as they arrive, the count is closed
/// before the root task is told it, and a latecomer is turned away (it
/// would otherwise be a processor nothing knows of, or one counted that
/// never runs the threads placed on it).
#[test]
fn processors_are_counted_as_they_arrive_and_latecomers_are_turned_away() {
    use kernel_core::smp::Admission;
    let a = std::sync::Arc::new(Admission::new());
    // Seven of the firmware's eight secondaries arrive, from many threads.
    let got: Vec<usize> = (0..7)
        .map(|_| {
            let a = a.clone();
            std::thread::spawn(move || a.admit().unwrap())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let mut sorted = got.clone();
    sorted.sort();
    assert_eq!(sorted, (1..=7).collect::<Vec<_>>(), "indices are not 1 to 7");
    assert_eq!(a.arrived(), 7);
    assert_eq!(a.close(), 8, "the boot processor and the seven that came");
    assert_eq!(a.admit(), None, "a latecomer was admitted after the count was closed");
    assert_eq!(a.close(), 8);
}

/// Unbinding a message line takes it from the device: the machine's routing
/// for it (an interrupt remapping entry, an ITS mapping and its device's
/// translation table) is given back, so a device whose driver let it go
/// raises nothing, and the machine's routing slots do not run out.
#[test]
fn unbinding_a_message_line_releases_its_routing() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, SLOT_DEVICES, 15, 0x18 | 0x19 << 32, 0]);
    m.ok(S::IrqBind, [SLOT_IRQS, 70, FREE, IRQ_MSI, FREE + 1, 0x18]);
    m.ok(S::IrqBind, [SLOT_IRQS, 70, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.irq_log.borrow().last().map(String::as_str), Some("release 70"));
    // A line never bound has nothing to release.
    m.ok(S::IrqBind, [SLOT_IRQS, 71, 0, 0, 0, 0]);
    assert_eq!(m.k.hal.irq_log.borrow().iter().filter(|l| l.as_str() == "release 71").count(), 0);
}

/// An address space made isolated (`ASPACE_ISOLATED`) has the processor's
/// branch predictors flushed whenever a processor switches into or out of
/// it; switches between ordinary address spaces do not pay for that.
#[test]
fn switching_into_and_out_of_an_isolated_address_space_flushes_predictors() {
    use silica_abi::ASPACE_ISOLATED;
    let m = Machine::new(64);
    let (ut, _) = m.big_untyped();
    let in_space = |slot: u64, bits: u64| {
        m.retype(ut, T::AddressSpace, bits, slot + 1, 1);
        m.retype(ut, T::Thread, 0, slot, 1);
        m.ok(S::ThreadConfigure, [slot, SLOT_CAPSPACE, slot + 1, 0, 0, 0]);
        m.ok(S::ThreadStart, [slot, 0x40_0000, 0, 0, 1, 0]);
        m.cap(slot).unwrap().obj
    };
    assert_eq!(
        m.err(S::Retype, [ut, T::AddressSpace as u64, 16 | ASPACE_ISOLATED | silica_abi::ASPACE_IO, SLOT_CAPSPACE, FREE + 9, 1]),
        Error::BadArg
    );
    let plain = in_space(FREE, 16);
    let isolations = || m.k.hal.irq_log.borrow().iter().filter(|l| l.starts_with("isolate")).cloned().collect::<Vec<_>>();
    m.sys_as(plain, S::Yield, [0; 6]);
    m.sys(S::Yield, [0; 6]);
    assert!(isolations().is_empty(), "ordinary switches paid for isolation: {:?}", isolations());
    let secret = in_space(FREE + 2, 16 | ASPACE_ISOLATED);
    m.sys_as(secret, S::Yield, [0; 6]);
    m.sys(S::Yield, [0; 6]);
    assert_eq!(isolations(), ["isolate true", "isolate false"], "into and out of the isolated space");
    // An isolated space destroyed while it is loaded (its only thread
    // deletes the space and then itself): whatever runs next still finds
    // the predictors flushed, though the root it left is gone. (The first
    // isolated thread goes first, so only ordinary threads run beside it.)
    m.ok(S::CapDelete, [FREE + 2, 0, 0, 0, 0, 0]);
    let doomed = in_space(FREE + 4, 16 | ASPACE_ISOLATED);
    m.sys_as(doomed, S::CapDelete, [FREE + 5, 0, 0, 0, 0, 0]);
    let before = isolations().len();
    assert_eq!(isolations()[before - 1], "isolate true");
    m.sys_as(doomed, S::CapDelete, [FREE + 4, 0, 0, 0, 0, 0]);
    m.sys(S::Yield, [0; 6]);
    assert_eq!(isolations()[before..], ["isolate false"], "leaving a destroyed isolated space did not flush");
}

/// Threads are made from any memory, and a program may put as many as it
/// has memory for on the processor every service runs on. Making one of
/// them runnable placed it in a list kept in order of virtual runtime,
/// walked with interrupts off, and blocking with a deadline did the same
/// in the timer list: the more threads a program queued there, the longer
/// every wake-up and every timed wait on that processor took. Both are
/// trees now. Measured as the fastest of many operations beside 1,700
/// threads (600 with deadlines) against beside ten.
#[test]
fn many_queued_threads_do_not_slow_a_processors_wake_ups_or_timed_waits() {
    const SLEEPERS: u64 = 48;
    /// The fastest wake-up of a thread that slept, beside `hogs` runnable
    /// threads that have run more than it. Each woken sleeper wakes the
    /// next, so the waker is always the processor's current thread.
    fn fastest_wake_ns(hogs: u64) -> u128 {
        let m = Machine::new(64);
        let (ut, _) = m.big_untyped();
        m.retype(ut, T::Notification, 0, FREE, SLEEPERS);
        let sleepers: Vec<_> = (0..SLEEPERS)
            .map(|i| {
                let t = m.spawn(ut, FREE + 100 + i);
                assert!(m.sys_as(t, S::Recv, [FREE + i, 0, 0, 0, 0, 0]).blocked);
                t
            })
            .collect();
        // The root runs 10 ms alone, so the threads made next start level
        // with it, past where a sleeper is placed when it wakes (at most
        // 3 ms behind the least-run thread).
        m.k.hal.now.set(m.k.hal.now.get() + 10_000_000);
        m.ok(S::Yield, [0; 6]);
        for i in 0..hogs {
            m.spawn(ut, FREE + 200 + i);
        }
        let (mut best, mut waker) = (u128::MAX, m.root);
        for (i, s) in sleepers.into_iter().enumerate() {
            let start = std::time::Instant::now();
            let r = m.sys_as(waker, S::Send, [FREE + i as u64, 0, 0, 0, 0, 0]);
            best = best.min(start.elapsed().as_nanos());
            assert_eq!(r.status, 0);
            waker = s;
        }
        m.k.check_runqueues().unwrap();
        best
    }
    /// The fastest timed wait (a receive with a deadline after every
    /// other) beside `hogs` threads blocked with deadlines.
    fn fastest_timed_wait_ns(hogs: u64) -> u128 {
        let m = Machine::new(64);
        let (ut, _) = m.big_untyped();
        m.retype(ut, T::Notification, 0, FREE, SLEEPERS);
        let walkers: Vec<_> = (0..SLEEPERS).map(|i| m.spawn(ut, FREE + 100 + i)).collect();
        let far = m.k.hal.now.get() + 1_000_000_000_000;
        for i in 0..hogs {
            let t = m.spawn(ut, FREE + 200 + i);
            m.retype(ut, T::Notification, 0, FREE + 900 + i, 1);
            assert!(m.sys_as(t, S::Recv, [FREE + 900 + i, far - 1_000_000 + i, 0, 0, 0, 0]).blocked);
        }
        let mut best = u128::MAX;
        for (i, w) in walkers.into_iter().enumerate() {
            let start = std::time::Instant::now();
            assert!(m.sys_as(w, S::Recv, [FREE + i as u64, far, 0, 0, 0, 0]).blocked);
            best = best.min(start.elapsed().as_nanos());
        }
        m.k.check_timers().unwrap();
        best
    }
    let (few, many) = (fastest_wake_ns(10), fastest_wake_ns(1700));
    assert!(many < few * 3, "a wake-up beside 1,700 runnable threads took {many} ns, beside 10 {few} ns");
    let (few, many) = (fastest_timed_wait_ns(10), fastest_timed_wait_ns(600));
    assert!(many < few * 3, "a timed wait beside 600 others took {many} ns, beside 10 {few} ns");
}

/// A program reuses the addresses where lent memory was only once nothing
/// is mapped there any more, including the page of zeros its supervisor
/// mapped in place of memory the lender took back (whose capability is the
/// supervisor's, not the program's). `Unmap` of an address space clears a
/// range whole, whoever's frames are in it; it takes the write right.
#[test]
fn unmapping_a_range_clears_every_mapping_in_it_with_the_write_right_only() {
    let m = Machine::new(64);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 18, FREE + 10, 1);
    m.retype(ut, T::Frame, 12, FREE, 3);
    m.retype(ut, T::Frame, 21, FREE + 3, 1);
    let space = m.cap(FREE + 10).unwrap().obj;
    let base: u64 = 0x80_0000_0000;
    // Two pages inside the range, a 2 MiB frame overlapping its end, and a
    // page past it.
    m.ok(S::Map, [FREE + 10, FREE, base, 3, 0, 0]);
    m.ok(S::Map, [FREE + 10, FREE + 1, base + 0x5000, 3, 0, 0]);
    m.ok(S::Map, [FREE + 10, FREE + 3, base + 0x20_0000, 3, 0, 0]);
    m.ok(S::Map, [FREE + 10, FREE + 2, base + 0x60_0000, 3, 0, 0]);
    narrowed(&m, FREE + 10, FREE + 11, Rights::READ);
    assert_eq!(m.err(S::Unmap, [FREE + 11, base, 0x30_0000, 0, 0, 0]), Error::NoRights);
    assert_eq!(m.err(S::Unmap, [FREE + 10, u64::MAX, 2, 0, 0, 0]), Error::BadArg);
    assert!(m.k.translate(space, base).is_some());
    m.ok(S::Unmap, [FREE + 10, base, 0x30_0000, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 3, "three mappings overlapped the range");
    for at in [base, base + 0x5000, base + 0x20_0000, base + 0x3F_F000] {
        assert!(m.k.translate(space, at).is_none(), "{at:#x} still mapped");
    }
    assert!(m.k.translate(space, base + 0x60_0000).is_some(), "a page past the range went");
    // The frames are unmapped, not gone: each maps again, at the same place.
    m.ok(S::Map, [FREE + 10, FREE, base, 3, 0, 0]);
    m.ok(S::Map, [FREE + 10, FREE + 3, base + 0x20_0000, 3, 0, 0]);
    assert!(m.k.translate(space, base + 0x20_0000).is_some());
}
