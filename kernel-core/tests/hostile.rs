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
/// tell, writes nothing on the kernel's console: the fault stops the thread
/// and stays in it for whoever holds it. (Before, each was a console line
/// until one ration of ten a second for the machine ran out, so one
/// program's faults took the lines another's would have had.) Only the
/// thread the kernel started itself, which nobody holds, is spoken for by
/// the kernel.
#[test]
fn faults_with_nobody_to_tell_write_nothing_on_the_console() {
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
    let said = || String::from_utf8_lossy(&m.k.hal.console.borrow()).matches("no fault endpoint").count();
    assert_eq!(said(), 0, "console lines for a program's faults");
    // The root task's first thread: the kernel says it, at once.
    while m.k.current() != m.root {
        m.sys_as(t, S::Yield, [0; 6]);
    }
    m.fault(Fault::Exception, 0, 13);
    assert_eq!(said(), 1, "the root task stopped and the kernel did not say so");
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
/// scheduling context is a share or a reservation of a processor only by
/// a capability to that processor's time, and memory alone makes none.
/// Without this, any holder of an Untyped took 1,024 times a service's
/// share, or 90% of the processor every service runs on.
#[test]
fn processor_time_takes_a_capability_to_processor_time() {
    use silica_abi::{SCHED_DEADLINE, SCHED_DEFAULT_WEIGHT, SCHED_FAIR, SCHED_MAX_WEIGHT};
    const MS: u64 = 1_000_000;
    let m = Machine::with_cpus(16, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::SchedContext, 0, FREE, 2);
    let (sc, other) = (FREE, FREE + 1);
    // Memory alone: nothing, large or small.
    for weight in [SCHED_MAX_WEIGHT, SCHED_DEFAULT_WEIGHT + 1, SCHED_DEFAULT_WEIGHT, 1] {
        assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, weight, 0, 0, 0]), Error::NoRights);
    }
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_DEADLINE, 9 * MS, 10 * MS, 0, 0]), Error::NoRights);
    // Not any capability: one to processor time.
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, SLOT_IRQS]), Error::WrongType);
    // Narrowed to processor 1: a share or a reservation there, none on
    // processor 0.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, SLOT_SCHED, 15, 1 | 2 << 32, 0]);
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_DEADLINE, 9 * MS, 10 * MS, 0, FREE + 2]), Error::NoRights);
    assert_eq!(m.err(S::SchedConfigure, [sc, SCHED_FAIR, 1, 0, 0, FREE + 2]), Error::NoRights);
    m.ok(S::SchedConfigure, [sc, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 1, FREE + 2]);
    m.ok(S::SchedConfigure, [sc, SCHED_DEADLINE, 9 * MS, 10 * MS, 1, FREE + 2]);
    // The whole machine's: anything.
    m.ok(S::SchedConfigure, [other, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, SLOT_SCHED]);
    m.ok(S::SchedConfigure, [other, SCHED_FAIR, 1, 0, 1, SLOT_SCHED]);
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
/// One test per frame size and kind of space, so Miri runs them side by side
/// (in one test, one after another, they were the longest of its run).
#[test]
fn page_tables_are_given_back_when_their_mappings_go_16k() {
    tables_given_back(14, false);
}

#[test]
fn page_tables_are_given_back_when_their_mappings_go_1m() {
    tables_given_back(20, false);
}

#[test]
fn page_tables_are_given_back_when_their_mappings_go_4m() {
    tables_given_back(22, false);
}

#[test]
fn page_tables_are_given_back_when_their_mappings_go_for_devices() {
    tables_given_back(20, true);
}

fn tables_given_back(bits: u64, io: bool) {
    {
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
    // The root task's secret, derived from the boot's entropy (`ident.rs`).
    let words = kernel_core::ident::root_secret(&support::BOOT_SECRET);
    let info = m.mem(m.info, 4096);
    assert!(!info.windows(8).any(|w| words.iter().any(|s| s.to_le_bytes() == w)), "the RootInfo frame holds the boot secret");
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
        m.ok(S::ThreadControl, [slot, 2, silica_abi::root::SLOT_CONTEXT, 0, 0, 0]);
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
    // Under Miri (which checks the memory safety of the same paths) a few
    // threads, and no time: its times are not the machine's.
    if cfg!(miri) {
        fastest_wake_ns(40);
        fastest_timed_wait_ns(40);
        return;
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

/// Processor time went by thread: each fair thread was a share, so a
/// program with the memory for a hundred threads took a hundred shares of
/// the processor every service runs on. Measured on the kernel before that
/// was mended, a default-weight service beside a program of n threads (all
/// on one processor, a second of time): n = 1, 501 ms and waits of at most
/// 3 ms; n = 10, 93 ms and 30 ms; n = 100, 12 ms and 300 ms. Now time is a
/// capability: a fair context (made with authority over the processor's
/// time) is one share however many threads are bound to it, a thread runs
/// only once it is bound, and a program has nothing to bind its threads to
/// but the context it was given: not no context at all, not one it makes
/// from its own memory.
#[test]
fn a_program_with_many_threads_takes_one_share() {
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    use std::collections::HashMap;
    const MS: u64 = 1_000_000;
    fn next_timer(m: &Machine) -> u64 {
        let t = m.k.hal.timer();
        if m.k.hal.armed[0].get() && t > m.k.hal.now.get() { t } else { u64::MAX }
    }
    let share = |n: u64| -> (u64, u64) {
        let m = Machine::with_cpus(64, 2);
        m.on(0);
        let (ut, _) = m.big_untyped();
        // Two contexts on processor 0, made by the holder of processor time.
        m.retype(ut, T::SchedContext, 0, FREE, 2);
        let (gs, gh) = (FREE, FREE + 1);
        for g in [gs, gh] {
            m.ok(S::SchedConfigure, [g, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
        }
        let service = m.spawn(ut, FREE + 2);
        m.ok(S::ThreadControl, [FREE + 2, 2, gs, 0, 0, 0]);
        let first = m.spawn(ut, FREE + 3);
        m.ok(S::ThreadControl, [FREE + 3, 2, gh, 0, 0, 0]);
        // The program makes the rest itself, and tries every way to more
        // than its share: a context of its own is no time, whatever it
        // asks of it.
        m.retype(ut, T::SchedContext, 0, FREE + 4, 1);
        let own = FREE + 4;
        for weight in [SCHED_DEFAULT_WEIGHT, 1] {
            assert_eq!(m.sys_as(first, S::SchedConfigure, [own, SCHED_FAIR, weight, 0, 0, 0]).status, Error::NoRights as u64);
        }
        let mut threads = vec![first];
        for i in 1..n {
            let slot = FREE + 10 + i;
            assert_eq!(m.sys_as(first, S::Retype, [ut, T::Thread as u64, 0, SLOT_CAPSPACE, slot, 1]).status, 0);
            assert_eq!(m.sys_as(first, S::ThreadConfigure, [slot, SLOT_CAPSPACE, SLOT_ASPACE, 0, 0, 0]).status, 0);
            // Bound to nothing, or to its own context, the thread has no
            // time: it is not started, and not resumed.
            let refused = match i % 3 {
                0 => m.sys_as(first, S::ThreadStart, [slot, 0x40_0000, 0, 0, 1, 0]),
                1 => m.sys_as(first, S::ThreadControl, [slot, 2, own, 0, 0, 0]),
                _ => m.sys_as(first, S::ThreadControl, [slot, 1, 0, 0, 0, 0]),
            };
            assert_eq!(refused.status, Error::NoTime as u64);
            // What is left is the context the program was given.
            assert_eq!(m.sys_as(first, S::ThreadControl, [slot, 2, gh, 0, 0, 0]).status, 0);
            assert_eq!(m.sys_as(first, S::ThreadStart, [slot, 0x40_0000, 0, 0, 1, 0]).status, 0);
            threads.push(m.cap(slot).unwrap().obj);
        }
        assert!(m.sys(S::ThreadControl, [SLOT_THREAD, 0, 0, 0, 0, 0]).blocked);
        m.k.check_runqueues().unwrap();
        let (mut ran, mut last, mut worst) = (HashMap::<u64, u64>::new(), m.k.hal.now.get(), 0);
        let until = m.k.hal.now.get() + 1000 * MS;
        let mut steps = 0;
        while m.k.hal.now.get() < until {
            let now = m.k.hal.now.get();
            let next = next_timer(&m).min(until);
            let cur = m.k.current();
            if cur == service {
                worst = worst.max(now - last);
                last = next;
            }
            *ran.entry(cur).or_insert(0) += next - now;
            m.k.hal.now.set(next);
            m.tick();
            steps += 1;
            assert!(steps < 1_000_000);
        }
        m.k.check_runqueues().unwrap();
        (ran.get(&service).copied().unwrap_or(0) / MS, worst / MS)
    };
    for n in [1, 10, 100] {
        let (ran, worst) = share(n);
        println!("a service beside a program of {n} threads: {ran} ms of 1000, waits of at most {worst} ms");
        assert!(ran >= 450, "beside a program of {n} threads the service ran {ran} ms of 1000");
        assert!(worst <= 7, "beside a program of {n} threads the service waited {worst} ms");
    }
}

/// A thread runs only on time it was bound to through a capability. One
/// made and never bound cannot be started or resumed, and never runs; a
/// context made from memory is nobody's time until a holder of a
/// processor's time says so, for that processor only; and a program in a
/// CapSpace of its own binds its threads to the context it holds and to no
/// other: not one it cannot name, not one it holds without the read right,
/// and it cannot raise the share it was given.
#[test]
fn a_thread_runs_only_on_time_it_was_bound_to_through_a_capability() {
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR, SCHED_MAX_WEIGHT};
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    // Made, configured, never bound.
    m.retype(ut, T::Thread, 0, FREE, 1);
    let t = m.cap(FREE).unwrap().obj;
    m.ok(S::ThreadConfigure, [FREE, SLOT_CAPSPACE, SLOT_ASPACE, 0, 0, 0]);
    assert_eq!(m.err(S::ThreadStart, [FREE, 0x40_0000, 0, 0, 1, 0]), Error::NoTime);
    m.ok(S::ThreadStart, [FREE, 0x40_0000, 0, 0, 0, 0]);
    assert_eq!(m.err(S::ThreadControl, [FREE, 1, 0, 0, 0, 0]), Error::NoTime);
    // A context from memory alone: no time, and none to be had for it
    // without the processor's (here, a capability to processor 1's only).
    m.retype(ut, T::SchedContext, 0, FREE + 1, 2);
    assert_eq!(m.err(S::ThreadControl, [FREE, 2, FREE + 1, 0, 0, 0]), Error::NoTime);
    assert_eq!(m.err(S::SchedConfigure, [FREE + 1, SCHED_FAIR, 1, 0, 0, 0]), Error::NoRights);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 3, SLOT_SCHED, 15, 1 | 2 << 32, 0]);
    assert_eq!(m.err(S::SchedConfigure, [FREE + 1, SCHED_FAIR, 1, 0, 0, FREE + 3]), Error::NoRights);
    assert_eq!(m.k.thread_view(t).state, 0);
    for _ in 0..50 {
        m.k.hal.now.set(m.k.hal.now.get() + 1_000_000);
        m.tick();
        assert_ne!(m.k.current(), t, "a thread bound to no context ran");
    }
    m.k.check_runqueues().unwrap();
    // A program: a CapSpace of its own holding its second thread (slot 1),
    // its own context read-only (slot 2) and another's without the read
    // right (slot 3).
    let (own, others) = (FREE + 1, FREE + 2);
    for g in [own, others] {
        m.ok(S::SchedConfigure, [g, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
    }
    m.retype(ut, T::CapSpace, 8, FREE + 4, 1);
    m.retype(ut, T::Thread, 0, FREE + 5, 2);
    m.ok(S::CapCopy, [FREE + 4, 1, FREE + 6, 15, 0, 0]);
    m.ok(S::CapCopy, [FREE + 4, 2, own, Rights::READ.0 as u64, 0, 0]);
    m.ok(S::CapCopy, [FREE + 4, 3, others, Rights::WRITE.0 as u64, 0, 0]);
    m.ok(S::ThreadConfigure, [FREE + 5, FREE + 4, SLOT_ASPACE, 0, 0, 0]);
    m.ok(S::ThreadConfigure, [FREE + 6, FREE + 4, SLOT_ASPACE, 0, 0, 0]);
    m.ok(S::ThreadControl, [FREE + 5, 2, own, 0, 0, 0]);
    m.ok(S::ThreadStart, [FREE + 5, 0x40_0000, 0, 0, 1, 0]);
    let p = m.cap(FREE + 5).unwrap().obj;
    let second = m.cap(FREE + 6).unwrap().obj;
    let as_p = |n: S, a: [u64; 6]| Error::result(m.sys_as(p, n, a).status);
    assert_eq!(as_p(S::ThreadControl, [1, 2, others, 0, 0, 0]), Err(Error::BadCap), "bound to a context it cannot name");
    assert_eq!(as_p(S::ThreadControl, [1, 2, SLOT_CONTEXT, 0, 0, 0]), Err(Error::BadCap), "bound to the root task's own context");
    assert_eq!(as_p(S::ThreadControl, [1, 2, 3, 0, 0, 0]), Err(Error::NoRights), "bound to a context it holds without the read right");
    assert_eq!(as_p(S::ThreadStart, [1, 0x40_0000, 0, 0, 1, 0]), Err(Error::NoTime));
    assert_eq!(as_p(S::SchedConfigure, [2, SCHED_FAIR, SCHED_MAX_WEIGHT, 0, 0, 0]), Err(Error::NoRights), "raised its own share");
    assert_eq!(as_p(S::ThreadControl, [1, 2, 2, 0, 0, 0]), Ok(()));
    assert_eq!(as_p(S::ThreadStart, [1, 0x40_0000, 0, 0, 1, 0]), Ok(()));
    assert_eq!(m.k.thread_view(second).state, RUNNABLE);
    m.k.check_runqueues().unwrap();
}

/// Holding every processor's lock stops the machine, so the time a program
/// spends so is bounded (the 2026-11-24 audit: a flood of capability
/// operations took two thirds of other processors' round trips). The bound
/// was each processor's: a program that used it up held back every other
/// program on its processor, whose calls waited for a share they had not
/// touched. It is the caller's scheduling context's: the flood waits, and
/// a program beside it on the same processor carries on.
#[test]
fn a_flood_of_lock_taking_calls_holds_back_its_own_context_only() {
    use kernel_core::thread::THROTTLED;
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    // A thread running on processor 1: what holding every lock costs.
    m.spawn(ut, FREE + 20);
    m.move_to(FREE + 20, 1);
    m.settle(1);
    m.on(0);
    // The flood and its victim on processor 0, each on a context of its own.
    m.retype(ut, T::SchedContext, 0, FREE, 2);
    for g in [FREE, FREE + 1] {
        m.ok(S::SchedConfigure, [g, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
    }
    let flood = m.spawn(ut, FREE + 2);
    m.ok(S::ThreadControl, [FREE + 2, 2, FREE, 0, 0, 0]);
    let victim = m.spawn(ut, FREE + 3);
    m.ok(S::ThreadControl, [FREE + 3, 2, FREE + 1, 0, 0, 0]);
    m.retype(ut, T::Endpoint, 0, FREE + 4, 1);
    // Every reading of the clock is 10 us later than the last, so a call
    // that takes every lock holds them at least that long.
    m.k.hal.now_step.set(10_000);
    // Capability copies and deletes until the flood is held back.
    let mut calls = 0;
    while m.k.thread_view(flood).state != THROTTLED {
        let slot = FREE + 5;
        let call = if m.cap(slot).is_none() {
            (S::CapCopy, [SLOT_CAPSPACE, slot, FREE + 4, 15, 0, 0])
        } else {
            (S::CapDelete, [slot, 0, 0, 0, 0, 0])
        };
        let r = m.sys_as(flood, call.0, call.1);
        assert!(r.blocked || r.status == 0, "{:?}", Error::result(r.status));
        calls += 1;
        assert!(calls < 100_000, "a flood of calls that take every lock was never held back");
    }
    // The victim makes the same calls beside it, and none of them waits.
    for i in 0..100 {
        let slot = FREE + 6;
        let call = if i % 2 == 0 { (S::CapCopy, [SLOT_CAPSPACE, slot, FREE + 4, 15, 0, 0]) } else { (S::CapDelete, [slot, 0, 0, 0, 0, 0]) };
        let r = m.sys_as(victim, call.0, call.1);
        assert!(
            !r.blocked,
            "call {i} of a program on the flood's processor waited for the flood's share (after {calls} calls of the flood)"
        );
        assert_eq!(r.status, 0);
    }
    // The flood's share comes back with time, and it carries on.
    m.k.hal.now_step.set(0);
    m.k.hal.now.set(m.k.hal.now.get() + 1_000_000_000);
    m.tick();
    assert_eq!(m.k.thread_view(flood).state, RUNNABLE);
    m.k.check_runqueues().unwrap();
}

/// What holding every lock costs a program on another processor is the
/// whole time held, however many other processors are idle: a pair of
/// programs calling each other across two processors has one of them
/// running at a time, and each call of the pair waits for the locks. So a
/// flood on a machine of four processors beside one busy one is held
/// back after spending its burst, as on a machine of two (before: it was
/// charged a third, and on eight processors a flood beside such a pair
/// kept 69% of the pair's round trips; `smp_global_flood8`).
#[test]
fn a_flood_beside_idle_processors_pays_the_whole_time_it_holds_every_lock() {
    use kernel_core::smp::GLOBAL_BURST_NS;
    use kernel_core::thread::THROTTLED;
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    let m = Machine::with_cpus(64, 4);
    let (ut, _) = m.big_untyped();
    // One other processor has work; the other two are idle.
    m.spawn(ut, FREE + 20);
    m.move_to(FREE + 20, 1);
    m.settle(1);
    m.on(0);
    m.retype(ut, T::SchedContext, 0, FREE, 1);
    m.ok(S::SchedConfigure, [FREE, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
    let flood = m.spawn(ut, FREE + 2);
    m.ok(S::ThreadControl, [FREE + 2, 2, FREE, 0, 0, 0]);
    m.retype(ut, T::Endpoint, 0, FREE + 4, 1);
    // Every reading of the clock is 10 us later than the last, so a call
    // that takes every lock holds them at least that long.
    m.k.hal.now_step.set(10_000);
    let mut calls = 0u64;
    while m.k.thread_view(flood).state != THROTTLED {
        let slot = FREE + 5;
        let call = if m.cap(slot).is_none() {
            (S::CapCopy, [SLOT_CAPSPACE, slot, FREE + 4, 15, 0, 0])
        } else {
            (S::CapDelete, [slot, 0, 0, 0, 0, 0])
        };
        let r = m.sys_as(flood, call.0, call.1);
        assert!(r.blocked || r.status == 0, "{:?}", Error::result(r.status));
        calls += 1;
        assert!(calls < 100_000, "a flood of calls that take every lock was never held back");
    }
    assert!(
        calls * 10_000 <= 2 * GLOBAL_BURST_NS,
        "a flood beside two idle processors and one busy one held every lock for {} calls of at least 10 us before it was held back, its burst being {} us",
        calls,
        GLOBAL_BURST_NS / 1000
    );
}

/// Binding takes a thread to its context's processor (the context's
/// capability is the authority to), fair or reserved, and a context with
/// threads bound keeps its processor. When a context's last capability
/// goes, so does the time its threads ran on: they stop, bound to nothing,
/// until they are bound to other time and resumed.
#[test]
fn a_context_holds_its_threads_on_its_processor() {
    use silica_abi::{SCHED_DEADLINE, SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    let m = Machine::with_cpus(64, 2);
    m.on(0);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::SchedContext, 0, FREE, 2);
    m.ok(S::SchedConfigure, [FREE, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 1, SLOT_SCHED]);
    let t = m.spawn(ut, FREE + 2);
    m.ok(S::ThreadControl, [FREE + 2, 2, FREE, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).cpu, 1, "binding took it to the context's processor");
    m.k.check_runqueues().unwrap();
    // A reservation is on its processor too, and a thread goes from one
    // context to another wherever each is (suspended, it stays so).
    m.ok(S::SchedConfigure, [FREE + 1, SCHED_DEADLINE, 1_000_000, 10_000_000, 0, SLOT_SCHED]);
    let d = m.spawn(ut, FREE + 3);
    m.ok(S::ThreadControl, [FREE + 3, 2, FREE, 0, 0, 0]);
    assert_eq!(m.k.thread_view(d).cpu, 1);
    m.ok(S::ThreadControl, [FREE + 3, 0, 0, 0, 0, 0]);
    m.ok(S::ThreadControl, [FREE + 3, 2, FREE + 1, 0, 0, 0]);
    assert_eq!((m.k.thread_view(d).cpu, m.k.thread_view(d).state), (0, 0));
    m.k.check_runqueues().unwrap();
    // A context in use keeps its processor.
    assert_eq!(m.err(S::SchedConfigure, [FREE, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]), Error::InUse);
    assert_eq!(m.err(S::SchedConfigure, [FREE + 1, SCHED_DEADLINE, 1_000_000, 10_000_000, 1, SLOT_SCHED]), Error::InUse);
    // Its last capability gone: its thread has no time.
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert_eq!(m.k.thread_view(t).state, 0, "a thread whose context is gone still runs");
    assert_eq!(m.err(S::ThreadControl, [FREE + 2, 1, 0, 0, 0, 0]), Error::NoTime);
    m.k.check_runqueues().unwrap();
    m.move_to(FREE + 2, 0);
    m.ok(S::ThreadControl, [FREE + 2, 1, 0, 0, 0, 0]);
    assert_eq!((m.k.thread_view(t).cpu, m.k.thread_view(t).state), (0, RUNNABLE));
    m.k.check_runqueues().unwrap();
}

/// A device (or a driver programming one) that raises an edge or message
/// line without end cost a processor: each interrupt was delivered whole.
/// A line that fires while its notification bit is still pending brings no
/// news, and the kernel knows it exactly, so that delivery holds the line
/// off (masked, or its remapping entry sent to the discard vector) until
/// the waiter receives: one interrupt per receive, whatever the device
/// does. (Before, the hold began after 64 such deliveries in a row, a
/// count standing in for the pending bit.)
#[test]
fn a_line_that_fires_with_its_news_pending_is_held_off_until_its_waiter_receives() {
    let m = Machine::with_cpus(16, 1);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 2);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, FREE, 15, 0b10, 0]);
    m.ok(S::IrqBind, [silica_abi::root::SLOT_IRQS, 5, FREE + 2, 0, 0, 0]);
    let log = || m.k.hal.irq_log.borrow().clone();
    let held = || log().iter().filter(|l| l.as_str() == "mask 5").count();
    let let_go = || log().iter().filter(|l| l.as_str() == "unmask 5").count();
    m.irq(5);
    assert_eq!(held(), 0, "the first interrupt is news");
    m.irq(5);
    assert_eq!(held(), 1, "the second found the news pending: it holds the line off: {:?}", log());
    for _ in 0..100 {
        m.irq(5);
    }
    assert_eq!(held(), 1, "held once: {:?}", log());
    // A receive on another notification lets nothing of this one go.
    m.ok(S::Send, [FREE + 1, 0, 0, 0, 0, 0]);
    m.ok(S::Recv, [FREE + 1, 1, 0, 0, 0, 0]);
    assert_eq!(let_go(), 0);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), 0b10);
    assert_eq!((let_go(), m.k.line_held(5)), (1, false), "taking the news lets the line interrupt again");
    // Heard and taken one at a time, it is never held.
    let before = held();
    for _ in 0..10 {
        m.irq(5);
        m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    }
    assert_eq!(held(), before);
    // Bound anew while held, the hold ends with the old binding.
    m.irq(5);
    m.irq(5);
    assert_eq!(held(), before + 1);
    m.ok(S::IrqBind, [silica_abi::root::SLOT_IRQS, 5, FREE + 2, 0, 0, 0]);
    assert!(!m.k.line_held(5));
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    m.irq(5);
    assert_eq!(held(), before + 1, "a hold of the old binding does not outlive it");
    assert_eq!(let_go(), 1, "the new binding is not released twice");
}

/// A hold is its notification's: when the notification goes, no line stays
/// held for it. (Before, a driver that let its line storm and then deleted
/// the notification left the kernel's count of held lines above zero for
/// good, and every receive on any notification of the machine walked all
/// 192 lines from then on.)
#[test]
fn a_held_line_does_not_outlive_its_notification() {
    let m = Machine::with_cpus(16, 1);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::IrqBind, [silica_abi::root::SLOT_IRQS, 5, FREE, 0, 0, 0]);
    m.ok(S::IrqBind, [silica_abi::root::SLOT_IRQS, 6, FREE, 0, 0, 0]);
    for _ in 0..70 {
        m.irq(5);
        m.irq(6);
    }
    assert!(m.k.line_held(5) && m.k.line_held(6));
    // One of them unbound: the other is still held, and let go by a receive.
    m.ok(S::IrqBind, [silica_abi::root::SLOT_IRQS, 5, 0, 0, 0, 0]);
    assert!(!m.k.line_held(5) && m.k.line_held(6));
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert!(!m.k.line_held(6), "a line is still held off for a notification that is gone");
}

/// A watch is kept in its line's own entry: a driver can watch every line
/// it holds, and no number of them takes another driver's. (Before, the
/// kernel had sixteen watches for the machine and whoever asked first had
/// them: the seventeenth `IrqWatch` was `NoMemory`, whoever made it.)
#[test]
fn watching_many_lines_takes_no_watch_from_anyone() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 1, SLOT_DEVICES, 15, 0x18 | 0x19 << 32, 0]);
    m.retype(ut, T::Frame, 12, FREE + 2, 1);
    let frame = m.cap(FREE + 2).unwrap().obj;
    for line in 70..110u64 {
        m.ok(S::IrqBind, [SLOT_IRQS, line, FREE, IRQ_MSI, FREE + 1, 0x18]);
        m.ok(S::IrqWatch, [SLOT_IRQS, line, FREE + 2, (line - 70) * 4, 0xFFFF, 0]);
    }
    // Each still tells news from none: line 109's word changed, 70's not.
    m.k.phys.write_val(frame + 39 * 4, &7u32);
    m.irq(70);
    assert_eq!(m.err(S::Recv, [FREE, 1, 0, 0, 0, 0]), Error::WouldBlock, "an interrupt with its word unchanged is no news");
    m.irq(109);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    // Unwatched, a line's every interrupt is news again; the rest stay.
    m.ok(S::IrqWatch, [SLOT_IRQS, 70, 0, 0, 0, 0]);
    m.irq(70);
    m.ok(S::Recv, [FREE, 1, 0, 0, 0, 0]);
    m.irq(71);
    assert_eq!(m.err(S::Recv, [FREE, 1, 0, 0, 0, 0]), Error::WouldBlock);
}

/// One kernel entry of system call `n` as the root task, with processor 1
/// waiting for a lock throughout (each look for a pending interrupt counts
/// a turn of its wait): whether it stopped to be made again.
fn one_entry_beside_a_waiter(m: &Machine, n: S, a: [u64; 6]) -> bool {
    m.k.hal.contender.set(m.k.lock_wait_turns(1));
    let r = m.k.regs(m.root);
    r[TestHal::IP].set(0x40_0002);
    r[TestHal::SYS].set(n as u64);
    r[TestHal::BY_SYSCALL].set(1);
    for (i, v) in a.iter().enumerate() {
        r[TestHal::ARG[i]].set(*v);
    }
    m.k.enter();
    m.k.syscall();
    m.k.leave();
    m.k.hal.contender.set(core::ptr::null());
    r[TestHal::IP].get() == 0x40_0000
}

/// How many capabilities derive from the one in slot `s` of the root CapSpace.
fn children(m: &Machine, s: u64) -> usize {
    let at = m.k.slot_addr(&m.cs, s).unwrap();
    let Some(v) = m.k.view(at) else { return 0 };
    let (mut n, mut next) = (0, v.next);
    while let Some(c) = (next != 0).then(|| m.k.view(next)).flatten().filter(|c| c.depth > v.depth) {
        n += 1;
        next = c.next;
    }
    n
}

/// A capability may have as many descendants as there is memory for slots
/// (64 bytes each), and deleting it empties them holding every lock. It
/// stopped only for an interrupt pending on its own processor, which a
/// processor with no timer armed may never have: one `Delete` held every
/// other processor for as long as the memory a program had paid for let it
/// make (the 2026-11-27 audit: H4, N1). It stops for a processor waiting
/// for a lock too, a bounded number of capabilities after it began, and
/// is made again until it is done.
#[test]
fn a_delete_of_many_children_stops_for_a_processor_waiting() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.retype(ut, T::CapSpace, 14, FREE + 1, 1);
    let many = if cfg!(miri) { 1200 } else { 10_000 };
    for i in 1..=many {
        m.ok(S::CapCopy, [FREE + 1, i, FREE, 15, 0, 0]);
    }
    assert_eq!(children(&m, FREE), many as usize);
    assert!(one_entry_beside_a_waiter(&m, S::CapDelete, [FREE, 0, 0, 0, 0, 0]), "one entry deleted all {many} children");
    let left = children(&m, FREE);
    assert!(left > 0 && (many as usize - left) <= 1024, "one entry deleted {} children", many as usize - left);
    // Made again, it finishes.
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    assert!(m.cap(FREE).is_none() && m.cap(FREE + 1).is_some());
    assert!(m.k.view(m.k.slot_addr(&m.cs, FREE + 1).unwrap()).is_some());
}

/// The same for a CapSpace being emptied: one of its slots may hold a
/// capability with as many descendants, which the emptying deleted in one
/// go whatever was pending.
#[test]
fn emptying_a_capspace_stops_inside_one_slot_for_a_processor_waiting() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Notification, 0, FREE, 1);
    m.retype(ut, T::CapSpace, 4, FREE + 1, 1);
    m.retype(ut, T::CapSpace, 14, FREE + 3, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2, FREE, 15, 0, 0]);
    let many = if cfg!(miri) { 1200 } else { 10_000 };
    for i in 1..=many {
        m.ok(S::CapCopy, [FREE + 3, i, FREE + 2, 15, 0, 0]);
    }
    // The parent of them all goes into the small CapSpace, which is deleted.
    m.ok(S::CapMove, [FREE + 1, 1, FREE + 2, 0, 0, 0]);
    assert_eq!(children(&m, FREE), many as usize + 1);
    assert!(one_entry_beside_a_waiter(&m, S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]), "one entry emptied the slot's {many} descendants");
    let left = children(&m, FREE);
    assert!(left > 0 && (many as usize + 1 - left) <= 1024, "one entry deleted {} capabilities", many as usize + 1 - left);
    m.ok(S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]);
    assert_eq!(children(&m, FREE), 0, "the emptying was finished");
}

/// An unmap of many frames tells each other processor once that the
/// address space's mappings changed, not once a frame: told at each frame,
/// every processor running in the space was brought into the kernel and
/// dropped its translations, once a frame, as often as a program asked
/// (the 2026-11-27 audit: N3). Here the other processor has the space not
/// loaded, so it forgets what it cached: once.
#[test]
fn an_unmap_of_many_frames_tells_each_processor_once() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 16, FREE, 1);
    let frames = 64u64;
    for k in (0..frames).step_by(250) {
        m.retype(ut, T::Frame, 12, FREE + 1 + k, 250.min(frames - k));
    }
    for i in 0..frames {
        m.ok(S::Map, [FREE, FREE + 1 + i, 0x10_0000 + i * 0x1000, 3, 0, 0]);
    }
    let before = m.k.hal.forgets.get();
    m.ok(S::Unmap, [FREE, 0x10_0000, frames * 0x1000, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 0), frames, "every frame was unmapped");
    let told = m.k.hal.forgets.get() - before;
    assert!(told <= 1, "the other processor was told {told} times for one unmap of {frames} frames");
    // Deleting many mapped frames at once, likewise.
    for i in 0..frames {
        m.ok(S::Map, [FREE, FREE + 1 + i, 0x10_0000 + i * 0x1000, 3, 0, 0]);
    }
    let before = m.k.hal.forgets.get();
    m.ok(S::CapDelete, [FREE, 0, 0, 0, 0, 0]);
    let told = m.k.hal.forgets.get() - before;
    assert!(told <= 2, "the other processor was told {told} times when the address space went");
}

/// An address space may have as many mappings as there is memory for
/// frames, and both an unmap of a range and the space's destruction went
/// through all of them holding every lock, whatever was waiting (the
/// 2026-11-27 audit: H4). Beside a processor waiting for a lock, an unmap
/// of a range stops a bounded number of mappings in and is made again from
/// where it stopped, its count whole; if the mapping it stopped at goes
/// meanwhile it starts over, and still unmaps exactly what is in range.
#[test]
fn an_unmap_of_a_large_range_stops_for_a_processor_waiting_and_goes_on() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 20, FREE, 1);
    let frames = if cfg!(miri) { 600 } else { 2000 };
    for k in (0..frames).step_by(250) {
        m.retype(ut, T::Frame, 12, FREE + 1 + k, 250.min(frames - k));
    }
    let at = |i: u64| 0x10_0000 + i * 0x1000;
    for i in 0..frames {
        m.ok(S::Map, [FREE, FREE + 1 + i, at(i), 3, 0, 0]);
    }
    let mapped = |m: &Machine| {
        (0..frames).filter(|&i| m.k.view(m.k.slot_addr(&m.cs, FREE + 1 + i).unwrap()).unwrap().mapped.is_some()).count() as u64
    };
    // Everything but the first frame is in range.
    let range = [FREE, at(1), (frames - 1) * 0x1000, 0, 0, 0];
    assert!(one_entry_beside_a_waiter(&m, S::Unmap, range), "one entry unmapped all of {frames} frames");
    let left = mapped(&m);
    assert!(left > 1 && frames - left <= 1024, "one entry unmapped {} frames", frames - left);
    // The frame it goes on from is deleted meanwhile (its slot emptied).
    m.k.hal.contender.set(core::ptr::null());
    let gone = (1..frames).find(|&i| m.k.view(m.k.slot_addr(&m.cs, FREE + 1 + i).unwrap()).unwrap().mapped.is_some()).unwrap();
    m.ok(S::CapDelete, [FREE + 1 + gone, 0, 0, 0, 0, 0]);
    m.ok(S::Unmap, range);
    let still: Vec<u64> = (0..frames)
        .filter(|&i| m.cap(FREE + 1 + i).is_some() && m.k.view(m.k.slot_addr(&m.cs, FREE + 1 + i).unwrap()).unwrap().mapped.is_some())
        .collect();
    assert_eq!(still, [0], "exactly the frame out of range is still mapped");
}

/// The destruction of an address space with many mappings stops likewise,
/// and its frames can be mapped elsewhere at once, before every one has
/// been let go of.
#[test]
fn destroying_an_address_space_of_many_mappings_stops_for_a_processor_waiting() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 20, FREE, 1);
    m.retype(ut, T::AddressSpace, 16, FREE + 1, 1);
    let frames = if cfg!(miri) { 600 } else { 2000 };
    for k in (0..frames).step_by(250) {
        m.retype(ut, T::Frame, 12, FREE + 2 + k, 250.min(frames - k));
    }
    for i in 0..frames {
        m.ok(S::Map, [FREE, FREE + 2 + i, 0x10_0000 + i * 0x1000, 3, 0, 0]);
    }
    let mapped = |m: &Machine| {
        (0..frames).filter(|&i| m.k.view(m.k.slot_addr(&m.cs, FREE + 2 + i).unwrap()).unwrap().mapped.is_some()).count() as u64
    };
    assert!(one_entry_beside_a_waiter(&m, S::CapDelete, [FREE, 0, 0, 0, 0, 0]), "one entry let go of all {frames} mappings");
    assert!(m.cap(FREE).is_none(), "the address space itself is gone at once");
    let left = mapped(&m);
    assert!(left > 0 && frames - left <= 1024, "one entry let go of {} mappings", frames - left);
    // A frame still listed in the gone space maps elsewhere at once.
    let last = FREE + 2 + frames - 1;
    m.ok(S::Map, [FREE + 1, last, 0x20_0000, 3, 0, 0]);
    let v = m.k.view(m.k.slot_addr(&m.cs, last).unwrap()).unwrap();
    assert_eq!(v.mapped.map(|(a, _)| a), Some(m.cap(FREE + 1).unwrap().obj), "mapped in the new space");
    // The next system call that takes every lock finishes the rest.
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 2050, FREE + 1, 15, 0, 0]);
    assert_eq!(mapped(&m), 1, "only the frame mapped again is mapped");
}

/// A fault whose fault endpoint another processor owns, or nobody yet,
/// is sent holding every lock, and faults cost a program nothing to make:
/// unlike a system call that takes every lock, they were never paid for
/// (the 2026-11-27 audit: M10). They are now, out of the same share of the
/// faulting thread's context: a flood of them is held back (the thread
/// waits, to fault again where it was), and a program beside it carries on.
#[test]
fn a_flood_of_faults_that_take_every_lock_is_held_back() {
    use kernel_core::thread::THROTTLED;
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.spawn(ut, FREE + 20);
    m.move_to(FREE + 20, 1);
    m.settle(1);
    m.on(0);
    m.retype(ut, T::SchedContext, 0, FREE, 1);
    m.ok(S::SchedConfigure, [FREE, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
    // Nobody receives on the fault endpoint: it is no processor's, so each
    // fault sent to it takes every lock.
    m.retype(ut, T::Endpoint, 0, FREE + 1, 1);
    let threads: Vec<_> = (0..1800u64)
        .map(|i| {
            let slot = FREE + 100 + i;
            let t = m.spawn(ut, slot);
            m.ok(S::ThreadConfigure, [slot, KEEP, KEEP, FREE + 1, KEEP, 0]);
            m.ok(S::ThreadControl, [slot, 2, FREE, 0, 0, 0]);
            // Suspended until its turn: the flood is the faults, not a
            // run queue of threads to yield through.
            m.ok(S::ThreadControl, [slot, 0, 0, 0, 0, 0]);
            t
        })
        .collect();
    // During each fault every reading of the clock is 10 us later than
    // the last: a fault that takes every lock holds them that long at least.
    let mut faults = 0;
    let mut throttled = false;
    for (i, &t) in threads.iter().enumerate() {
        m.ok(S::ThreadControl, [FREE + 100 + i as u64, 1, 0, 0, 0, 0]);
        m.sys_as(t, S::Yield, [0; 6]);
        while m.k.current() != t {
            m.sys(S::Yield, [0; 6]);
        }
        m.k.hal.now_step.set(10_000);
        m.fault(Fault::Page, 0xdead_0000, 6);
        m.k.hal.now_step.set(0);
        faults += 1;
        if m.k.thread_view(t).state == THROTTLED {
            throttled = true;
            break;
        }
    }
    assert!(throttled, "{faults} faults that took every lock were never held back");
}

/// Inspect shows no physical address: what it gives as an object's
/// identity is equal for all capabilities to one object and different for
/// different objects, but none of the frame's, the Untyped's, the
/// endpoint's or (as what a thread waits on) the queue's physical address,
/// nor anything that keeps their order or distance (the 2026-11-27 audit:
/// M13, N4: Inspect gave out physical addresses, a thread's `waits_on`
/// among them, and with them the machine's memory layout).
#[test]
fn inspect_shows_identities_not_physical_addresses() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Frame, 12, FREE, 2);
    m.retype(ut, T::Endpoint, 0, FREE + 2, 1);
    m.ok(S::CapCopy, [SLOT_CAPSPACE, FREE + 3, FREE + 2, 15, 9, 0]);
    let shown = |m: &Machine, s: u64| {
        m.ok(S::Inspect, [SLOT_CAPSPACE, s, 0, 0, 0, 0]);
        m.arg(m.root, 3)
    };
    let [f0, f1, e, e2, u] = [shown(&m, FREE), shown(&m, FREE + 1), shown(&m, FREE + 2), shown(&m, FREE + 3), shown(&m, ut)];
    let addr = |s: u64| m.cap(s).unwrap().obj;
    for (id, s) in [(f0, FREE), (f1, FREE + 1), (e, FREE + 2), (u, ut)] {
        assert_ne!(id, addr(s), "Inspect showed the physical address of slot {s}'s object");
    }
    assert_eq!(e, e2, "two capabilities to one endpoint show one identity");
    assert!(f0 != f1 && f0 != e && f1 != e, "different objects show different identities");
    // Neighbouring frames 4 KiB apart are not shown 4 KiB apart.
    assert_ne!(f1.wrapping_sub(f0), addr(FREE + 1).wrapping_sub(addr(FREE)));
    // A thread waiting on the endpoint shows that identity.
    let t = m.spawn(ut, FREE + 4);
    assert!(m.sys_as(t, S::Recv, [FREE + 2, 0, 0, 0, 0, 0]).blocked);
    m.ok(S::Inspect, [FREE + 4, 0, 0, 0, 0, 0]);
    assert_eq!(m.arg(m.root, 4), e, "what it waits on, by identity");
}

/// Deleting a capability to an address space made every other processor
/// look again at what it should have loaded, through a kick it waited
/// for, once a capability: deleting many such capabilities kicked every
/// processor as often (the 2026-11-27 audit, beside N3). It is told once,
/// when the entry lets go of every lock.
#[test]
fn deleting_many_address_space_capabilities_kicks_each_processor_once() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 16, FREE, 1);
    m.retype(ut, T::CapSpace, 10, FREE + 1, 1);
    for i in 1..=500 {
        m.ok(S::CapCopy, [FREE + 1, i, FREE, 15, 0, 0]);
    }
    let before = m.k.hal.kicks.borrow().len();
    m.ok(S::CapRevoke, [FREE, 0, 0, 0, 0, 0]);
    let kicks = m.k.hal.kicks.borrow()[before..].iter().filter(|k| k.0 == 1).count();
    assert!(kicks <= 1, "processor 1 was kicked {kicks} times for one revoke of 500 copies");
}

/// A page table given back (its last mapping gone) may still be cached by
/// another processor, at whatever level it was, until that processor is
/// told the space's mappings changed; the telling is put off to the end of
/// the global entry (`shoot_later`). Reused before that, in the same
/// entry, as a table of another level, it would give that processor a walk
/// through whatever the new table points at: any physical memory (the
/// review of audit-kernel-stalls). Here one global entry finishes emptying
/// a CapSpace, unmapping a frame alone in its tables, and then answers a
/// fault by mapping a frame where new tables are needed: the others are
/// told before any table given back is taken again.
#[test]
fn a_table_given_back_is_not_reused_before_the_others_are_told() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.retype(ut, T::CapSpace, 12, FREE + 1, 1);
    m.retype(ut, T::Frame, 12, FREE + 2, 2);
    let asp = m.cap(SLOT_ASPACE).unwrap().obj;
    // A frame alone in its tables, its capability late in a CapSpace whose
    // emptying stops before it reaches it.
    m.ok(S::Map, [SLOT_ASPACE, FREE + 2, 0x6000_0000, 3, 0, 0]);
    m.ok(S::CapMove, [FREE + 1, 3000, FREE + 2, 0, 0, 0]);
    // A thread of this space faults to the root task.
    let t = m.spawn(ut, FREE + 4);
    m.ok(S::ThreadConfigure, [FREE + 4, KEEP, KEEP, FREE, KEEP, 0]);
    while m.k.current() != t {
        m.sys(S::Yield, [0; 6]);
    }
    m.fault(Fault::Page, 0x7000_0123, 6);
    m.ok(S::Recv, [FREE, 0, 0, 0, 0, 0]);
    assert!(one_entry_beside_a_waiter(&m, S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]));
    assert!(m.k.translate(asp, 0x6000_0000).is_some(), "the emptying reached the frame in its first entry");
    // The answer (a capability in it: the general path, every lock) maps
    // a frame where new tables are needed, in the entry that finishes the
    // emptying and so gives the frame's tables back.
    m.ok(S::ReplyRecv, [0, MsgInfo::with_caps(0, 4, 1).0, silica_abi::FAULT_MAP, SLOT_ASPACE, FREE + 3, 0x7000_0000 | 3]);
    assert!(m.k.translate(asp, 0x6000_0000).is_none());
    assert_eq!(m.k.translate(asp, 0x7000_0123), Some(m.cap(FREE + 3).unwrap().obj + 0x123));
}

/// One kernel entry of system call `n` as thread `t` (others yield until it
/// runs), processor 1 waiting for a lock throughout: whether it stopped.
fn one_entry_as_beside_a_waiter(m: &Machine, t: u64, n: S, a: [u64; 6]) -> bool {
    while m.k.current() != t {
        let c = m.k.current();
        m.k.regs(c)[TestHal::SYS].set(S::Yield as u64);
        m.k.enter();
        m.k.syscall();
        m.k.schedule();
        m.k.leave();
    }
    m.k.hal.contender.set(m.k.lock_wait_turns(1));
    let r = m.k.regs(t);
    r[TestHal::IP].set(0x40_0002);
    r[TestHal::SYS].set(n as u64);
    r[TestHal::BY_SYSCALL].set(1);
    for (i, v) in a.iter().enumerate() {
        r[TestHal::ARG[i]].set(*v);
    }
    m.k.enter();
    m.k.syscall();
    m.k.leave();
    m.k.hal.contender.set(core::ptr::null());
    r[TestHal::IP].get() == 0x40_0000
}

/// Two holders of one address space unmapping ranges of it at once, each
/// stopping beside a processor waiting, both finish, each with its whole
/// count, and a frame's holder moving its capability meanwhile loses
/// neither place. With one place kept a space, taken by whoever called
/// last, each started the other over: with more mappings in neither range
/// at the head of the list than one entry gets through, neither ever got
/// past them (the review of audit-kernel-stalls).
#[test]
fn two_holders_unmapping_ranges_of_one_space_both_finish() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 20, FREE, 1);
    let (a_n, b_n, c_n) = if cfg!(miri) { (100u64, 100u64, 600u64) } else { (300, 300, 600) };
    let total = a_n + b_n + c_n;
    for k in (0..total).step_by(250) {
        m.retype(ut, T::Frame, 12, FREE + 1 + k, 250.min(total - k));
    }
    let at = |i: u64| 0x10_0000 + i * 0x1000;
    // A's range first, then B's, then the rest last (the head of the list).
    for i in 0..total {
        m.ok(S::Map, [FREE, FREE + 1 + i, at(i), 3, 0, 0]);
    }
    let a_range = [FREE, at(0), a_n * 0x1000, 0, 0, 0];
    let b_range = [FREE, at(a_n), b_n * 0x1000, 0, 0, 0];
    let a = m.spawn(ut, FREE + 1990);
    let b = m.spawn(ut, FREE + 1991);
    let (mut a_done, mut b_done) = (None, None);
    for _ in 0..200 {
        if a_done.is_none() && !one_entry_as_beside_a_waiter(&m, a, S::Unmap, a_range) {
            a_done = Some(m.arg(a, 0));
        }
        if b_done.is_none() && !one_entry_as_beside_a_waiter(&m, b, S::Unmap, b_range) {
            b_done = Some(m.arg(b, 0));
        }
    }
    assert_eq!((a_done, b_done), (Some(a_n), Some(b_n)), "the two unmaps, alternating, did not both finish with their counts");
}

/// The same for an unmap whose place a frame's holder moves: the head's
/// frames' capabilities are moved between its entries.
#[test]
fn an_unmap_of_a_range_finishes_while_frames_are_moved() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::AddressSpace, 20, FREE, 1);
    let (a_n, b_n, c_n) = if cfg!(miri) { (100u64, 100u64, 600u64) } else { (300, 300, 600) };
    let total = a_n + b_n + c_n;
    for k in (0..total).step_by(250) {
        m.retype(ut, T::Frame, 12, FREE + 1 + k, 250.min(total - k));
    }
    let at = |i: u64| 0x10_0000 + i * 0x1000;
    for i in 0..total {
        m.ok(S::Map, [FREE, FREE + 1 + i, at(i), 3, 0, 0]);
    }
    let a_range = [FREE, at(0), a_n * 0x1000, 0, 0, 0];
    let a = m.spawn(ut, FREE + 1990);
    let mut a_done = None;
    for round in 0..200u64 {
        if !one_entry_as_beside_a_waiter(&m, a, S::Unmap, a_range) {
            a_done = Some(m.arg(a, 0));
            break;
        }
        for i in a_n + b_n..total {
            let (from, to) = if round % 2 == 0 { (FREE + 1 + i, FREE + 1 + i + 700) } else { (FREE + 1 + i + 700, FREE + 1 + i) };
            if to < 4096 {
                m.ok(S::CapMove, [SLOT_CAPSPACE, to, from, 0, 0, 0]);
            }
        }
    }
    assert_eq!(a_done, Some(a_n), "an unmap whose place a frame's holder moved meanwhile did not finish with its count");
}

/// Timeouts of waiters in a queue another processor owns are delivered
/// holding every lock, each paid for by its own thread's context: in one
/// pass over the timers, one whose context has used its share has its
/// timeout delivered when the share is back, even after another's took
/// every lock (before, only the first was asked, and it paid for all: the
/// review of audit-kernel-stalls).
#[test]
fn every_timeout_that_takes_every_lock_is_paid_for_by_its_own_context() {
    use kernel_core::thread::{RECEIVING, THROTTLED};
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    // An endpoint, and a thread on processor 1 that receives on it too.
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.spawn(ut, FREE + 20);
    m.move_to(FREE + 20, 1);
    m.settle(1);
    m.on(0);
    // Two contexts on processor 0: the victim's, and the flood's.
    m.retype(ut, T::SchedContext, 0, FREE + 1, 2);
    for g in [FREE + 1, FREE + 2] {
        m.ok(S::SchedConfigure, [g, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
    }
    let bind = |slot: u64, g: u64| {
        let t = m.spawn(ut, slot);
        m.ok(S::ThreadControl, [slot, 2, g, 0, 0, 0]);
        t
    };
    let victim = bind(FREE + 3, FREE + 1);
    let flooder = bind(FREE + 4, FREE + 2);
    let flood = bind(FREE + 5, FREE + 2);
    // Both wait on the endpoint with deadlines, the victim's first.
    let deadline = m.k.hal.now.get() + 1_000;
    assert!(m.sys_as(victim, S::Recv, [FREE, deadline, 0, 0, 0, 0]).blocked);
    assert!(m.sys_as(flooder, S::Recv, [FREE, deadline + 1, 0, 0, 0, 0]).blocked);
    // The queue becomes processor 1's: the last to wait on it is there.
    m.on(1);
    assert!(m.sys_as(m.cap(FREE + 20).unwrap().obj, S::Recv, [FREE, 0, 0, 0, 0, 0]).blocked);
    m.on(0);
    // The flood uses up its context's share of time holding every lock.
    m.k.hal.now_step.set(10_000);
    m.retype(ut, T::Endpoint, 0, FREE + 6, 1);
    let mut calls = 0;
    while m.k.thread_view(flood).state != THROTTLED {
        let call = if m.cap(FREE + 7).is_none() {
            (S::CapCopy, [SLOT_CAPSPACE, FREE + 7, FREE + 6, 15, 0, 0])
        } else {
            (S::CapDelete, [FREE + 7, 0, 0, 0, 0, 0])
        };
        m.sys_as(flood, call.0, call.1);
        calls += 1;
        assert!(calls < 100_000);
    }
    m.k.hal.now_step.set(0);
    // Both deadlines pass, close enough that the flood's share is not back.
    m.k.hal.now.set(m.k.hal.now.get().max(deadline + 2));
    m.tick();
    assert_eq!(m.k.thread_view(victim).state, RUNNABLE, "the victim's timeout was delivered");
    assert_eq!(m.k.thread_view(flooder).state, RECEIVING, "a timeout whose context had used its share rode on another's");
}

/// A fault the thread would not make again where it was (a breakpoint: the
/// saved instruction pointer is past it) is never held back for its
/// context's share: held back, the thread went on past it when the share
/// came back and the fault was lost (the review of audit-kernel-stalls). It
/// is sent, and charged after.
#[test]
fn a_fault_that_does_not_recur_is_sent_when_its_context_has_no_share_left() {
    use kernel_core::thread::{SENDING, THROTTLED};
    use silica_abi::{SCHED_DEFAULT_WEIGHT, SCHED_FAIR};
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.spawn(ut, FREE + 20);
    m.move_to(FREE + 20, 1);
    m.settle(1);
    m.on(0);
    m.retype(ut, T::SchedContext, 0, FREE, 1);
    m.ok(S::SchedConfigure, [FREE, SCHED_FAIR, SCHED_DEFAULT_WEIGHT, 0, 0, SLOT_SCHED]);
    m.retype(ut, T::Endpoint, 0, FREE + 1, 2);
    let bind = |slot: u64| {
        let t = m.spawn(ut, slot);
        m.ok(S::ThreadControl, [slot, 2, FREE, 0, 0, 0]);
        t
    };
    let flood = bind(FREE + 3);
    let x = bind(FREE + 4);
    // Nobody receives on its fault endpoint: the fault takes every lock.
    m.ok(S::ThreadConfigure, [FREE + 4, KEEP, KEEP, FREE + 1, KEEP, 0]);
    m.ok(S::ThreadControl, [FREE + 4, 0, 0, 0, 0, 0]);
    // The context's share of time holding every lock is used up.
    m.k.hal.now_step.set(10_000);
    let mut calls = 0;
    while m.k.thread_view(flood).state != THROTTLED {
        let call = if m.cap(FREE + 5).is_none() {
            (S::CapCopy, [SLOT_CAPSPACE, FREE + 5, FREE + 2, 15, 0, 0])
        } else {
            (S::CapDelete, [FREE + 5, 0, 0, 0, 0, 0])
        };
        m.sys_as(flood, call.0, call.1);
        calls += 1;
        assert!(calls < 100_000);
    }
    m.k.hal.now_step.set(0);
    m.ok(S::ThreadControl, [FREE + 4, 1, 0, 0, 0, 0]);
    m.sys_as(x, S::Yield, [0; 6]);
    while m.k.current() != x {
        m.sys(S::Yield, [0; 6]);
    }
    // A breakpoint (x86 vector 3: a trap, its instruction pointer past it).
    m.fault(Fault::Exception, 3, 0);
    assert_eq!(m.k.thread_view(x).state, SENDING, "a breakpoint was held back, to be lost when the share came back");
}

/// A CapDelete made again only to finish emptying what it destroyed does
/// not look at its slot again: by then another capability may be there
/// (received, or put there by another thread of the CapSpace), which is
/// not the call's to delete (the review of audit-kernel-stalls).
#[test]
fn a_capdelete_made_again_does_not_delete_what_came_into_its_slot() {
    let m = Machine::with_cpus(64, 2);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    m.retype(ut, T::CapSpace, 12, FREE + 1, 1);
    for i in 1..2000 {
        m.ok(S::CapCopy, [FREE + 1, i, FREE, 15, 0, 0]);
    }
    let other = m.spawn(ut, FREE + 2);
    // The CapSpace goes, its emptying stopped beside a waiting processor:
    // the call will be made again to finish it.
    assert!(one_entry_beside_a_waiter(&m, S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]));
    assert!(m.cap(FREE + 1).is_none());
    // Meanwhile another thread of this CapSpace puts a capability there.
    assert_eq!(m.sys_as(other, S::CapCopy, [SLOT_CAPSPACE, FREE + 1, FREE, 15, 7, 0]).status, 0);
    // The CapDelete, made again, finishes without touching it.
    assert!(!one_entry_as_beside_a_waiter(&m, m.root, S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]));
    assert!(m.cap(FREE + 1).is_some(), "the CapDelete made again deleted a capability that came into its slot meanwhile");
}

/// The root task holds the boot secret it was started with: it must not
/// be able to work out from it the key of the identities Inspect shows
/// (with it, identities turn back into physical addresses). Here it tries:
/// it takes its start registers for the entropy the key is made from,
/// makes the key as the kernel does, and predicts an endpoint's identity.
/// Before, the key was made from the very bytes the root task was given
/// (the review of audit-kernel-stalls).
#[test]
fn the_root_task_cannot_work_out_the_identities_key_from_its_secret() {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    m.retype(ut, T::Endpoint, 0, FREE, 1);
    let mut given = [0u8; 32];
    for (i, w) in m.start_args[1..5].iter().enumerate() {
        given[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    let key = kernel_core::ident::key_from(&given);
    let predicted = kernel_core::ident::permute(key, m.cap(FREE).unwrap().obj);
    m.ok(S::Inspect, [SLOT_CAPSPACE, FREE, 0, 0, 0, 0]);
    assert_ne!(m.arg(m.root, 3), predicted, "the root task worked out an identity from its own boot secret");
}
