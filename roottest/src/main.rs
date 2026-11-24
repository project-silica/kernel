//! The kernel's in-OS test program, run as the root task.
//!
//! With `test=<name>` on the kernel command line it runs that test and
//! prints `TEST <name> PASS` or `TEST <name> FAIL <reason>`; with no test it
//! prints `BOOT OK`. Either way it leaves through the emulator's exit device
//! (I/O port 0xF4) using the root task's port capability.
//!
//! `unsafe` here is the test subject itself: deliberate faults, thread entry
//! points and shared test state (`spec/UNSAFE_MODULES.md`).
#![no_std]
#![no_main]

use core::fmt::{self, Write};
use core::sync::atomic::{AtomicU64, Ordering::SeqCst};
#[cfg_attr(target_arch = "x86_64", path = "arch/x86_64.rs")]
#[cfg_attr(target_arch = "aarch64", path = "arch/aarch64.rs")]
mod arch;

use libsys::root::*;
use libsys::*;
use silica_formats::acpi;

const PORTS: Cap = Cap(SLOT_PORTS);
const SPACE: Cap = Cap(SLOT_CAPSPACE);
const ASPACE: Cap = Cap(SLOT_ASPACE);
const RW: Rights = Rights(3);

struct Serial;
impl Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        arch::console_write(s.as_bytes());
        Ok(())
    }
}
macro_rules! println {
    ($($a:tt)*) => {{ let _ = write!(Serial, $($a)*); let _ = Serial.write_str("\r\n"); }};
}

fn exit(ok: bool) -> ! {
    arch::exit_harness(ok);
    loop {
        yield_now();
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    println!("TEST panic FAIL root task panicked: {}", info.message());
    exit(false)
}

type T = core::result::Result<(), &'static str>;

static NEXT_SLOT: AtomicU64 = AtomicU64::new(0);
static UNTYPED: AtomicU64 = AtomicU64::new(0);
static NEXT_VADDR: AtomicU64 = AtomicU64::new(0x1_0000_0000);
static SHARED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn slot() -> u64 {
    NEXT_SLOT.fetch_add(1, SeqCst)
}

/// A new object from the test Untyped.
fn new(ty: ObjectType, bits: u8) -> Cap {
    let s = slot();
    retype(Cap(UNTYPED.load(SeqCst)), ty, bits, SPACE, s, 1).expect("retype");
    Cap(s)
}

/// Address of a fresh region of `bytes`, with an unmapped gap before it.
fn region(bytes: u64) -> usize {
    (NEXT_VADDR.fetch_add(bytes + 0x20_0000, SeqCst) + 0x10_0000) as usize
}

/// A new 64 KiB read-write mapping; returns (frame, address).
fn memory() -> (Cap, usize) {
    let f = new(ObjectType::Frame, 16);
    let at = region(1 << 16);
    map(ASPACE, f, at, RW).expect("map");
    (f, at)
}

/// A thread in this address space running `entry(arg)`, reporting faults to `fault`.
fn spawn(entry: extern "C" fn(u64) -> !, arg: u64, fault: Cap) -> (Cap, usize) {
    let t = new(ObjectType::Thread, 0);
    let (_, stack) = memory();
    thread_configure(t, SPACE, ASPACE, fault, Cap::NONE).expect("configure");
    // SAFETY: `entry` is a function of this program; the stack was just mapped.
    unsafe { thread_start(t, entry as usize, stack + (1 << 16) - 8, arg | t.0 << 32, true) }.expect("start");
    (t, stack)
}

/// What a spawned thread does when it is finished.
fn finish(arg: u64) -> ! {
    let _ = thread_suspend(Cap(arg >> 32));
    loop {
        yield_now();
    }
}

use arch::raw_syscall;

fn wait_for(flag: &AtomicU64, value: u64) -> T {
    for _ in 0..1_000_000 {
        if flag.load(SeqCst) == value {
            return Ok(());
        }
        yield_now();
    }
    Err("timed out waiting for a thread")
}

extern "C" fn set_flag(arg: u64) -> ! {
    SHARED[0].store(arg & 0xFFFF_FFFF, SeqCst);
    finish(arg)
}

fn t_syscall() -> T {
    yield_now();
    if raw_syscall(4) != 0 {
        return Err("yield did not return success");
    }
    if raw_syscall(9999) != 1 {
        return Err("an unknown syscall number did not return BadSyscall");
    }
    if call(Cap(0), MsgInfo(0), [0; 4]) != Err(Error::BadCap) {
        return Err("a call on slot 0 did not return BadCap");
    }
    let (t, _) = spawn(set_flag, 77, Cap::NONE);
    wait_for(&SHARED[0], 77)?;
    // SAFETY: the thread has suspended itself; nothing else uses it.
    unsafe { cap_delete(t) }.map_err(|_| "cannot delete a finished thread")?;
    if thread_resume(t) != Err(Error::BadCap) {
        return Err("a deleted thread capability still works");
    }
    Ok(())
}

extern "C" fn do_fault(arg: u64) -> ! {
    // SAFETY: each arm faults on purpose; that is the test.
    unsafe {
        match arg & 0xFFFF_FFFF {
            1 => core::ptr::write_volatile(0x10 as *mut u8, 1),
            2 | 3 | 6 | 7 | 8 => arch::deliberate(arg & 0xFFFF_FFFF),
            4 => {
                core::ptr::read_volatile(0xFFFF_8000_0010_0000 as *const u8);
            }
            5 => {
                core::ptr::read_volatile(0xFFFF_FFFF_8000_0000 as *const u8);
            }
            _ => {}
        }
    }
    SHARED[1].store(1, SeqCst);
    finish(arg)
}

/// Run `do_fault(what)` in a thread and return the fault message it causes.
fn fault_of(what: u64) -> core::result::Result<Message, &'static str> {
    let ep = new(ObjectType::Endpoint, 0);
    let (t, _) = spawn(do_fault, what, ep);
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000)).map_err(|_| "no fault message arrived")?;
    // SAFETY: the faulted thread is stopped and is not used again.
    unsafe { cap_delete(t) }.map_err(|_| "cannot delete a faulted thread")?;
    // The kernel is still running and still schedules.
    yield_now();
    Ok(m)
}

fn kind(m: &Message) -> u64 {
    m.info.label().wrapping_sub(FAULT_LABEL)
}

fn t_fault_page() -> T {
    let m = fault_of(1)?;
    if kind(&m) != Fault::Page as u64 || m.words[1] != 0x10 {
        return Err("a write to address 0x10 was not reported as a page fault there");
    }
    if m.words[2] != FAULT_WRITE {
        return Err("the fault's access bits do not say write to an unmapped page");
    }
    if m.words[0] < 0x40_0000 || m.words[3] == 0 {
        return Err("the fault message lacks the instruction or stack pointer");
    }
    Ok(())
}

fn t_fault_instruction() -> T {
    let m = fault_of(2)?;
    (kind(&m) == Fault::Instruction as u64).then_some(()).ok_or("an undefined instruction was not reported as an instruction fault")
}

fn t_fault_privileged() -> T {
    for what in [3, 6, 7, 8] {
        let m = fault_of(what)?;
        if !arch::privileged_fault(kind(&m), &m) {
            return Err("a privileged instruction was not reported as the architecture reports one");
        }
    }
    Ok(())
}

fn t_fault_kernel_memory() -> T {
    for what in [4, 5] {
        let m = fault_of(what)?;
        if kind(&m) != Fault::Page as u64 || m.words[1] < 0xFFFF_8000_0000_0000 {
            return Err("reading kernel memory from user mode was not a page fault");
        }
        // Bit 0 set would mean "the page is mapped but protected", which
        // tells user code where kernel memory is; it is, for supervisor pages.
        if SHARED[1].load(SeqCst) != 0 {
            return Err("user code read kernel memory");
        }
    }
    Ok(())
}

#[inline(never)]
#[allow(unconditional_recursion)]
fn recurse(n: u64) -> u64 {
    let mut pad = [0u8; 512];
    // SAFETY: plain volatile accesses to a local, to keep the frame.
    unsafe { core::ptr::write_volatile(pad.as_mut_ptr(), n as u8) };
    let below = recurse(n + 1);
    // SAFETY: as above.
    below + unsafe { core::ptr::read_volatile(pad.as_ptr()) } as u64
}

extern "C" fn overflow(arg: u64) -> ! {
    SHARED[1].store(recurse(0), SeqCst);
    finish(arg)
}

fn t_fault_stack() -> T {
    let ep = new(ObjectType::Endpoint, 0);
    let (t, stack) = spawn(overflow, 0, ep);
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000)).map_err(|_| "no fault message arrived")?;
    // SAFETY: the faulted thread is stopped and is not used again.
    unsafe { cap_delete(t) }.map_err(|_| "cannot delete the thread")?;
    let addr = m.words[1] as usize;
    if kind(&m) != Fault::Page as u64 || addr >= stack || addr < stack - 0x1000 {
        return Err("a stack overflow was not a page fault just below the stack");
    }
    Ok(())
}

static DEMAND_AT: AtomicU64 = AtomicU64::new(0);

extern "C" fn touch(arg: u64) -> ! {
    let p = DEMAND_AT.load(SeqCst) as *mut u64;
    // SAFETY: the page is absent at first; the pager maps it and the access restarts.
    unsafe {
        let old = core::ptr::read_volatile(p);
        core::ptr::write_volatile(p, old + 41);
        SHARED[2].store(core::ptr::read_volatile(p) + 1, SeqCst);
    }
    finish(arg)
}

/// A userspace pager: resolve a fault by mapping a frame and resuming.
fn t_fault_resume() -> T {
    let at = region(1 << 12);
    DEMAND_AT.store(at as u64 + 8, SeqCst);
    let ep = new(ObjectType::Endpoint, 0);
    let (_t, _) = spawn(touch, 0, ep);
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000)).map_err(|_| "no fault message arrived")?;
    if kind(&m) != Fault::Page as u64 || m.words[1] != at as u64 + 8 {
        return Err("the fault address is not the address touched");
    }
    let f = new(ObjectType::Frame, 12);
    map(ASPACE, f, at, RW).map_err(|_| "cannot map the missing page")?;
    reply(MsgInfo::new(0, 1), [FAULT_RESUME, 0, 0, 0]).map_err(|_| "cannot reply to the fault")?;
    wait_for(&SHARED[2], 42)?;
    // The same again with the mapping carried by the reply. A mapping that
    // cannot be made leaves the fault unanswered and reports why.
    let at = region(1 << 12);
    DEMAND_AT.store(at as u64 + 16, SeqCst);
    SHARED[2].store(0, SeqCst);
    let _ = spawn(touch, 0, ep);
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000)).map_err(|_| "no second fault message arrived")?;
    let g = new(ObjectType::Frame, 12);
    if fault_map(ASPACE, f, m.words[1] as usize, RW) != Err(Error::Mapped) {
        return Err("a fault reply mapping an already mapped frame did not fail with Mapped");
    }
    if fault_map(ASPACE, g, m.words[1] as usize, Rights(7)) != Err(Error::BadArg) {
        return Err("a fault reply mapping writable and executable was accepted");
    }
    fault_map(ASPACE, g, m.words[1] as usize, RW).map_err(|_| "cannot map and resume in one reply")?;
    wait_for(&SHARED[2], 42)
}

/// The second page collides after the first would have required a new
/// table. A later mapping needs all three remaining table pages.
fn t_map_rollback() -> T {
    let space = new(ObjectType::AddressSpace, 15);
    let existing = new(ObjectType::Frame, 12);
    let pair = new(ObjectType::Frame, 13);
    let boundary = 1 << 21;
    map(space, existing, boundary, RW).map_err(|_| "cannot map the existing page")?;
    if map(space, pair, boundary - 4096, RW) != Err(Error::Overlap) {
        return Err("a range that overlaps on its second page did not return Overlap");
    }
    map(space, pair, 1 << 39, RW).map_err(|_| "the refused mapping consumed page tables or retained the frame")?;
    Ok(())
}

fn t_w_xor_x() -> T {
    let f = new(ObjectType::Frame, 12);
    let at = region(1 << 12);
    if map(ASPACE, f, at, Rights(7)) != Err(Error::BadArg) {
        return Err("a writable and executable mapping was accepted");
    }
    map(ASPACE, f, at, RW).map_err(|_| "cannot map")?;
    // SAFETY: the page was just mapped read-write; RET is a return instruction.
    unsafe { core::ptr::copy_nonoverlapping(arch::RET.as_ptr(), at as *mut u8, arch::RET.len()) };
    arch::sync_code(at, arch::RET.len());
    // Executing it must fault (not executable) with the instruction-fetch bit.
    let ep = new(ObjectType::Endpoint, 0);
    let t = new(ObjectType::Thread, 0);
    let (_, stack) = memory();
    thread_configure(t, SPACE, ASPACE, ep, Cap::NONE).map_err(|_| "configure")?;
    // SAFETY: the thread is expected to fault on its first instruction.
    unsafe { thread_start(t, at, stack + 0xFFF8, 0, true) }.map_err(|_| "start")?;
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000)).map_err(|_| "executing a data page did not fault")?;
    if kind(&m) != Fault::Page as u64 || m.words[1] != at as u64 || m.words[2] & FAULT_FETCH == 0 {
        return Err("executing a data page was not an instruction-fetch page fault");
    }
    // Remapped read-execute, the same bytes run; and writing them faults.
    // SAFETY: nothing else uses the page.
    unsafe { unmap(f) }.map_err(|_| "unmap")?;
    map(ASPACE, f, at, Rights(5)).map_err(|_| "cannot map read-execute")?;
    // SAFETY: the page holds a `ret` instruction and is executable.
    unsafe { core::mem::transmute::<usize, extern "C" fn()>(at)() };
    DEMAND_AT.store(at as u64, SeqCst);
    let (_t2, _) = spawn(touch, 0, ep);
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000)).map_err(|_| "writing a read-only page did not fault")?;
    (kind(&m) == Fault::Page as u64 && m.words[2] == FAULT_PRESENT | FAULT_WRITE)
        .then_some(())
        .ok_or("writing a read-only page was not a protection fault")
}

extern "C" fn echo_server(arg: u64) -> ! {
    let ep = Cap(SHARED[3].load(SeqCst));
    let Ok(mut m) = recv(ep, Wait::Forever) else { finish(arg) };
    loop {
        if m.info.label() == 99 {
            let _ = reply(MsgInfo(0), [0; 4]);
            finish(arg);
        }
        let w = [m.words[0] * 2, m.words[1] + m.badge, !m.words[2], m.info.words() as u64];
        m = match reply_recv(ep, MsgInfo::new(m.info.label() + 1, 4), w) {
            Ok(m) => m,
            Err(_) => finish(arg),
        };
    }
}

fn t_ipc() -> T {
    let ep = new(ObjectType::Endpoint, 0);
    let badged = slot();
    cap_derive(SPACE, badged, ep, Rights(2), 1000).map_err(|_| "cannot badge an endpoint")?;
    SHARED[3].store(ep.0, SeqCst);
    let _ = spawn(echo_server, 0, Cap::NONE);
    for i in 0..10_000u64 {
        let r = call(Cap(badged), MsgInfo::new(7, 3), [i, i, i, 0]).map_err(|_| "call failed")?;
        if r.words != [i * 2, i + 1000, !i, 3] || r.info.label() != 8 || r.badge != 0 {
            return Err("the reply does not match the request");
        }
    }
    if recv(Cap(badged), Wait::Poll) != Err(Error::NoRights) {
        return Err("a send-only capability could receive");
    }
    call(ep, MsgInfo::new(99, 0), [0; 4]).map_err(|_| "cannot stop the server")?;
    // The server is gone: a call now waits for ever, so only poll.
    if try_send(ep, MsgInfo(0), [0; 4]) != Err(Error::WouldBlock) {
        return Err("a send with no receiver did not report WouldBlock");
    }
    // Destroying the endpoint while a caller waits fails the caller with Gone.
    let ep2 = new(ObjectType::Endpoint, 0);
    SHARED[3].store(ep2.0, SeqCst);
    SHARED[0].store(0, SeqCst);
    let _ = spawn(caller_of_doomed, 0, Cap::NONE);
    wait_for(&SHARED[0], 1)?;
    yield_now();
    // SAFETY: the endpoint is used only by this test.
    unsafe { cap_delete(ep2) }.map_err(|_| "cannot delete the endpoint")?;
    wait_for(&SHARED[0], Error::Gone as u64 + 100)
}

extern "C" fn caller_of_doomed(arg: u64) -> ! {
    SHARED[0].store(1, SeqCst);
    let r = call(Cap(SHARED[3].load(SeqCst)), MsgInfo(0), [0; 4]);
    SHARED[0].store(r.map_or_else(|e| e as u64 + 100, |_| 2), SeqCst);
    finish(arg)
}

fn t_timer() -> T {
    let n = new(ObjectType::Notification, 0);
    for ms in [1u64, 5, 20] {
        let start = CLOCK.now();
        if recv(n, Wait::Until(start + ms * 1_000_000)) != Err(Error::Timeout) {
            return Err("a wait with a deadline did not time out");
        }
        let took = CLOCK.now() - start;
        if took < ms * 1_000_000 {
            return Err("the wait returned before its deadline");
        }
        if took > ms * 1_000_000 + 500_000_000 {
            return Err("the wait overshot its deadline by more than half a second");
        }
    }
    if recv(n, Wait::Until(1000)) != Err(Error::Timeout) {
        return Err("a deadline in the past did not time out at once");
    }
    signal(n).map_err(|_| "signal")?;
    recv(n, Wait::Poll).map(|_| ()).map_err(|_| "a pending signal was not received")
}

extern "C" fn spin(arg: u64) -> ! {
    let c = &SHARED[(arg & 1) as usize];
    loop {
        c.fetch_add(1, SeqCst);
    }
}

fn t_preempt() -> T {
    SHARED[0].store(0, SeqCst);
    SHARED[1].store(0, SeqCst);
    let (a, _) = spawn(spin, 0, Cap::NONE);
    let (b, _) = spawn(spin, 1, Cap::NONE);
    // Neither thread ever yields; only the timer lets this one run again.
    let n = new(ObjectType::Notification, 0);
    let _ = recv(n, Wait::Until(CLOCK.now() + 100_000_000));
    thread_suspend(a).map_err(|_| "suspend")?;
    thread_suspend(b).map_err(|_| "suspend")?;
    let (x, y) = (SHARED[0].load(SeqCst), SHARED[1].load(SeqCst));
    if x == 0 || y == 0 {
        return Err("a thread that never yields starved another");
    }
    let _ = recv(n, Wait::Until(CLOCK.now() + 20_000_000));
    if SHARED[0].load(SeqCst) != x || SHARED[1].load(SeqCst) != y {
        return Err("a suspended thread kept running");
    }
    Ok(())
}

use arch::hold_in_vector_registers;

extern "C" fn fpu_worker(arg: u64) -> ! {
    let id = arg & 0xFFFF_FFFF;
    let slice = CLOCK_CELL[0].load(SeqCst) / 500; // 2 ms: several per time slice
    let mut bad = 0u64;
    for round in 0..60u64 {
        let v = (id + 1) << 32 | round;
        if hold_in_vector_registers(v, slice) != [v, v, v] {
            bad += 1;
        }
    }
    SHARED[id as usize].store(1 + bad, SeqCst);
    finish(arg)
}

/// Set by `fpu_secret_holder` when it sees control state that is not its own.
static CONTROLS_LEAKED: AtomicU64 = AtomicU64::new(0);

extern "C" fn fpu_secret_holder(arg: u64) -> ! {
    // Fill the vector registers with a recognisable value and yield, for ever.
    // Also record whether, after a yield, its control words or thread
    // pointer were ever anything but its own (the defaults): another
    // thread's leaking in.
    loop {
        if arch::yield_holding_secret(0x5EC2_E75E_C2E7_u64 | arg << 48) != arch::CONTROLS_DEFAULT {
            CONTROLS_LEAKED.store(1, SeqCst);
        }
        if SHARED[2].load(SeqCst) != 0 {
            finish(arg)
        }
    }
}

fn t_fpu() -> T {
    // 1. Preempted threads keep their vector registers: three threads spin
    // without a system call, so only the timer switches between them.
    SHARED[0].store(0, SeqCst);
    SHARED[1].store(0, SeqCst);
    let _ = spawn(fpu_worker, 0, Cap::NONE);
    let _ = spawn(fpu_worker, 1, Cap::NONE);
    let slice = CLOCK_CELL[0].load(SeqCst) / 500;
    for round in 0..60u64 {
        let v = 0x7777 << 32 | round;
        if hold_in_vector_registers(v, slice) != [v, v, v] {
            return Err("the main thread's vector registers changed while it was preempted");
        }
    }
    let n = new(ObjectType::Notification, 0);
    for flag in [&SHARED[0], &SHARED[1]] {
        for _ in 0..200 {
            if flag.load(SeqCst) != 0 {
                break;
            }
            let _ = recv(n, Wait::Until(CLOCK.now() + 5_000_000));
        }
        if flag.load(SeqCst) != 1 {
            return Err("SSE or AVX registers changed across a preemption (or the worker did not finish)");
        }
    }
    // 2. A system call does not keep vector registers, and what comes back
    // is never another thread's: after yielding to a thread that fills its
    // registers with a secret, this thread's are clear.
    SHARED[2].store(0, SeqCst);
    let _ = spawn(fpu_secret_holder, 1, Cap::NONE);
    for _ in 0..200 {
        let (a, b, x87) = arch::vectors_after_yield();
        if a != 0 || b != 0 {
            SHARED[2].store(1, SeqCst);
            return Err("after a system call a thread saw another thread's vector registers");
        }
        if x87 {
            SHARED[2].store(1, SeqCst);
            return Err("after a system call a thread saw another thread's x87 stack");
        }
    }
    // 3. Floating-point control words and the thread pointer are each
    // thread's own across a switch, both ways: this thread gets its own
    // back, and the thread it yielded to never sees them.
    for _ in 0..20 {
        let (set, got) = arch::fp_control_across_yield();
        if got != set {
            SHARED[2].store(1, SeqCst);
            return Err(
                "the floating-point control words or the thread pointer were not preserved across a system call that switched threads",
            );
        }
    }
    let _ = arch::vectors_after_yield();
    SHARED[2].store(1, SeqCst);
    if CONTROLS_LEAKED.load(SeqCst) != 0 {
        return Err("a thread saw another thread's floating-point control words or thread pointer after a switch");
    }
    Ok(())
}

fn t_untyped() -> T {
    let pool = new(ObjectType::Untyped, 20);
    let s = slot();
    let at = region(1 << 16);
    for round in 0..50u64 {
        retype(pool, ObjectType::Frame, 16, SPACE, s, 1).map_err(|_| "retype from a revoked Untyped failed")?;
        map(ASPACE, Cap(s), at, RW).map_err(|_| "map")?;
        let p = at as *mut u64;
        // SAFETY: 64 KiB mapped read-write at `at` just above.
        unsafe {
            for i in 0..8192 {
                if core::ptr::read_volatile(p.add(i)) != 0 {
                    return Err("memory from a reused Untyped was not zero");
                }
            }
            for i in 0..8192 {
                core::ptr::write_volatile(p.add(i), round + 1);
            }
            cap_revoke(pool).map_err(|_| "revoke")?;
        }
    }
    // Exhaustion is an error for the caller, and nothing else is affected.
    if retype(pool, ObjectType::Frame, 21, SPACE, s, 1) != Err(Error::NoMemory) {
        return Err("an oversized retype did not return NoMemory");
    }
    let mut made = 0;
    while retype(pool, ObjectType::Frame, 12, SPACE, slot(), 1).is_ok() {
        made += 1;
    }
    if made != 256 {
        return Err("a 1 MiB Untyped did not yield exactly 256 pages");
    }
    let _other = new(ObjectType::Endpoint, 0);
    Ok(())
}

fn t_caps() -> T {
    let n = new(ObjectType::Notification, 0);
    let (a, b) = (slot(), slot());
    cap_derive(SPACE, a, n, Rights::ALL, 1).map_err(|_| "copy")?;
    cap_derive(SPACE, b, Cap(a), Rights::WRITE, 0).map_err(|_| "copy of a copy")?;
    if cap_derive(SPACE, b, n, Rights::ALL, 0) != Err(Error::SlotFull) {
        return Err("copying into a full slot was accepted");
    }
    signal(Cap(b)).map_err(|_| "signal through a derived capability")?;
    if recv(n, Wait::Poll).map(|m| m.badge) != Ok(1) {
        return Err("the badge of a derived capability was not delivered");
    }
    // SAFETY: only capabilities made by this test derive from `n`.
    unsafe { cap_revoke(n) }.map_err(|_| "revoke")?;
    if signal(Cap(a)) != Err(Error::BadCap) || signal(Cap(b)) != Err(Error::BadCap) {
        return Err("a revoked capability still works");
    }
    signal(n).map_err(|_| "the revoked-from capability itself must survive")?;
    if cap_derive(SPACE, a, Cap(UNTYPED.load(SeqCst)), Rights::ALL, 0) != Err(Error::InUse) {
        return Err("an Untyped was copied");
    }
    // A narrowed range of interrupt lines reaches only its lines (an
    // acknowledgement checks the range and changes nothing on an unbound line).
    cap_derive(SPACE, a, Cap(SLOT_IRQS), Rights::ALL, 4 | 5 << 32).map_err(|_| "narrow interrupt lines")?;
    if irq_ack(Cap(a), 5) != Err(Error::BadArg) || irq_ack(Cap(a), 4).is_err() {
        return Err("a narrowed interrupt-line capability did not confine its lines");
    }
    // A narrowed port range reaches only its ports.
    if arch::HAS_PORTS {
        cap_derive(SPACE, b, PORTS, Rights::ALL, 0x3F8 | 0x400 << 32).map_err(|_| "narrow ports")?;
        if port_in(Cap(b), 0xF4, 1) != Err(Error::BadArg) || port_in(Cap(b), 0x3FD, 1).is_err() {
            return Err("a narrowed port capability did not confine port access");
        }
    }
    Ok(())
}

static SERVER_BUFFER: AtomicU64 = AtomicU64::new(0);

/// A thread with its own IPC buffer, mapped at the returned address (also
/// left in `SERVER_BUFFER` for the thread to find).
fn spawn_with_buffer(entry: extern "C" fn(u64) -> !, arg: u64) -> (Cap, usize) {
    let t = new(ObjectType::Thread, 0);
    let (_, stack) = memory();
    let frame = new(ObjectType::Frame, 12);
    let at = region(1 << 12);
    map(ASPACE, frame, at, RW).expect("map buffer");
    thread_configure(t, SPACE, ASPACE, Cap::NONE, frame).expect("configure");
    // SAFETY: `entry` is a function of this program; the stack was just mapped.
    SERVER_BUFFER.store(at as u64, SeqCst);
    unsafe { thread_start(t, entry as usize, stack + (1 << 16) - 8, arg | t.0 << 32, true) }.expect("start");
    (t, at)
}

extern "C" fn cap_server(arg: u64) -> ! {
    // SAFETY: the address of the buffer frame this thread was configured with.
    let buf = unsafe { IpcBuffer::new(SERVER_BUFFER.load(SeqCst) as usize) };
    let ep = Cap(SHARED[3].load(SeqCst));
    let slot_for_gift = SHARED[2].load(SeqCst);
    buf.accept(0, slot_for_gift);
    let Ok(m) = recv(ep, Wait::Forever) else { finish(arg) };
    // Use what was sent: signal the notification, then say how many arrived.
    let got = m.info.caps() as u64;
    let _ = signal(Cap(slot_for_gift));
    let _ = reply(MsgInfo::new(0, 2), [got, buf.word(5), 0, 0]);
    finish(arg)
}

/// A capability lent in a message works for the receiver until the sender revokes it.
fn t_ipc_caps() -> T {
    // SAFETY: the root thread's buffer, mapped by the kernel.
    let buf = unsafe { IpcBuffer::new(ROOT_BUFFER.load(SeqCst) as usize) };
    let ep = new(ObjectType::Endpoint, 0);
    let n = new(ObjectType::Notification, 0);
    let gift = slot();
    SHARED[3].store(ep.0, SeqCst);
    SHARED[2].store(gift, SeqCst);
    let _ = spawn_with_buffer(cap_server, 0);
    buf.pass(0, n, Pass::Lend);
    buf.set_word(5, 0xFEED);
    let r = call(ep, MsgInfo::with_caps(1, 6, 1), [0; 4]).map_err(|_| "the call carrying a capability failed")?;
    if r.words[0] != 1 {
        return Err("the receiver did not get the capability");
    }
    if r.words[1] != 0xFEED {
        return Err("a message word beyond the registers did not arrive through the buffers");
    }
    if recv(n, Wait::Poll).is_err() {
        return Err("the receiver could not use the capability it was lent");
    }
    // SAFETY: only the lent capability derives from `n`.
    unsafe { cap_revoke(n) }.map_err(|_| "revoke")?;
    if signal(Cap(gift)) != Err(Error::BadCap) {
        return Err("a lent capability survived the lender's revoke");
    }
    // Without the GRANT right on the endpoint nothing is passed.
    let plain = slot();
    cap_derive(SPACE, plain, ep, Rights(3), 0).map_err(|_| "derive")?;
    let _ = spawn_with_buffer(cap_server, 0);
    buf.pass(0, n, Pass::Duplicate);
    let r = call(Cap(plain), MsgInfo::with_caps(1, 0, 1), [0; 4]).map_err(|_| "call")?;
    (r.words[0] == 0).then_some(()).ok_or("a capability passed through an endpoint capability without GRANT")
}

extern "C" fn parking_server(arg: u64) -> ! {
    let ep = Cap(SHARED[3].load(SeqCst));
    let (n1, n2) = (Cap(SHARED[1].load(SeqCst)), Cap(SHARED[2].load(SeqCst)));
    // Take two calls before answering either, then answer the second first.
    let (Ok(a), Ok(b)) = (recv_parking(ep, Wait::Forever, n1), recv_parking(ep, Wait::Forever, n2)) else { finish(arg) };
    let _ = answer(n2, MsgInfo::new(0, 1), [b.words[0] + 1, 0, 0, 0]);
    let _ = answer(n1, MsgInfo::new(0, 1), [a.words[0] + 1, 0, 0, 0]);
    finish(arg)
}

extern "C" fn parked_client(arg: u64) -> ! {
    let r = call(Cap(SHARED[3].load(SeqCst)), MsgInfo::new(0, 1), [500, 0, 0, 0]);
    SHARED[0].store(r.map_or(1, |m| m.words[0]), SeqCst);
    finish(arg)
}

/// A server holds two calls at once and answers them out of order.
fn t_ipc_parked() -> T {
    let ep = new(ObjectType::Endpoint, 0);
    SHARED[3].store(ep.0, SeqCst);
    SHARED[1].store(new(ObjectType::Notification, 0).0, SeqCst);
    SHARED[2].store(new(ObjectType::Notification, 0).0, SeqCst);
    SHARED[0].store(0, SeqCst);
    let _ = spawn(parking_server, 0, Cap::NONE);
    let _ = spawn(parked_client, 0, Cap::NONE);
    let r = call(ep, MsgInfo::new(0, 1), [700, 0, 0, 0]).map_err(|_| "a parked call failed")?;
    wait_for(&SHARED[0], 501).or_else(|_| wait_for(&SHARED[0], 701))?;
    let other = SHARED[0].load(SeqCst);
    // Each caller got the answer to its own call.
    match (r.words[0], other) {
        (701, 501) | (501, 701) => Ok(()),
        _ => Err("parked calls were answered with the wrong replies"),
    }
}

/// A device interrupt reaches userspace: the serial port's "transmitter
/// empty" interrupt (ISA line 4), delivered as a notification.
fn t_irq() -> T {
    if !arch::HAS_PORTS {
        return t_irq_mmio();
    }
    const LINES: Cap = Cap(SLOT_IRQS);
    let n = new(ObjectType::Notification, 0);
    let badged = slot();
    cap_derive(SPACE, badged, n, Rights::ALL, 0x40).map_err(|_| "derive")?;
    if irq_bind(PORTS, 4, Cap(badged), 0) != Err(Error::WrongType) {
        return Err("a port capability was accepted as an interrupt capability");
    }
    irq_bind(LINES, 4, Cap(badged), 0).map_err(|_| "cannot bind interrupt line 4")?;
    if recv(n, Wait::Until(CLOCK.now() + 20_000_000)) != Err(Error::Timeout) {
        return Err("an interrupt arrived before the device was asked for one");
    }
    // OUT2 gates the UART's interrupt onto the line; then enable "transmitter empty".
    let mcr = port_in(PORTS, 0x3FC, 1).unwrap_or(0);
    port_out(PORTS, 0x3FC, 1, mcr | 0x08).map_err(|_| "port")?;
    port_out(PORTS, 0x3F9, 1, 0x02).map_err(|_| "port")?;
    let m = recv(n, Wait::Until(CLOCK.now() + 1_000_000_000));
    port_out(PORTS, 0x3F9, 1, 0).map_err(|_| "port")?;
    let _ = port_in(PORTS, 0x3FA, 1); // reading the cause clears it in the device
    match m {
        Ok(m) if m.badge == 0x40 => {}
        Ok(_) => return Err("the interrupt was delivered with the wrong badge"),
        Err(_) => return Err("the serial port's interrupt never arrived as a notification"),
    }
    irq_ack(LINES, 4).map_err(|_| "ack")?;
    // Unbound, the line is quiet again even if the device asks.
    irq_unbind(LINES, 4).map_err(|_| "unbind")?;
    port_out(PORTS, 0x3F9, 1, 0x02).map_err(|_| "port")?;
    let quiet = recv(n, Wait::Until(CLOCK.now() + 20_000_000)) == Err(Error::Timeout);
    port_out(PORTS, 0x3F9, 1, 0).map_err(|_| "port")?;
    quiet.then_some(()).ok_or("an unbound interrupt line still signalled")
}

/// [`t_irq`] where devices are in memory space: a PL011 UART the firmware
/// describes (ACPI DSDT, `ARMH0011`) other than the kernel's console (SPCR),
/// found, mapped and driven from here, its "transmit" interrupt bound to a
/// notification.
fn t_irq_mmio() -> T {
    const LINES: Cap = Cap(SLOT_IRQS);
    // SAFETY: INFO holds the kernel's start-up record.
    let rsdp = unsafe { (*(INFO.load(SeqCst) as *const RootInfo)).rsdp };
    let mut read = |a: u64, b: &mut [u8]| read_phys(a, b);
    let (spcr, _) = acpi::find_table(&mut read, rsdp, b"SPCR").ok_or("the firmware has no SPCR table")?;
    let mut w = [0u8; 8];
    if !read_phys(spcr + 44, &mut w) {
        return Err("cannot read the SPCR table");
    }
    let console = u64::from_le_bytes(w);
    let (dsdt, len) = acpi::dsdt(&mut read, rsdp).ok_or("the firmware has no DSDT")?;
    // SAFETY: only this test uses the buffer, from one thread.
    let buf = unsafe { &mut *(&raw mut DSDT) };
    let aml = buf.get_mut(..len as usize).ok_or("the DSDT is larger than 128 KiB")?;
    if !read_phys(dsdt, aml) {
        return Err("cannot read the DSDT");
    }
    let dev = acpi::devices(aml, b"ARMH0011")
        .find(|r| r.mem.0 != console && r.mem.0 != 0 && r.irq.is_some_and(|i| i >= 32))
        .ok_or("the firmware describes no PL011 UART besides the console (the harness gives the machine a second one)")?;
    let line = dev.irq.unwrap_or(0) - 32;
    println!("irq: a PL011 at {:#x} (the console is at {console:#x}) on interrupt line {line}", dev.mem.0);
    let uart = phys_window(dev.mem.0).ok_or("the second UART's registers are not in memory handed to the root task")?;
    let reg = |off: u64| (uart + off) as *mut u32;
    // SAFETY: the UART's registers, mapped as device memory just above.
    let (get, set) = (|off| unsafe { core::ptr::read_volatile(reg(off)) }, |off, v| unsafe { core::ptr::write_volatile(reg(off), v) });
    const DR: u64 = 0x00;
    const LCR_H: u64 = 0x2C;
    const CR: u64 = 0x30;
    const IMSC: u64 = 0x38;
    const MIS: u64 = 0x40;
    const ICR: u64 = 0x44;
    const TX: u32 = 1 << 5;
    set(IMSC, 0);
    set(ICR, 0x7FF);
    set(LCR_H, 0x60); // 8 bits, no FIFO: a written byte leaves at once
    set(CR, 0x101); // enabled, transmitting

    let n = new(ObjectType::Notification, 0);
    let badged = slot();
    cap_derive(SPACE, badged, n, Rights::ALL, 0x40).map_err(|_| "derive")?;
    irq_bind(LINES, line, Cap(badged), 0).map_err(|_| "cannot bind the second UART's interrupt line")?;
    if recv(n, Wait::Until(CLOCK.now() + 20_000_000)) != Err(Error::Timeout) {
        return Err("an interrupt arrived before the device was asked for one");
    }
    set(IMSC, TX);
    set(DR, b'.' as u32);
    let m = recv(n, Wait::Until(CLOCK.now() + 1_000_000_000));
    let cause = get(MIS);
    set(IMSC, 0);
    set(ICR, 0x7FF);
    match m {
        Ok(m) if m.badge == 0x40 => {}
        Ok(_) => return Err("the interrupt was delivered with the wrong badge"),
        Err(_) => return Err("the UART's interrupt never arrived as a notification"),
    }
    if cause & TX == 0 {
        return Err("the notification came, but the UART does not say it raised its transmit interrupt");
    }
    irq_ack(LINES, line).map_err(|_| "ack")?;
    // Unbound, the line is quiet again even if the device asks.
    irq_unbind(LINES, line).map_err(|_| "unbind")?;
    set(IMSC, TX);
    set(DR, b'.' as u32);
    let quiet = recv(n, Wait::Until(CLOCK.now() + 20_000_000)) == Err(Error::Timeout);
    set(IMSC, 0);
    set(ICR, 0x7FF);
    quiet.then_some(()).ok_or("an unbound interrupt line still signalled")
}

extern "C" fn wait_for_ever(arg: u64) -> ! {
    let _ = recv(Cap(SHARED[3].load(SeqCst)), Wait::Forever);
    finish(arg)
}

/// The owner's view: every capability of a CapSpace and the state of every
/// thread, read through `Inspect` and printed as a debug console would.
fn t_inspect() -> T {
    let ep = new(ObjectType::Endpoint, 0);
    SHARED[3].store(ep.0, SeqCst);
    let (t, _) = spawn(wait_for_ever, 0, Cap::NONE);
    yield_now();
    let me = inspect_thread(Cap(SLOT_THREAD)).map_err(|_| "cannot inspect the calling thread")?;
    if me.state != THREAD_RUNNABLE {
        return Err("the calling thread is not reported runnable");
    }
    let ep_info = inspect_slot(SPACE, ep.0).map_err(|_| "inspect")?.ok_or("the endpoint's slot is reported empty")?;
    let waiter = inspect_thread(t).map_err(|_| "cannot inspect another thread")?;
    if waiter.state != THREAD_RECEIVING || waiter.waits_on != ep_info.object {
        return Err("a thread waiting on an endpoint is not reported as receiving on it");
    }
    // The same thread seen through the CapSpace that holds it.
    let through = inspect_thread_in(SPACE, t.0).map_err(|_| "cannot inspect a thread through its CapSpace")?;
    if (through.state, through.cpu, through.waits_on) != (waiter.state, waiter.cpu, waiter.waits_on) {
        return Err("a thread seen through its CapSpace differs from the same thread seen directly");
    }
    if inspect_thread_in(SPACE, ep.0) != Err(Error::WrongType) {
        return Err("inspecting a thread in a slot that holds something else was accepted");
    }
    if inspect_thread_in(t, 1) != Err(Error::BadArg) {
        return Err("inspecting a thread with arguments meant for a CapSpace was accepted");
    }
    if libsys::raw(Syscall::Inspect, [SPACE.0, t.0, 2, 0, 0, 0]).map(|_| ()) != Err(Error::BadArg) {
        return Err("an unknown inspection was accepted");
    }
    // Walk the whole CapSpace, as the console's `caps` command would.
    let slots = 4096;
    let (mut caps, mut threads) = (0u64, 0u64);
    let mut by_type = [0u64; 11];
    for i in 1..slots {
        let Some(info) = inspect_slot(SPACE, i).map_err(|_| "inspect failed on a valid slot")? else { continue };
        caps += 1;
        by_type[info.ty as usize] += 1;
        if info.ty == ObjectType::Thread {
            threads += 1;
            let th = inspect_thread(Cap(i)).map_err(|_| "inspect thread")?;
            println!("thread in slot {i}: state {} cpu {} ip {:#x} waits on {:#x}", th.state, th.cpu, th.ip, th.waits_on);
        }
    }
    println!(
        "capabilities: {caps} (untyped {}, capspace {}, thread {}, address space {}, frame {}, endpoint {}, notification {}, hw {})",
        by_type[1], by_type[2], by_type[3], by_type[4], by_type[5], by_type[6], by_type[7], by_type[9]
    );
    if threads < 2 || by_type[1] == 0 || by_type[9] != arch::HW_CAPS {
        return Err("the walk of the CapSpace missed capabilities that are there");
    }
    if inspect_slot(SPACE, slots) != Err(Error::BadCap) || inspect_slot(ep, 1) != Err(Error::WrongType) {
        return Err("inspecting outside a CapSpace, or through something that is not one, was accepted");
    }
    Ok(())
}

// ---- more than one processor ---------------------------------------------

static CPUS: AtomicU64 = AtomicU64::new(1);
/// `RootInfo::entropy` as four words, and `entropy_sources`.
static ENTROPY: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// The root task's start-up record, for tests that read the firmware's tables.
static INFO: AtomicU64 = AtomicU64::new(0);
/// Where each device or firmware Untyped is mapped (address + 1; 0: not yet).
static WINDOWS: [AtomicU64; MAX_UNTYPED] = [const { AtomicU64::new(0) }; MAX_UNTYPED];

/// The virtual address of physical address `phys` in device or firmware
/// memory: the whole Untyped holding it becomes one Frame, mapped once.
fn phys_window(phys: u64) -> Option<u64> {
    // SAFETY: INFO holds the kernel's start-up record, read-only for ever.
    let info = unsafe { &*(INFO.load(SeqCst) as *const RootInfo) };
    let i = (0..info.untyped_count as usize).find(|&i| {
        let d = info.untyped[i];
        (d.kind == KIND_DEVICE || d.kind == KIND_FIRMWARE) && phys >= d.base && phys - d.base < 1 << d.bits
    })?;
    let d = info.untyped[i];
    let mut at = WINDOWS[i].load(SeqCst);
    if at == 0 {
        let f = slot();
        retype(Cap(SLOT_FIRST_UNTYPED + i as u64), ObjectType::Frame, d.bits, SPACE, f, 1).ok()?;
        let size = 1u64 << d.bits;
        let base = (region(2 * size) as u64 + size - 1) & !(size - 1);
        map(ASPACE, Cap(f), base as usize, RW).ok()?;
        at = base + 1;
        WINDOWS[i].store(at, SeqCst);
    }
    Some(at - 1 + (phys - d.base))
}

/// Read physical memory through [`phys_window`], a byte at a time (it may
/// be mapped as device memory, where wider accesses must be aligned).
fn read_phys(at: u64, buf: &mut [u8]) -> bool {
    for (i, b) in buf.iter_mut().enumerate() {
        let Some(v) = phys_window(at + i as u64) else { return false };
        // SAFETY: phys_window mapped this byte readable.
        *b = unsafe { core::ptr::read_volatile(v as *const u8) };
    }
    true
}

/// A copy of the DSDT, for [`t_irq`] on machines without I/O ports.
static mut DSDT: [u8; 1 << 17] = [0; 1 << 17];
static ROOT_BUFFER: AtomicU64 = AtomicU64::new(0);

/// Like `spawn`, on processor `cpu`.
fn spawn_on(cpu: usize, entry: extern "C" fn(u64) -> !, arg: u64, fault: Cap) -> (Cap, usize) {
    let t = new(ObjectType::Thread, 0);
    let (_, stack) = memory();
    thread_configure(t, SPACE, ASPACE, fault, Cap::NONE).expect("configure");
    // SAFETY: `entry` is a function of this program; the stack was just mapped.
    unsafe { thread_start(t, entry as usize, stack + (1 << 16) - 8, arg | t.0 << 32, false) }.expect("start");
    thread_set_cpu(t, cpu).expect("move to processor");
    thread_resume(t).expect("resume");
    (t, stack)
}

fn need_cpus(n: u64) -> T {
    if CPUS.load(SeqCst) < n { Err("this test needs more processors than the machine has") } else { Ok(()) }
}

/// Whether a long test is still getting somewhere. A hang is no progress,
/// on a fast machine or a slow one (emulated aarch64 on a hosted runner is
/// 3 to 30 times slower than on the reference host, 2026-11-24), so a test
/// that can count its progress judges that rather than a total time. It
/// says how far it got every ten seconds: the harness ends a guest that
/// says nothing for a minute (`tools/qemu.sh --quiet`), which finds a hung
/// kernel as soon on a slow machine as on a fast one.
struct Progress {
    what: &'static str,
    last: u64,
    changed: u64,
    said: u64,
}

impl Progress {
    /// Longest without progress before the test fails.
    const STALL_NS: u64 = 30_000_000_000;
    /// How often it says how far it got.
    const SAY_NS: u64 = 10_000_000_000;

    fn new(what: &'static str) -> Progress {
        let now = CLOCK.now();
        Progress { what, last: 0, changed: now, said: now }
    }

    /// `done` is how far the test has got: an error once it has not moved
    /// for [`Self::STALL_NS`].
    fn check(&mut self, done: u64) -> T {
        let now = CLOCK.now();
        if done != self.last {
            self.last = done;
            self.changed = now;
        }
        if now - self.said >= Self::SAY_NS {
            println!("{}: {done} so far", self.what);
            self.said = now;
        }
        if now - self.changed > Self::STALL_NS {
            println!("{}: {done}, and none for 30 s", self.what);
            return Err("no progress for 30 seconds");
        }
        Ok(())
    }
}

fn sleep_ms(ms: u64) {
    static NAP: AtomicU64 = AtomicU64::new(0);
    if NAP.load(SeqCst) == 0 {
        NAP.store(new(ObjectType::Notification, 0).0, SeqCst);
    }
    let _ = recv(Cap(NAP.load(SeqCst)), Wait::Until(CLOCK.now() + ms * 1_000_000));
}

static RAN: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

extern "C" fn count_for_ever(arg: u64) -> ! {
    let c = &RAN[(arg & 63) as usize];
    loop {
        c.fetch_add(1, SeqCst);
    }
}

/// Every processor runs threads, and they run at the same time.
fn t_smp_run() -> T {
    need_cpus(2)?;
    let n = CPUS.load(SeqCst).min(64) as usize;
    let mut threads = [Cap::NONE; 64];
    for (cpu, t) in threads.iter_mut().enumerate().take(n).skip(1) {
        *t = spawn_on(cpu, count_for_ever, cpu as u64, Cap::NONE).0;
    }
    // This thread never yields its processor: the others advance only if
    // their own processors run them.
    let end = CLOCK.now() + 300_000_000;
    while CLOCK.now() < end {
        if (1..n).all(|c| RAN[c].load(SeqCst) > 1000) {
            break;
        }
        core::hint::spin_loop();
    }
    for c in 1..n {
        if RAN[c].load(SeqCst) <= 1000 {
            return Err("a thread placed on another processor did not run while this one was busy");
        }
    }
    for t in &threads[1..n] {
        thread_suspend(*t).map_err(|_| "cannot suspend a thread on another processor")?;
    }
    let before: [u64; 64] = core::array::from_fn(|c| RAN[c].load(SeqCst));
    sleep_ms(20);
    for c in 1..n {
        if RAN[c].load(SeqCst) != before[c] {
            return Err("a thread suspended from another processor kept running");
        }
    }
    Ok(())
}

/// Calls to a server on another processor, then on the same one.
fn t_smp_ipc() -> T {
    need_cpus(2)?;
    let ep = new(ObjectType::Endpoint, 0);
    SHARED[3].store(ep.0, SeqCst);
    let _ = spawn_on(1, echo_server, 0, Cap::NONE);
    for i in 0..20_000u64 {
        let r = call(ep, MsgInfo::new(7, 3), [i, i, i, 0]).map_err(|_| "a call to a server on another processor failed")?;
        if r.words != [i * 2, i, !i, 3] {
            return Err("the reply from another processor does not match the request");
        }
    }
    call(ep, MsgInfo::new(99, 0), [0; 4]).map_err(|_| "cannot stop the server")?;
    Ok(())
}

static VICTIM: AtomicU64 = AtomicU64::new(0);

extern "C" fn read_for_ever(arg: u64) -> ! {
    let p = VICTIM.load(SeqCst) as *const u64;
    loop {
        // SAFETY: the page is mapped at first; when it is unmapped under
        // this thread, the fault is the test.
        let v = unsafe { core::ptr::read_volatile(p) };
        SHARED[0].store(v, SeqCst);
        let _ = arg;
    }
}

/// Unmapping a page takes it away from a thread using it on another processor at once.
fn t_smp_shootdown() -> T {
    need_cpus(2)?;
    let (frame, at) = memory();
    // SAFETY: just mapped read-write.
    unsafe { core::ptr::write_volatile(at as *mut u64, 0xC0FFEE) };
    VICTIM.store(at as u64, SeqCst);
    SHARED[0].store(0, SeqCst);
    let ep = new(ObjectType::Endpoint, 0);
    let (t, _) = spawn_on(1, read_for_ever, 0, ep);
    wait_for(&SHARED[0], 0xC0FFEE)?;
    // SAFETY: only the reader thread uses the page, and losing it is the test.
    unsafe { unmap(frame) }.map_err(|_| "unmap")?;
    // From here on the other processor must not read the page again: the
    // unmap returned only after it dropped its cached translation.
    SHARED[0].store(1, SeqCst);
    let m =
        recv(ep, Wait::Until(CLOCK.now() + 1_000_000_000)).map_err(|_| "the reader on another processor never faulted after the unmap")?;
    if kind(&m) != Fault::Page as u64 || m.words[1] != at as u64 {
        return Err("the reader's fault is not a page fault at the unmapped address");
    }
    if SHARED[0].load(SeqCst) != 1 {
        return Err("another processor read a page after unmap returned");
    }
    // SAFETY: the faulted reader is stopped and not used again.
    unsafe { cap_delete(t) }.map_err(|_| "delete")?;
    Ok(())
}

static ROUNDS: AtomicU64 = AtomicU64::new(0);
/// Round trips done so far by the stress clients, in steps of 1024: the
/// test's progress (ROUNDS is each client's count, added once it is done).
static UNDERWAY: AtomicU64 = AtomicU64::new(0);
static LOST: AtomicU64 = AtomicU64::new(0);
static RETRIES: AtomicU64 = AtomicU64::new(0);

extern "C" fn stress_client(arg: u64) -> ! {
    let ep = Cap(SHARED[3].load(SeqCst));
    let want = arg & 0xFFFF_FFFF;
    let (mut done, mut told) = (0u64, 0u64);
    while done < want {
        // Round trips done, told every 1024 (a retry is not one).
        if done % 1024 == 0 && done > 0 && done != told {
            UNDERWAY.fetch_add(1024, SeqCst);
            told = done;
        }
        match call(ep, MsgInfo::new(7, 3), [done, 5, 6, 0]) {
            Ok(r) if r.words[0] == done * 2 => done += 1,
            Ok(_) => {
                LOST.fetch_add(1, SeqCst);
                done += 1;
            }
            // The endpoint capability is being revoked and granted again
            // under us: the documented errors, then try again.
            Err(Error::BadCap) | Err(Error::Gone) => {
                RETRIES.fetch_add(1, SeqCst);
                yield_now()
            }
            Err(_) => {
                LOST.fetch_add(1 << 32, SeqCst);
                done += 1;
            }
        }
    }
    ROUNDS.fetch_add(done, SeqCst);
    finish(arg)
}

extern "C" fn stress_server(arg: u64) -> ! {
    let ep = Cap(arg & 0xFFFF_FFFF);
    let mut m = loop {
        if let Ok(m) = recv(ep, Wait::Forever) {
            break m;
        }
    };
    loop {
        let w = [m.words[0] * 2, 0, 0, 0];
        m = loop {
            match reply_recv(ep, MsgInfo::new(8, 1), w) {
                Ok(m) => break m,
                Err(_) => {
                    if let Ok(m) = recv(ep, Wait::Forever) {
                        break m;
                    }
                }
            }
        };
    }
}

/// A million round trips between two threads on one processor while other
/// processors are busy, with nothing lost; then the same while a thread on
/// a third processor keeps revoking the clients' endpoint capability and
/// granting it again.
fn smp_stress(rounds: u64, revoke: bool) -> T {
    need_cpus(4)?;
    let master = new(ObjectType::Endpoint, 0);
    let lent = slot();
    cap_derive(SPACE, lent, master, Rights::ALL, 0).map_err(|_| "derive")?;
    SHARED[3].store(lent, SeqCst);
    ROUNDS.store(0, SeqCst);
    UNDERWAY.store(0, SeqCst);
    LOST.store(0, SeqCst);
    RETRIES.store(0, SeqCst);
    // Server and client share processor 1 (the fast, local path); a second
    // client calls from processor 2 (the cross-processor path).
    let threads = [
        ("server", spawn_on(1, stress_server, master.0, Cap::NONE).0),
        ("client on processor 1", spawn_on(1, stress_client, rounds, Cap::NONE).0),
        ("client on processor 2", spawn_on(2, stress_client, rounds / 10, Cap::NONE).0),
        ("spinner", spawn_on(3, count_for_ever, 3, Cap::NONE).0),
    ];
    let total = rounds + rounds / 10;
    let mut progress = Progress::new(if revoke { "smp_revoke: round trips" } else { "smp_stress: round trips" });
    let mut revokes = 0u64;
    while ROUNDS.load(SeqCst) < total {
        if let Err(e) = progress.check(UNDERWAY.load(SeqCst)) {
            println!("smp_stress: {} of {total} round trips, {revokes} revokes, {} retries", ROUNDS.load(SeqCst), RETRIES.load(SeqCst));
            // Where each thread is: the first question about a stall.
            for (who, t) in threads {
                match inspect_thread(t) {
                    Ok(i) => println!(
                        "smp_stress:   {who}: state {} on processor {}, ip {:#x}, waits on {:#x}",
                        i.state, i.cpu, i.ip, i.waits_on
                    ),
                    Err(_) => println!("smp_stress:   {who}: cannot be inspected"),
                }
            }
            return Err(e);
        }
        if revoke {
            // SAFETY: only the stress clients use the lent capability.
            unsafe { cap_delete(Cap(lent)) }.map_err(|_| "revoke")?;
            cap_derive(SPACE, lent, master, Rights::ALL, 0).map_err(|_| "grant again")?;
            revokes += 1;
            // A revoke fails the calls in flight through the capability
            // (spec/ARCHITECTURE.md section 6, IPC). Revoking again before
            // any call could finish would starve the clients by design, so
            // the capability stays granted for 20 microseconds: longer than
            // a cross-processor round trip, short enough that thousands of
            // calls are cut off mid-flight.
            let until = CLOCK.now() + 20_000;
            while CLOCK.now() < until {
                core::hint::spin_loop();
            }
        } else {
            sleep_ms(5);
        }
    }
    if LOST.load(SeqCst) != 0 {
        return Err("a call returned a wrong reply or an undocumented error");
    }
    if revoke && revokes < 1000 {
        return Err("the capability was revoked fewer than 1000 times during the run");
    }
    if revoke && RETRIES.load(SeqCst) < 100 {
        return Err("fewer than 100 calls were cut off by a revoke: the abort path was not exercised");
    }
    if revoke {
        println!("smp_revoke: {total} round trips, {revokes} revokes, {} calls cut off and retried", RETRIES.load(SeqCst));
    }
    Ok(())
}

fn t_smp_stress() -> T {
    smp_stress(1_000_000, false)
}

fn t_smp_revoke() -> T {
    smp_stress(1_000_000, true)
}

static TIMEOUT_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const STOP: usize = 5;
const GOT: usize = 0;
const SENT: usize = 1;
const TIMED_OUT: usize = 2;
const WRONG: usize = 3;
const FINISHED: usize = 4;

extern "C" fn impatient_receiver(arg: u64) -> ! {
    let ep = Cap(SHARED[3].load(SeqCst));
    let mut n = arg & 0xFF;
    while TIMEOUT_STATS[STOP].load(SeqCst) == 0 {
        n += 1;
        match recv(ep, Wait::Until(CLOCK.now() + 10_000 + n % 7 * 3_000)) {
            Ok(m) if m.words[0] == 9 => TIMEOUT_STATS[GOT].fetch_add(1, SeqCst),
            Err(Error::Timeout) => TIMEOUT_STATS[TIMED_OUT].fetch_add(1, SeqCst),
            _ => TIMEOUT_STATS[WRONG].fetch_add(1, SeqCst),
        };
    }
    TIMEOUT_STATS[FINISHED].fetch_add(1, SeqCst);
    finish(arg)
}

extern "C" fn hasty_sender(arg: u64) -> ! {
    let ep = Cap(SHARED[3].load(SeqCst));
    // The senders pause on one notification they share, so its queue too
    // holds waiters of several processors, each with a deadline.
    let pause = Cap(arg & 0xFFFF_FFFF);
    for i in 0..100_000u64 {
        if i % 32 == 0 {
            let _ = recv(pause, Wait::Until(CLOCK.now() + 30_000));
        }
        match try_send(ep, MsgInfo::new(7, 1), [9, 0, 0, 0]) {
            Ok(()) => TIMEOUT_STATS[SENT].fetch_add(1, SeqCst),
            Err(Error::WouldBlock) => 0,
            Err(_) => TIMEOUT_STATS[WRONG].fetch_add(1, SeqCst),
        };
    }
    TIMEOUT_STATS[FINISHED].fetch_add(1, SeqCst);
    finish(arg)
}

/// Threads on four processors receive on one endpoint, each with a deadline
/// a few microseconds away, while threads on three processors send to it
/// without waiting. The queue of receivers is linked through threads of
/// every processor and belongs to whichever waited last, so each timeout
/// and each send unlinks a waiter that may be another processor's. Every
/// message sent is received exactly once and nothing else happens.
fn t_smp_timeouts() -> T {
    need_cpus(4)?;
    let ep = new(ObjectType::Endpoint, 0);
    SHARED[3].store(ep.0, SeqCst);
    for s in &TIMEOUT_STATS {
        s.store(0, SeqCst);
    }
    for cpu in 0..4 {
        let _ = spawn_on(cpu, impatient_receiver, cpu as u64, Cap::NONE);
    }
    let pause = new(ObjectType::Notification, 0);
    for cpu in 1..4 {
        let _ = spawn_on(cpu, hasty_sender, pause.0, Cap::NONE);
    }
    // Progress is anything done: a message sent or received, a timeout.
    let mut progress = Progress::new("smp_timeouts: sends, receives and timeouts");
    let mut wait = |finished: u64| {
        while TIMEOUT_STATS[FINISHED].load(SeqCst) < finished {
            progress.check(TIMEOUT_STATS[SENT].load(SeqCst) + TIMEOUT_STATS[GOT].load(SeqCst) + TIMEOUT_STATS[TIMED_OUT].load(SeqCst))?;
            sleep_ms(1);
        }
        Ok(())
    };
    wait(3)?;
    TIMEOUT_STATS[STOP].store(1, SeqCst);
    wait(7)?;
    let stat = |i: usize| TIMEOUT_STATS[i].load(SeqCst);
    println!("smp_timeouts: {} sent, {} received, {} timeouts", stat(SENT), stat(GOT), stat(TIMED_OUT));
    if stat(WRONG) != 0 {
        return Err("a receive or a send returned something undocumented");
    }
    if stat(GOT) != stat(SENT) {
        return Err("messages sent and messages received differ");
    }
    if stat(SENT) < 100 || stat(TIMED_OUT) < 100 {
        return Err("the test did not exercise both sends and timeouts");
    }
    Ok(())
}

static CLOCK_CELL: [AtomicU64; 3] = [const { AtomicU64::new(1) }; 3];
struct LazyClock;
static CLOCK: LazyClock = LazyClock;
impl LazyClock {
    fn get(&self) -> Clock {
        Clock::new([CLOCK_CELL[0].load(SeqCst), CLOCK_CELL[1].load(SeqCst), CLOCK_CELL[2].load(SeqCst)])
    }
    fn now(&self) -> u64 {
        self.get().now()
    }
}

// ---- benchmarks -----------------------------------------------------------
//
// `bench` on the command line runs these instead of a test and prints
// `BENCH <metric> <value> <unit>` lines, the same metrics with the same
// payloads and statistics as the Linux baseline program
// (os/bench/linux/init.c).

const SAMPLES: usize = 20_000;
const WARMUP: usize = 2_000;
static mut SAMPLE: [u64; SAMPLES] = [0; SAMPLES];

use arch::counter as rdtsc;

fn samples() -> &'static mut [u64; SAMPLES] {
    // SAFETY: only the benchmark's main thread uses the sample buffer.
    unsafe { &mut *(&raw mut SAMPLE) }
}

fn report(name: &str, n: usize, div: u64) {
    let s = &mut samples()[..n];
    s.sort_unstable();
    let ns = |ticks: u64| (ticks as u128 * 1_000_000_000 / CLOCK_CELL[0].load(SeqCst) as u128) as u64 / div;
    println!("BENCH {name}_min {} ns", ns(s[0]));
    println!("BENCH {name} {} ns", ns(s[n / 2]));
    println!("BENCH {name}_p99 {} ns", ns(s[n - n / 100 - 1]));
}

unsafe extern "C" {
    static stub_echo: u8;
    static stub_echo_end: u8;
    static stub_yield: u8;
    static stub_yield_end: u8;
    static stub_spin: u8;
    static stub_spin_end: u8;
}

fn stub(start: *const u8, end: *const u8) -> &'static [u8] {
    // SAFETY: both symbols are labels around one stub in this program's text.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

/// A thread in its own address space running `code` (position-independent
/// machine code that needs no stack and no data).
fn foreign_thread(code: &[u8], arg: u64) -> Cap {
    foreign_thread_on(0, code, arg)
}

/// The page `foreign_thread_on` writes code through (0: not chosen yet).
static STAGING: AtomicU64 = AtomicU64::new(0);

/// Like `foreign_thread`, on processor `cpu`.
fn foreign_thread_on(cpu: usize, code: &[u8], arg: u64) -> Cap {
    let aspace = new(ObjectType::AddressSpace, 16);
    let frame = new(ObjectType::Frame, 12);
    // One page of this address space is where code is written every time,
    // so making threads again and again (a benchmark repeated for a
    // profile) uses no more page tables.
    let at = match STAGING.load(SeqCst) {
        0 => {
            let at = region(1 << 12) as u64;
            STAGING.store(at, SeqCst);
            at as usize
        }
        at => at as usize,
    };
    map(ASPACE, frame, at, RW).expect("map code for writing");
    // SAFETY: one page mapped read-write at `at`; `code` fits in it.
    unsafe {
        core::ptr::copy_nonoverlapping(code.as_ptr(), at as *mut u8, code.len());
        arch::sync_code(at, code.len());
        unmap(frame).expect("unmap");
    }
    map(aspace, frame, 0x10_0000, Rights(5)).expect("map code in the new address space");
    let t = new(ObjectType::Thread, 0);
    thread_configure(t, SPACE, aspace, Cap::NONE, Cap::NONE).expect("configure");
    // SAFETY: the code was written for exactly this: it runs at any address without a stack.
    unsafe { thread_start(t, 0x10_0000, 0, arg, false) }.expect("start");
    if cpu != 0 {
        thread_set_cpu(t, cpu).expect("move to processor");
    }
    thread_resume(t).expect("resume");
    t
}

fn bench_null_syscall() {
    for i in 0..SAMPLES + WARMUP {
        let t = rdtsc();
        raw_syscall(9999);
        let t = rdtsc() - t;
        if i >= WARMUP {
            samples()[i - WARMUP] = t;
        }
    }
    report("null_syscall", SAMPLES, 1);
}

/// A call and its reply between two address spaces on one processor, four
/// message words (32 bytes) each way.
fn bench_ipc() {
    let ep = new(ObjectType::Endpoint, 0);
    let server = foreign_thread(stub(&raw const stub_echo, &raw const stub_echo_end), ep.0);
    for i in 0..SAMPLES + WARMUP {
        let t = rdtsc();
        let r = call(ep, MsgInfo::new(1, 4), [i as u64, 2, 3, 4]);
        let t = rdtsc() - t;
        if r.map(|m| m.words[0]) != Ok(i as u64) {
            println!("ERROR: bench: the echo server answered wrongly");
            exit(false);
        }
        if i >= WARMUP {
            samples()[i - WARMUP] = t;
        }
    }
    // SAFETY: the server thread is used by nothing else.
    let _ = unsafe { cap_delete(server) };
    report("ipc_roundtrip", SAMPLES, 1);
}

/// Two threads in different address spaces yielding to each other, each
/// with live AVX state: one sample is two switches.
fn bench_ctx_switch() {
    let other = foreign_thread(stub(&raw const stub_yield, &raw const stub_yield_end), 7);
    for i in 0..SAMPLES + WARMUP {
        let t = arch::timed_yield(i as u64);
        if i >= WARMUP {
            samples()[i - WARMUP] = t;
        }
    }
    // SAFETY: the other thread is used by nothing else.
    let _ = unsafe { cap_delete(other) };
    report("ctx_switch", SAMPLES, 2);
}

static FAULT_BASE: AtomicU64 = AtomicU64::new(0);
const FAULT_PAGES: usize = 2048;

extern "C" fn toucher(arg: u64) -> ! {
    let base = FAULT_BASE.load(SeqCst) as *mut u8;
    for i in 0..FAULT_PAGES {
        let t = rdtsc();
        // SAFETY: the page is absent; the pager maps it and the write restarts.
        unsafe { core::ptr::write_volatile(base.add(i * 4096), 1) };
        // SAFETY: the main thread reads the samples only after this thread signals.
        unsafe { (*(&raw mut SAMPLE))[i] = rdtsc() - t };
    }
    SHARED[0].store(99, SeqCst);
    finish(arg)
}

/// First touch of a page, resolved by a pager thread in userspace.
fn bench_page_fault() {
    // One set of frames and one range of addresses, made by the first run
    // and unmapped again at the end of each, so the benchmark can run any
    // number of times in a boot (bench=fault:<times>) at the same cost each
    // time: retyped anew per run, the frames' memory stayed with them (a
    // deleted capability gives no memory back to the Untyped), and the
    // tenth run of a boot failed for want of it.
    let first = match FAULT_SLOTS.load(SeqCst) {
        0 => {
            let f = NEXT_SLOT.fetch_add(FAULT_PAGES as u64, SeqCst);
            for i in 0..FAULT_PAGES as u64 / 256 {
                retype(Cap(UNTYPED.load(SeqCst)), ObjectType::Frame, 12, SPACE, f + i * 256, 256).expect("frames");
            }
            FAULT_BASE.store(region((FAULT_PAGES as u64) << 12) as u64, SeqCst);
            FAULT_SLOTS.store(f, SeqCst);
            f
        }
        f => f,
    };
    SHARED[0].store(0, SeqCst);
    let ep = new(ObjectType::Endpoint, 0);
    let _ = spawn(toucher, 0, ep);
    let mut m = recv(ep, Wait::Forever).expect("fault");
    for i in 0..FAULT_PAGES as u64 {
        if i + 1 == FAULT_PAGES as u64 {
            fault_map(ASPACE, Cap(first + i), m.words[1] as usize, RW).expect("map and resume");
        } else {
            m = fault_map_recv(ep, ASPACE, Cap(first + i), m.words[1] as usize, RW).expect("map, resume, next fault");
        }
    }
    let _ = wait_for(&SHARED[0], 99);
    report("page_fault", FAULT_PAGES, 1);
    for i in 0..FAULT_PAGES as u64 {
        // SAFETY: the frames were mapped only at the benchmark's range,
        // which nothing touches any more (the toucher has finished).
        unsafe { unmap(Cap(first + i)) }.expect("unmap a faulted page");
    }
}

/// The slots `bench_page_fault` makes its frames in (0: not chosen yet).
static FAULT_SLOTS: AtomicU64 = AtomicU64::new(0);

fn bench_timer() {
    let n = new(ObjectType::Notification, 0);
    // As many as every other micro-benchmark takes (spec/TESTING.md 4.1),
    // so the 99th percentile rests on a hundred samples, not five.
    let count = 10_000;
    for i in 0..count {
        let at = CLOCK.now() + 1_000_000;
        let _ = recv(n, Wait::Until(at));
        samples()[i] = CLOCK.now().saturating_sub(at);
    }
    let s = &mut samples()[..count];
    s.sort_unstable();
    println!("BENCH timer_overshoot {} ns", s[count / 2]);
    println!("BENCH timer_overshoot_p99 {} ns", s[count - count / 100 - 1]);
}

/// Wake-up latencies of the scheduling benchmark, in 100 ns steps up to
/// 2 ms (the last bucket holds everything later; the maximum is kept apart).
static LATENCY: [AtomicU64; 20_001] = [const { AtomicU64::new(0) }; 20_001];

/// Scheduling latency under load (spec/TESTING.md section 4.2): every
/// processor busy with a thread of its own address space that computes and
/// never waits, and on processor 0, beside its load, a thread that sleeps
/// 1 ms at a time for `seconds` seconds, timing how late it runs again.
/// Both at the same (fair) priority, as the Linux baseline's are. Prints
/// the 99.9th percentile and the maximum.
fn bench_sched(seconds: u64) {
    let cpus = CPUS.load(SeqCst).min(64) as usize;
    let mut hogs = [Cap(0); 64];
    for (cpu, h) in hogs.iter_mut().enumerate().take(cpus) {
        *h = foreign_thread_on(cpu, stub(&raw const stub_spin, &raw const stub_spin_end), 0);
    }
    let n = new(ObjectType::Notification, 0);
    for b in &LATENCY {
        b.store(0, SeqCst);
    }
    let (end, mut count, mut max) = (CLOCK.now() + seconds * 1_000_000_000, 0u64, 0u64);
    while CLOCK.now() < end {
        let at = CLOCK.now() + 1_000_000;
        let _ = recv(n, Wait::Until(at));
        let late = CLOCK.now().saturating_sub(at);
        max = max.max(late);
        LATENCY[(late / 100).min(20_000) as usize].fetch_add(1, SeqCst);
        count += 1;
    }
    for &h in &hogs[..cpus] {
        // SAFETY: the load threads are used by nothing else.
        let _ = unsafe { cap_delete(h) };
    }
    let mut seen = 0;
    let want = count - count / 1000;
    let p999 = LATENCY.iter().position(|b| {
        seen += b.load(SeqCst);
        seen >= want
    });
    println!("BENCH sched_latency_p999 {} ns", p999.unwrap_or(20_000) as u64 * 100 + 100);
    println!("BENCH sched_latency_max {max} ns");
}

static GO: AtomicU64 = AtomicU64::new(0);
static COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

extern "C" fn pair_client(arg: u64) -> ! {
    let (pair, ep) = ((arg & 0xFF) as usize, Cap((arg >> 8) & 0xFF_FFFF));
    while GO.load(SeqCst) == 0 {
        core::hint::spin_loop();
    }
    let mut n = 0u64;
    while GO.load(SeqCst) == 1 {
        for _ in 0..64 {
            let _ = call(ep, MsgInfo::new(1, 4), [n, 2, 3, 4]);
        }
        n += 64;
    }
    COUNTS[pair].store(n, SeqCst);
    finish(arg)
}

/// Round trips per second of `pairs` independent client and server pairs,
/// each server in its own address space. Each pair has a processor to
/// itself, or with `cross` two: the client on one, the server on the next.
fn bench_pairs(pairs: usize, cross: bool) -> u64 {
    GO.store(0, SeqCst);
    let mut threads = [Cap::NONE; 16];
    for p in 0..pairs {
        COUNTS[p].store(0, SeqCst);
        let ep = new(ObjectType::Endpoint, 0);
        let (client_cpu, server_cpu) = if cross { (2 * p, 2 * p + 1) } else { (p, p) };
        threads[2 * p] = foreign_thread_on(server_cpu, stub(&raw const stub_echo, &raw const stub_echo_end), ep.0);
        threads[2 * p + 1] = spawn_on(client_cpu, pair_client, p as u64 | ep.0 << 8, Cap::NONE).0;
    }
    // Processor 0 hosts a pair too, so this thread sleeps while they run.
    sleep_ms(20);
    let start = CLOCK.now();
    GO.store(1, SeqCst);
    sleep_ms(300);
    GO.store(2, SeqCst);
    let took = CLOCK.now() - start;
    let mut total = 0;
    for p in 0..pairs {
        for _ in 0..1000 {
            if COUNTS[p].load(SeqCst) != 0 {
                break;
            }
            sleep_ms(1);
        }
        total += COUNTS[p].load(SeqCst);
    }
    for t in &threads[..2 * pairs] {
        // SAFETY: the benchmark's own threads, finished or blocked.
        let _ = unsafe { cap_delete(*t) };
    }
    (total as u128 * 1_000_000_000 / took as u128) as u64
}

fn bench() -> ! {
    bench_null_syscall();
    bench_ipc();
    bench_ctx_switch();
    bench_page_fault();
    bench_timer();
    let one = bench_pairs(1, false);
    println!("BENCH ipc_pairs1 {one} ops/s");
    if CPUS.load(SeqCst) >= 8 {
        let eight = bench_pairs(8, false);
        println!("BENCH ipc_pairs8 {eight} ops/s");
        println!("BENCH ipc_scaling {}.{:02} x", eight / one, eight * 100 / one % 100);
    }
    if CPUS.load(SeqCst) >= 2 {
        let one = bench_pairs(1, true);
        println!("BENCH ipc_cross_pairs1 {one} ops/s");
        if CPUS.load(SeqCst) >= 8 {
            let four = bench_pairs(4, true);
            println!("BENCH ipc_cross_pairs4 {four} ops/s");
            println!("BENCH ipc_cross_scaling {}.{:02} x", four / one, four * 100 / one % 100);
        }
    }
    // Two minutes a boot: five boots make the ten minutes the gate asks for.
    bench_sched(120);
    println!("BENCH DONE");
    exit(true)
}

/// The console and power capabilities: a line through the console, and
/// every way of misusing either is refused with the documented error.
fn t_console() -> T {
    console_write(Cap(SLOT_CONSOLE), b"console: a line written through the console capability\r\n")
        .map_err(|_| "a console write failed")?;
    let raw = |cap: u64, n: u64| libsys::raw(Syscall::Console, [cap, n, 0, 0, 0, 0]).map(|_| ());
    if raw(SLOT_CONSOLE, 33) != Err(Error::BadArg) {
        return Err("a console write of more than 32 bytes was accepted");
    }
    if raw(SLOT_IRQS, 1) != Err(Error::WrongType) || raw(SLOT_POWER, 1) != Err(Error::WrongType) {
        return Err("a hardware capability that is not the console wrote to it");
    }
    let ep = new(ObjectType::Endpoint, 0);
    if raw(ep.0, 1) != Err(Error::WrongType) || raw(0, 1) != Err(Error::BadCap) {
        return Err("a console write through something that is not a capability to it was accepted");
    }
    let weak = slot();
    cap_derive(SPACE, weak, Cap(SLOT_CONSOLE), Rights::READ, 0).map_err(|_| "derive a read-only console capability")?;
    if raw(weak, 1) != Err(Error::NoRights) {
        return Err("a console capability without write rights wrote");
    }
    if power_off(Cap(SLOT_CONSOLE), 0) != Error::WrongType || power_off(ep, 0) != Error::WrongType {
        return Err("something other than the power capability was accepted to turn the machine off");
    }
    Ok(())
}

/// Spinner state for `t_sched`: 0 wait, 1 count, 2 stop and report.
static SPIN: AtomicU64 = AtomicU64::new(0);
static SPUN: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// How late `sched_sleeper`'s 1 ms sleeps woke, at most, in nanoseconds (+1 when done).
static WORST_WAKE: AtomicU64 = AtomicU64::new(0);

/// Count as fast as possible until told to stop: CPU time, measured.
extern "C" fn sched_spinner(arg: u64) -> ! {
    let i = (arg & 7) as usize;
    while SPIN.load(SeqCst) == 0 {
        core::hint::spin_loop();
    }
    let mut n = 0u64;
    while SPIN.load(SeqCst) == 1 {
        n += 1;
        if n % 4096 == 0 {
            SPUN[i].store(n, SeqCst);
        }
    }
    SPUN[i].store(n, SeqCst);
    finish(arg)
}

/// Sleep 1 ms at a time, 200 times, and record the latest wake. The first
/// ten are not counted: a thread started beside the busy ones is their
/// equal until it has slept, and may wait one turn; what is measured is a
/// thread that mostly sleeps.
extern "C" fn sched_sleeper(arg: u64) -> ! {
    let n = new(ObjectType::Notification, 0);
    let mut worst = 0;
    for i in 0..200 {
        let at = CLOCK.now() + 1_000_000;
        let _ = recv(n, Wait::Until(at));
        if i >= 10 {
            worst = worst.max(CLOCK.now().saturating_sub(at));
        }
    }
    WORST_WAKE.store(worst + 1, SeqCst);
    finish(arg)
}

/// Run `spinners` (index, scheduling context or Cap::NONE) on processor
/// `cpu` for `ms` and return what each counted.
fn spin_on(cpu: usize, spinners: &[(usize, Cap)], ms: u64) -> core::result::Result<[u64; 8], &'static str> {
    SPIN.store(0, SeqCst);
    let mut threads = [Cap::NONE; 8];
    for &(i, sc) in spinners {
        SPUN[i].store(0, SeqCst);
        let (t, _) = spawn_on(cpu, sched_spinner, i as u64, Cap::NONE);
        if sc != Cap::NONE {
            thread_bind(t, sc).map_err(|_| "cannot bind a thread to a scheduling context")?;
        }
        threads[i] = t;
    }
    SPIN.store(1, SeqCst);
    sleep_ms(ms);
    SPIN.store(2, SeqCst);
    sleep_ms(20);
    let mut out = [0u64; 8];
    for &(i, _) in spinners {
        out[i] = SPUN[i].load(SeqCst);
        // SAFETY: the test's own thread, stopped or about to be.
        let _ = unsafe { cap_delete(threads[i]) };
    }
    Ok(out)
}

/// The scheduler on real time: weights, a deadline reservation, wake-up
/// latency beside busy threads, admission control.
fn t_sched() -> T {
    // A processor of its own where there is one; on one processor the
    // main thread sleeps while the others run.
    let cpus = CPUS.load(SeqCst) as usize;
    let cpu = cpus - 1;
    // 1. Weights 1024 and 3072 share the processor about 1:3.
    let heavy = new(ObjectType::SchedContext, 0);
    sched_fair(heavy, 3072).map_err(|_| "cannot configure a fair context")?;
    let c = spin_on(cpu, &[(0, Cap::NONE), (1, heavy)], 300)?;
    let ratio = c[1] as f64 / c[0].max(1) as f64;
    println!("sched: weights 1024 and 3072 counted {} and {} (ratio {:.2})", c[0], c[1], ratio);
    if !(2.5..=3.5).contains(&ratio) {
        return Err("weights 1024 and 3072 did not share the processor about 1:3");
    }
    // 2. A 2 ms in 10 ms reservation beside three fair spinners gets a fifth.
    let dl = new(ObjectType::SchedContext, 0);
    sched_deadline(dl, 2_000_000, 10_000_000, cpu).map_err(|_| "cannot reserve 20% of a processor")?;
    let c = spin_on(cpu, &[(0, Cap::NONE), (1, Cap::NONE), (2, Cap::NONE), (3, dl)], 400)?;
    let total: u64 = c[..4].iter().sum();
    let share = c[3] as f64 / total.max(1) as f64;
    println!("sched: a 2 ms in 10 ms reservation beside three fair spinners got {:.1}%", share * 100.0);
    if !(0.17..=0.23).contains(&share) {
        return Err("a 20% deadline reservation did not get about 20% beside fair spinners");
    }
    if inspect_sched(dl).map(|i| i.consumed).unwrap_or(0) < 60_000_000 {
        return Err("the reservation's context was not charged for the time its thread ran");
    }
    // 3. More than 90% of a processor is not promised.
    let more = new(ObjectType::SchedContext, 0);
    if sched_deadline(more, 8_000_000, 10_000_000, cpu) != Err(Error::Overload) {
        return Err("a reservation that would take the processor past 90% was accepted");
    }
    // SAFETY: nothing is bound to them any more.
    unsafe {
        let _ = cap_delete(dl);
        let _ = cap_delete(heavy);
    }
    sched_deadline(more, 8_000_000, 10_000_000, cpu).map_err(|_| "a reservation's share was not given back when it was destroyed")?;
    // SAFETY: nothing is bound to it.
    let _ = unsafe { cap_delete(more) };
    // 4. A thread waking from 1 ms sleeps beside two spinners runs promptly.
    WORST_WAKE.store(0, SeqCst);
    SPIN.store(0, SeqCst);
    let (sleeper, _) = spawn_on(cpu, sched_sleeper, 7, Cap::NONE);
    let c = spin_on(cpu, &[(0, Cap::NONE), (1, Cap::NONE)], 300);
    for _ in 0..200 {
        if WORST_WAKE.load(SeqCst) != 0 {
            break;
        }
        sleep_ms(5);
    }
    // SAFETY: the sleeper is finished.
    let _ = unsafe { cap_delete(sleeper) };
    c?;
    let worst = WORST_WAKE.load(SeqCst);
    println!("sched: beside two spinners, 1 ms sleeps woke at most {} us late", worst.saturating_sub(1) / 1000);
    if worst == 0 {
        return Err("the sleeping thread never finished beside the spinners");
    }
    if worst - 1 > 2_000_000 {
        return Err("a waking thread waited behind busy ones for more than 2 ms");
    }
    Ok(())
}

/// The secret the root task starts with: from at least one source the
/// loader trusts (the emulated machines have both the firmware's and the
/// processor's), and not a constant.
fn t_entropy() -> T {
    let w: [u64; 4] = core::array::from_fn(|i| ENTROPY[i].load(SeqCst));
    let sources = ENTROPY[4].load(SeqCst);
    println!(
        "entropy: sources{}{}",
        if sources & ENTROPY_FIRMWARE != 0 { " firmware" } else { "" },
        if sources & ENTROPY_CPU != 0 { " processor" } else { "" }
    );
    if sources == 0 {
        return Err("the loader found no source of randomness it trusts");
    }
    if w.iter().all(|&x| x == w[0]) {
        return Err("the root task's secret is a repeated word");
    }
    Ok(())
}

const TESTS: &[(&str, fn() -> T)] = &[
    ("entropy", t_entropy),
    ("sched", t_sched),
    ("console", t_console),
    ("syscall", t_syscall),
    ("fault_page", t_fault_page),
    ("fault_instruction", t_fault_instruction),
    ("fault_privileged", t_fault_privileged),
    ("fault_kernel_memory", t_fault_kernel_memory),
    ("fault_stack", t_fault_stack),
    ("fault_resume", t_fault_resume),
    ("w_xor_x", t_w_xor_x),
    ("map_rollback", t_map_rollback),
    ("ipc", t_ipc),
    ("timer", t_timer),
    ("preempt", t_preempt),
    ("fpu", t_fpu),
    ("untyped", t_untyped),
    ("caps", t_caps),
    ("irq", t_irq),
    ("inspect", t_inspect),
    ("ipc_caps", t_ipc_caps),
    ("ipc_parked", t_ipc_parked),
    ("smp_run", t_smp_run),
    ("smp_ipc", t_smp_ipc),
    ("smp_shootdown", t_smp_shootdown),
    ("smp_stress", t_smp_stress),
    ("smp_revoke", t_smp_revoke),
    ("smp_timeouts", t_smp_timeouts),
];

/// Program entry, called by libsys's `_start`.
// SAFETY: the one definition of the symbol `_start` calls.
#[unsafe(no_mangle)]
extern "C" fn silica_main(info: &'static RootInfo) -> ! {
    if info.magic != ROOT_MAGIC {
        exit(false);
    }
    for i in 0..3 {
        CLOCK_CELL[i].store(info.clock[i], SeqCst);
    }
    NEXT_SLOT.store(info.first_free_slot, SeqCst);
    CPUS.store(info.cpus, SeqCst);
    for (i, w) in info.entropy.as_chunks::<8>().0.iter().enumerate() {
        ENTROPY[i].store(u64::from_le_bytes(*w), SeqCst);
    }
    ENTROPY[4].store(info.entropy_sources, SeqCst);
    INFO.store(info as *const RootInfo as u64, SeqCst);
    ROOT_BUFFER.store(info.ipc_buffer, SeqCst);
    let big = (0..info.untyped_count as usize).filter(|&i| info.untyped[i].kind == KIND_RAM).max_by_key(|&i| info.untyped[i].bits);
    let Some(big) = big else {
        println!("ERROR: roottest: the kernel handed over no ordinary memory");
        exit(false)
    };
    UNTYPED.store(SLOT_FIRST_UNTYPED + big as u64, SeqCst);

    let cmdline = &info.cmdline[..info.cmdline_len as usize];
    let words = || cmdline.split(|&b| b == b' ' || b == b'\n').map(<[u8]>::trim_ascii);
    if words().any(|w| w == b"bench") {
        println!("BENCH MARK userspace");
        bench()
    }
    // "bench=<name>[:<times>]": that one benchmark alone, once or <times>
    // times over, for a profile of its path (tools/profile.sh).
    if let Some(arg) = words().find_map(|w| w.strip_prefix(b"bench=")) {
        let (name, times) = match arg.iter().position(|&b| b == b':') {
            Some(i) => (&arg[..i], core::str::from_utf8(&arg[i + 1..]).ok().and_then(|n| n.parse().ok()).unwrap_or(0)),
            None => (arg, 1),
        };
        if name == b"sched" {
            bench_sched(times);
            println!("BENCH DONE");
            exit(true)
        }
        let f: fn() = match name {
            b"null" => bench_null_syscall,
            b"ipc" => bench_ipc,
            b"ctx" => bench_ctx_switch,
            b"fault" => bench_page_fault,
            b"timer" => bench_timer,
            b"pairs8" => || println!("BENCH ipc_pairs8 {} ops/s", bench_pairs(8, false)),
            b"cross1" => || println!("BENCH ipc_cross_pairs1 {} ops/s", bench_pairs(1, true)),
            _ => {
                println!("ERROR: roottest: no benchmark of that name (null, ipc, ctx, fault, timer, pairs8, cross1; sched:<seconds>)");
                exit(false)
            }
        };
        if times == 0 {
            println!("ERROR: roottest: bench=<name>:<times> needs a count of at least 1");
            exit(false)
        }
        for _ in 0..times {
            f();
        }
        println!("BENCH DONE");
        exit(true)
    }
    let test = cmdline.split(|&b| b == b' ' || b == b'\n').find_map(|w| w.strip_prefix(b"test="));
    let Some(name) = test.and_then(|n| core::str::from_utf8(n).ok()).map(str::trim_ascii) else {
        let since_loader = CLOCK.get().since(info.clock[1]);
        println!("BENCH boot_userspace_from_loader_us {} us", since_loader / 1000);
        println!("{}", "BOOT OK");
        exit(true)
    };
    match TESTS.iter().find(|(n, _)| *n == name) {
        Some((_, f)) => match f() {
            Ok(()) => {
                println!("TEST {name} PASS");
                exit(true)
            }
            Err(why) => {
                println!("TEST {name} FAIL {why}");
                exit(false)
            }
        },
        None => {
            println!("TEST {name} FAIL no test of that name in roottest");
            exit(false)
        }
    }
}
