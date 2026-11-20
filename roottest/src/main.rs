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

use core::arch::asm;
use core::fmt::{self, Write};
use core::sync::atomic::{AtomicU64, Ordering::SeqCst};
use libsys::root::*;
use libsys::*;

const PORTS: Cap = Cap(SLOT_PORTS);
const SPACE: Cap = Cap(SLOT_CAPSPACE);
const ASPACE: Cap = Cap(SLOT_ASPACE);
const RW: Rights = Rights(3);

struct Serial;
impl Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            while port_in(PORTS, 0x3FD, 1).unwrap_or(0x20) & 0x20 == 0 {}
            let _ = port_out(PORTS, 0x3F8, 1, b as u32);
        }
        Ok(())
    }
}
macro_rules! println {
    ($($a:tt)*) => {{ let _ = write!(Serial, $($a)*); let _ = Serial.write_str("\r\n"); }};
}

fn exit(ok: bool) -> ! {
    let _ = port_out(PORTS, 0xF4, 1, if ok { 0x10 } else { 0x11 });
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
    unsafe {
        thread_start(
            t,
            entry as usize,
            stack + (1 << 16) - 8,
            arg | t.0 << 32,
            true,
        )
    }
    .expect("start");
    (t, stack)
}

/// What a spawned thread does when it is finished.
fn finish(arg: u64) -> ! {
    let _ = thread_suspend(Cap(arg >> 32));
    loop {
        yield_now();
    }
}

fn raw_syscall(n: u64) -> u64 {
    let r: u64;
    // SAFETY: a syscall with no arguments; the kernel preserves everything
    // but rax, rcx and r11.
    unsafe {
        asm!("syscall", inout("rax") n => r, out("rcx") _, out("r11") _, lateout("rdi") _, lateout("rsi") _, lateout("rdx") _, lateout("r10") _, lateout("r8") _, lateout("r9") _, options(nostack))
    }
    r
}

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
            2 => asm!("ud2"),
            3 => asm!("cli"),
            4 => {
                core::ptr::read_volatile(0xFFFF_8000_0010_0000 as *const u8);
            }
            5 => {
                core::ptr::read_volatile(0xFFFF_FFFF_8000_0000 as *const u8);
            }
            6 => asm!("hlt"),
            7 => asm!("mov rax, cr3", out("rax") _),
            8 => asm!("in al, dx", in("dx") 0x3F8u16, out("al") _),
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
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000))
        .map_err(|_| "no fault message arrived")?;
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
    if m.words[2] & 0b111 != 0b110 {
        return Err("the fault code does not say user-mode write to an unmapped page");
    }
    if m.words[0] < 0x40_0000 || m.words[3] == 0 {
        return Err("the fault message lacks the instruction or stack pointer");
    }
    Ok(())
}

fn t_fault_instruction() -> T {
    let m = fault_of(2)?;
    (kind(&m) == Fault::Instruction as u64)
        .then_some(())
        .ok_or("ud2 was not reported as an instruction fault")
}

fn t_fault_privileged() -> T {
    for what in [3, 6, 7, 8] {
        let m = fault_of(what)?;
        if kind(&m) != Fault::Exception as u64 || m.words[1] != 13 {
            return Err(
                "a privileged instruction (cli, hlt, mov from cr3, in) was not a general-protection fault",
            );
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
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000))
        .map_err(|_| "no fault message arrived")?;
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
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000))
        .map_err(|_| "no fault message arrived")?;
    if kind(&m) != Fault::Page as u64 || m.words[1] != at as u64 + 8 {
        return Err("the fault address is not the address touched");
    }
    let f = new(ObjectType::Frame, 12);
    map(ASPACE, f, at, RW).map_err(|_| "cannot map the missing page")?;
    reply(MsgInfo::new(0, 1), [FAULT_RESUME, 0, 0, 0]).map_err(|_| "cannot reply to the fault")?;
    wait_for(&SHARED[2], 42)
}

fn t_w_xor_x() -> T {
    let f = new(ObjectType::Frame, 12);
    let at = region(1 << 12);
    if map(ASPACE, f, at, Rights(7)) != Err(Error::BadArg) {
        return Err("a writable and executable mapping was accepted");
    }
    map(ASPACE, f, at, RW).map_err(|_| "cannot map")?;
    // SAFETY: the page was just mapped read-write; 0xC3 is `ret`.
    unsafe { core::ptr::write_volatile(at as *mut u8, 0xC3) };
    // Executing it must fault (not executable) with the instruction-fetch bit.
    let ep = new(ObjectType::Endpoint, 0);
    let t = new(ObjectType::Thread, 0);
    let (_, stack) = memory();
    thread_configure(t, SPACE, ASPACE, ep, Cap::NONE).map_err(|_| "configure")?;
    // SAFETY: the thread is expected to fault on its first instruction.
    unsafe { thread_start(t, at, stack + 0xFFF8, 0, true) }.map_err(|_| "start")?;
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000))
        .map_err(|_| "executing a data page did not fault")?;
    if kind(&m) != Fault::Page as u64 || m.words[1] != at as u64 || m.words[2] & 0x10 == 0 {
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
    let m = recv(ep, Wait::Until(CLOCK.now() + 2_000_000_000))
        .map_err(|_| "writing a read-only page did not fault")?;
    (kind(&m) == Fault::Page as u64 && m.words[2] & 0b11 == 0b11)
        .then_some(())
        .ok_or("writing a read-only page was not a protection fault")
}

extern "C" fn echo_server(arg: u64) -> ! {
    let ep = Cap(SHARED[3].load(SeqCst));
    let Ok(mut m) = recv(ep, Wait::Forever) else {
        finish(arg)
    };
    loop {
        if m.info.label() == 99 {
            let _ = reply(MsgInfo(0), [0; 4]);
            finish(arg);
        }
        let w = [
            m.words[0] * 2,
            m.words[1] + m.badge,
            !m.words[2],
            m.info.words() as u64,
        ];
        m = match reply_recv(ep, MsgInfo::new(m.info.label() + 1, 4), w) {
            Ok(m) => m,
            Err(_) => finish(arg),
        };
    }
}

fn t_ipc() -> T {
    let ep = new(ObjectType::Endpoint, 0);
    let badged = slot();
    cap_copy(SPACE, badged, ep, Rights(2), 1000).map_err(|_| "cannot badge an endpoint")?;
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
    recv(n, Wait::Poll)
        .map(|_| ())
        .map_err(|_| "a pending signal was not received")
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

extern "C" fn fpu_worker(arg: u64) -> ! {
    let id = arg & 0xFFFF_FFFF;
    let mut bad = 0u64;
    for round in 0..20_000u64 {
        let v = id << 32 | round;
        let (a, b, c): (u64, u64, u64);
        // SAFETY: loads a value into SSE and AVX registers, yields (syscall
        // 4) so the other worker runs with its own values, and reads the
        // registers back. Declares every register it touches.
        unsafe {
            asm!(
                "vmovq xmm0, {v}",
                "vpbroadcastq ymm15, xmm0",
                "vmovq xmm7, {v}",
                "mov eax, 4",
                "syscall",
                "vmovq {a}, xmm0",
                "vmovq {b}, xmm7",
                "vextracti128 xmm1, ymm15, 1",
                "vpextrq {c}, xmm1, 1",
                "vzeroupper",
                v = in(reg) v, a = out(reg) a, b = out(reg) b, c = out(reg) c,
                out("rax") _, out("rcx") _, out("r11") _, out("rdi") _, out("rsi") _, out("rdx") _, out("r10") _, out("r8") _, out("r9") _,
                out("xmm0") _, out("xmm1") _, out("xmm7") _, out("xmm15") _,
                options(nostack),
            );
        }
        if a != v || b != v || c != v {
            bad += 1;
        }
    }
    SHARED[id as usize].store(1 + bad, SeqCst);
    finish(arg)
}

fn t_fpu() -> T {
    SHARED[0].store(0, SeqCst);
    SHARED[1].store(0, SeqCst);
    let _ = spawn(fpu_worker, 0, Cap::NONE);
    let _ = spawn(fpu_worker, 1, Cap::NONE);
    // This thread computes in floating point across the same switches.
    let mut x = 1.0f64;
    for _ in 0..20_000 {
        x = x * 1.000_1 + 0.5;
        yield_now();
    }
    if !(x > 10_000.0 && x < 100_000.0) {
        return Err("floating-point state of the main thread was corrupted");
    }
    wait_for(&SHARED[0], 1)
        .and(wait_for(&SHARED[1], 1))
        .map_err(|_| "SSE or AVX registers changed across a thread switch")
}

fn t_untyped() -> T {
    let pool = new(ObjectType::Untyped, 20);
    let s = slot();
    let at = region(1 << 16);
    for round in 0..50u64 {
        retype(pool, ObjectType::Frame, 16, SPACE, s, 1)
            .map_err(|_| "retype from a revoked Untyped failed")?;
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
    cap_copy(SPACE, a, n, Rights::ALL, 1).map_err(|_| "copy")?;
    cap_copy(SPACE, b, Cap(a), Rights::WRITE, 0).map_err(|_| "copy of a copy")?;
    if cap_copy(SPACE, b, n, Rights::ALL, 0) != Err(Error::SlotFull) {
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
    if cap_copy(SPACE, a, Cap(UNTYPED.load(SeqCst)), Rights::ALL, 0) != Err(Error::InUse) {
        return Err("an Untyped was copied");
    }
    // A narrowed port range reaches only its ports.
    cap_copy(SPACE, a, PORTS, Rights::ALL, 0x3F8 | 0x400 << 32).map_err(|_| "narrow ports")?;
    if port_in(Cap(a), 0xF4, 1) != Err(Error::BadArg) || port_in(Cap(a), 0x3FD, 1).is_err() {
        return Err("a narrowed port capability did not confine port access");
    }
    Ok(())
}

static CLOCK_CELL: [AtomicU64; 3] = [const { AtomicU64::new(1) }; 3];
struct LazyClock;
static CLOCK: LazyClock = LazyClock;
impl LazyClock {
    fn get(&self) -> Clock {
        Clock::new([
            CLOCK_CELL[0].load(SeqCst),
            CLOCK_CELL[1].load(SeqCst),
            CLOCK_CELL[2].load(SeqCst),
        ])
    }
    fn now(&self) -> u64 {
        self.get().now()
    }
}

const TESTS: &[(&str, fn() -> T)] = &[
    ("syscall", t_syscall),
    ("fault_page", t_fault_page),
    ("fault_instruction", t_fault_instruction),
    ("fault_privileged", t_fault_privileged),
    ("fault_kernel_memory", t_fault_kernel_memory),
    ("fault_stack", t_fault_stack),
    ("fault_resume", t_fault_resume),
    ("w_xor_x", t_w_xor_x),
    ("ipc", t_ipc),
    ("timer", t_timer),
    ("preempt", t_preempt),
    ("fpu", t_fpu),
    ("untyped", t_untyped),
    ("caps", t_caps),
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
    let big = (0..info.untyped_count as usize)
        .filter(|&i| info.untyped[i].kind == KIND_RAM)
        .max_by_key(|&i| info.untyped[i].bits);
    let Some(big) = big else {
        println!("ERROR: roottest: the kernel handed over no ordinary memory");
        exit(false)
    };
    UNTYPED.store(SLOT_FIRST_UNTYPED + big as u64, SeqCst);

    let cmdline = &info.cmdline[..info.cmdline_len as usize];
    let test = cmdline
        .split(|&b| b == b' ' || b == b'\n')
        .find_map(|w| w.strip_prefix(b"test="));
    let Some(name) = test
        .and_then(|n| core::str::from_utf8(n).ok())
        .map(str::trim_ascii)
    else {
        let since_loader = CLOCK.get().since(info.clock[1]);
        println!(
            "BENCH boot_userspace_from_loader_us {} us",
            since_loader / 1000
        );
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
