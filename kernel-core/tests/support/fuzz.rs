//! The syscall fuzzer's interpreter (`spec/TESTING.md` section 3): turns a
//! byte string into syscalls, faults, timer ticks and pending interrupts
//! made by four threads on two processors that share the root task's
//! CapSpace and address space, then checks what must hold whatever they did:
//!
//! - the kernel did not panic and every syscall returned or blocked;
//! - after every step, each processor's timer list is well formed;
//! - the derivation list around every capability is consistent;
//! - an address space made for devices counts the devices attached to it right;
//! - after the root task revokes the Untyped everything was made from, no
//!   capability to memory inside it is left anywhere (nothing leaks), no
//!   device still reaches memory through tables inside it, and the Untyped
//!   can be retyped whole again (it balances).
//!
//! Used by the cargo-fuzz target `syscalls` and, on fixed pseudo-random
//! inputs, by the host tests.

use super::*;
use silica_abi::root::*;
use silica_abi::{Fault, KEEP, NONBLOCK, ObjectType as T, Syscall as S};

const WINDOW: u64 = 2000;
const NWIN: u64 = 16;
const THREADS: u64 = 1990;
/// The syscall numbers an input can name: every syscall and a few unknown
/// numbers. When the syscalls outgrow it, double it and carry the committed
/// corpus across (see [`Src::arg`]): every seed's syscall bytes change
/// meaning.
const SYSCALL_SPACE: u64 = 32;
const _: () = assert!(S::COUNT <= SYSCALL_SPACE, "more syscalls than the fuzzer's SYSCALL_SPACE: double it and carry the corpus across");
const VADDRS: [u64; 6] = [0x10000, 0x11000, 0x20_0000, 0x4000_0000, 0x7FFF_FFE0_0000, 0xFFFF_8000_0000_0000];

struct Src<'a>(&'a [u8]);
impl Src<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&b, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(b)
    }
    /// One syscall argument: mostly a slot the test uses, sometimes anything.
    ///
    /// The committed corpus (and the regression inputs the host tests name)
    /// is written in this grammar. Extend a table only where no existing
    /// byte changes what it decodes to; a change that does (a modulus, a
    /// category's meaning, SYSCALL_SPACE) must carry the corpus across in
    /// the same commit: run each seed under the old decoding, write the
    /// bytes that decode to the same values under the new one, and check
    /// that each rewritten seed makes the same syscalls with the same
    /// arguments (done once, for the commit that set SYSCALL_SPACE).
    fn arg(&mut self, ut: u64) -> Option<u64> {
        let b = self.byte()?;
        Some(match b >> 4 {
            0..=6 => WINDOW + (b & 15) as u64,
            7 => [SLOT_CAPSPACE, SLOT_ASPACE, SLOT_PORTS, SLOT_IPCBUF, SLOT_DEVICES, SLOT_IRQS, SLOT_DEVICES, SLOT_ASPACE][b as usize & 7],
            8 => ut,
            9 => THREADS + (b & 3) as u64,
            10 => (b & 15) as u64,
            11 => [12, 13, 16, 21, 6, 3, 2, 1][b as usize & 7],
            12 => VADDRS[b as usize % VADDRS.len()],
            13 => [KEEP, 0, 1, 3, 5, 7, 15, 0x105, 0x401, 0x800, 0x9f4, 0x10e, 0x110, 64, 65][b as usize % 15],
            14 if b & 8 == 0 => (WINDOW + (b & 7) as u64) | NONBLOCK,
            // A receive that listens on a notification as well.
            14 => (WINDOW + (b & 7) as u64) | (WINDOW + 8 + (b & 7) as u64) << silica_abi::LISTEN_SHIFT,
            _ => {
                // A wild value, including the root task's own vital slots.
                let mut v = 0u64;
                for _ in 0..(b & 7) + 1 {
                    v = v << 8 | self.byte()? as u64;
                }
                v
            }
        })
    }
}

/// Each address space made for devices that the root CapSpace holds counts
/// its devices right. (One held only in another CapSpace is not looked at
/// here; the check after the final revocation covers every one.)
fn check_devices(m: &Machine) {
    let attached = m.k.hal.attached.borrow();
    if attached.is_empty() {
        return;
    }
    for i in 1..4096 {
        if let Some(c) = m.cap(i).filter(|c| c.ty == T::AddressSpace && c.flags & kernel_core::cap::FLAG_IO != 0) {
            let n = attached.values().filter(|r| **r == c.obj + 4096).count() as u64;
            assert_eq!(m.k.attached_to(c.obj), n, "the device count of the address space at {:#x}", c.obj);
        }
    }
}

/// Run one fuzz input.
pub fn run(data: &[u8]) {
    let m = Machine::with_cpus(8, 2);
    let (ut, d) = m.big_untyped();
    let mut threads = vec![m.root];
    for i in 0..3 {
        threads.push(m.spawn(ut, THREADS + i));
    }
    // Every thread has an IPC buffer, so messages can carry capabilities.
    for i in 0..3 {
        m.retype(ut, T::Frame, 12, THREADS + 4 + i, 1);
        m.ok(S::ThreadConfigure, [THREADS + i, KEEP, KEEP, KEEP, THREADS + 4 + i, 0]);
    }
    // One of them starts out on the second processor.
    m.move_to(THREADS + 2, 1);
    m.ok(S::ThreadControl, [THREADS + 2, 1, 0, 0, 0, 0]);
    let ut_addr = m.k.slot_addr(&m.cs, ut).unwrap();
    let mut src = Src(data);
    let mut steps = 0;
    while steps < 4096 {
        steps += 1;
        if let Err(what) = m.k.check_runqueues() {
            panic!("run queues: {what}");
        }
        if let Err(what) = m.k.check_timers() {
            panic!("before step {steps}: {what}");
        }
        // Which processor makes the next kernel entry.
        let Some(sel) = src.byte() else { break };
        m.on((sel & 1) as usize);
        if !m.k.can_run() {
            m.on(1 - (sel & 1) as usize);
            if !m.k.can_run() {
                break;
            }
        }
        m.settle(m.k.hal.cpu.get());
        let t = m.k.current();
        if t == 0 {
            continue;
        }
        let Some(op) = src.byte() else { break };
        match op {
            // The architecture's other ways into the kernel.
            240..=243 => {
                m.fault([Fault::Page, Fault::Instruction, Fault::Exception][op as usize % 3], 0xdead_0000, op as u64);
                continue;
            }
            244..=247 => {
                m.k.hal.now.set(m.k.hal.now.get().saturating_add(3_000_000 * (op as u64 - 243)));
                m.tick();
                continue;
            }
            248..=251 => {
                m.k.hal.pending.set(op as u32 - 247);
                continue;
            }
            // A thread fills in which capabilities it sends or where it
            // wants received ones (its IPC buffer is its own memory).
            252..=255 => {
                let (Some(b), Some(v)) = (src.byte(), src.arg(ut)) else { break };
                let base = if b & 4 == 0 { silica_abi::IPC_CAPS } else { silica_abi::IPC_RECV };
                // The write goes to the frame the kernel holds as that
                // thread's buffer now. A thread whose buffer was revoked
                // has no mapping of it left to write through (the frame's
                // memory may be another object by then), and a thread that
                // was itself destroyed writes nothing.
                let t = threads[op as usize & 3];
                let alive = (0..4)
                    .any(|i| m.cap(if i == 0 { SLOT_THREAD } else { THREADS + i - 1 }).is_some_and(|c| c.ty == T::Thread && c.obj == t));
                if alive && let Some(f) = m.k.view(m.k.thread_view(t).slots[3]).filter(|f| f.cap.ty == T::Frame) {
                    m.k.phys.write_val(f.cap.obj + (base + (b as usize >> 3 & 3) * 8) as u64, &(v | (b as u64 >> 6) << 60));
                }
                continue;
            }
            _ => {}
        }
        // The syscall number is the byte modulo SYSCALL_SPACE, not modulo
        // the count: a syscall added later takes a number no seed used for
        // anything else (it was an unknown number, answered BadSyscall),
        // so the committed corpus keeps its meaning and its coverage.
        // (Modulo S::COUNT reshuffled every seed with each new syscall: the
        // corpus fell from 1487 to 1108 edges with DeviceAttach unseen.)
        let mut n = op as u64 % SYSCALL_SPACE;
        let mut a = [0u64; 6];
        for x in &mut a {
            let Some(v) = src.arg(ut) else { break };
            *x = v;
        }
        // The corpus was written when a thread was moved by naming a
        // processor, bound with the next operation, and shared a group
        // (class 2): each seed does the same in today's terms. A move
        // is a bind to time on that processor, a group a fair context.
        if n == S::ThreadControl as u64 && a[1] == 2 {
            a[2] = if a[2] & 1 == 0 { SLOT_CONTEXT } else { 1901 };
        } else if n == S::ThreadControl as u64 && a[1] == 3 {
            a[1] = 2;
        } else if n == S::SchedConfigure as u64 && a[1] == 2 {
            a[1] = 0;
        }
        if t == m.root {
            // The root task is the supervisor that cleans up at the end: it
            // never waits, and it does not stop or restart itself.
            match S::from_u64(n) {
                Some(S::Call) => n = S::Send as u64,
                Some(S::Recv | S::SignalRecv) => a[1] = 1,
                Some(S::ReplyRecv) => a[0] = 0,
                Some(S::ThreadStart | S::ThreadControl) if a[0] == SLOT_THREAD => continue,
                _ => {}
            }
            if n == S::Send as u64 {
                a[0] |= NONBLOCK;
            }
        }
        m.raw(n, a);
        check_devices(&m);
        if let Err(what) = m.k.check_runqueues() {
            panic!("run queues: {what}");
        }
        if let Err(what) = m.k.check_timers() {
            panic!("after step {steps} (syscall {n} {a:x?}): {what}");
        }
    }

    // The root task finishes on whichever processor it was last moved to.
    let home = m.k.thread_view(m.root).cpu;
    m.settle(1 - home);
    m.settle(home);
    // The root task may have destroyed itself (it holds the capabilities to
    // do so); then there is nobody left to ask and the run ends here.
    let rv = m.k.thread_view(m.root);
    let root_ok = m.cap(SLOT_THREAD).is_some_and(|c| c.obj == m.root)
        && m.k.view(rv.slots[0]).is_some_and(|v| v.cap.obj == m.cs.obj && v.cap.bits == 12)
        && m.k.view(ut_addr).is_some_and(|v| v.cap.ty == T::Untyped && v.cap.obj == d.base)
        && m.cap(SLOT_BOOT_UNTYPED).is_some()
        && m.cap(SLOT_CAPSPACE).is_some_and(|c| c.obj == m.cs.obj && c.rights.0 & 2 != 0)
        && rv.state == 1
        // Bound to time made from the Untyped it is about to revoke, it
        // would stop itself half way.
        && (rv.sched < d.base || rv.sched >= d.base + (1u64 << d.bits));
    if !root_ok {
        return;
    }
    // Derivation list consistency for every capability in the root CapSpace.
    for i in 1..4096 {
        let s = m.k.slot_addr(&m.cs, i).unwrap();
        let Some(v) = m.k.view(s) else { continue };
        if matches!(v.cap.ty, T::Endpoint | T::Notification)
            && let Err(what) = m.k.check_queue(v.cap.obj, v.cap.ty == T::Notification)
        {
            panic!("slot {i}, {:?}: {what}", v.cap);
        }
        if v.next != 0 {
            let n = m.k.view(v.next).expect("next link points at an empty slot");
            assert_eq!(n.prev, s, "slot {i}: next.prev");
            assert!(n.depth <= v.depth + 1, "slot {i}: a child more than one level deeper");
        }
        if v.prev != 0 {
            assert_eq!(m.k.view(v.prev).expect("prev link points at an empty slot").next, s, "slot {i}: prev.next");
        } else {
            assert_eq!(v.depth, 0, "slot {i}: a derived capability with no predecessor");
        }
        // A watch (`CapNotice`): each slot names the other, and a notice is
        // a notification.
        if v.watched_by != 0 {
            let n = m.k.view(v.watched_by).expect("a watched capability's notice slot is empty");
            assert_eq!((n.notice_of, n.cap.ty), (s, T::Notification), "slot {i}: its notice");
            assert!(v.notice_of == 0, "slot {i}: watched and a notice");
        }
        if v.notice_of != 0 {
            assert_eq!(m.k.view(v.notice_of).expect("a notice watches an empty slot").watched_by, s, "slot {i}: what its notice watches");
        }
    }
    // Nothing leaks, and the Untyped balances.
    let r = m.sys(S::CapRevoke, [ut, 0, 0, 0, 0, 0]);
    assert_eq!(r.status, 0, "the root task cannot revoke its Untyped");
    let (lo, hi) = (d.base, d.base + (1u64 << d.bits));
    for i in 1..4096 {
        if i == ut {
            continue;
        }
        if let Some(c) = m.cap(i) {
            assert!(c.ty == T::Hw || c.obj < lo || c.obj >= hi, "slot {i} still holds {c:?} after its Untyped was revoked");
        }
    }
    for s in rv.slots {
        if let Some(v) = m.k.view(s) {
            assert!(v.cap.obj < lo || v.cap.obj >= hi, "the root thread still uses {:?} from the revoked Untyped", v.cap);
        }
    }
    for (id, root) in m.k.hal.attached.borrow().iter() {
        assert!(*root < lo || *root >= hi, "device {id:#x} still reaches memory through tables in the revoked Untyped");
    }
    let v = m.k.view(ut_addr).unwrap();
    assert_eq!((v.watermark.0, v.next == 0 || m.k.view(v.next).unwrap().depth == 0), (0, true), "the revoked Untyped is not empty");
    let free = (1..4096).find(|&i| m.cap(i).is_none()).expect("a free slot");
    m.retype(ut, T::Frame, d.bits as u64, free, 1);
    assert!(m.k.can_run());
    // Nothing is left on the other processor: its threads came from the Untyped.
    m.on(1 - home);
    assert_eq!(m.k.current(), 0, "a thread from the revoked Untyped still runs on the other processor");
}
