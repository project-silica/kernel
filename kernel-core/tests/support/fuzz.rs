//! The syscall fuzzer's interpreter (`spec/TESTING.md` section 3): turns a
//! byte string into syscalls, faults, timer ticks and pending interrupts
//! made by four threads that share the root task's CapSpace and address
//! space, then checks what must hold whatever they did:
//!
//! - the kernel did not panic and every syscall returned or blocked;
//! - the derivation list around every capability is consistent;
//! - after the root task revokes the Untyped everything was made from, no
//!   capability to memory inside it is left anywhere (nothing leaks), and
//!   the Untyped can be retyped whole again (it balances).
//!
//! Used by the cargo-fuzz target `syscalls` and, on fixed pseudo-random
//! inputs, by the host tests.

use super::*;
use silica_abi::root::*;
use silica_abi::{Fault, KEEP, NONBLOCK, ObjectType as T, Syscall as S};

const WINDOW: u64 = 2000;
const NWIN: u64 = 16;
const THREADS: u64 = 1990;
const VADDRS: [u64; 6] = [0x10000, 0x11000, 0x20_0000, 0x4000_0000, 0x7FFF_FFE0_0000, 0xFFFF_8000_0000_0000];

struct Src<'a>(&'a [u8]);
impl Src<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&b, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(b)
    }
    /// One syscall argument: mostly a slot the test uses, sometimes anything.
    fn arg(&mut self, ut: u64) -> Option<u64> {
        let b = self.byte()?;
        Some(match b >> 4 {
            0..=6 => WINDOW + (b & 15) as u64,
            7 => [SLOT_CAPSPACE, SLOT_ASPACE, SLOT_PORTS, SLOT_IPCBUF][b as usize & 3],
            8 => ut,
            9 => THREADS + (b & 3) as u64,
            10 => (b & 15) as u64,
            11 => [12, 13, 16, 21, 6, 3, 2, 1][b as usize & 7],
            12 => VADDRS[b as usize % VADDRS.len()],
            13 => [KEEP, 0, 1, 3, 5, 7, 15, 0x105][b as usize & 7],
            14 => (WINDOW + (b & 15) as u64) | NONBLOCK,
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

/// Run one fuzz input.
pub fn run(data: &[u8]) {
    let m = Machine::new(8);
    let (ut, d) = m.big_untyped();
    for i in 0..3 {
        m.spawn(ut, THREADS + i);
    }
    let ut_addr = m.k.slot_addr(&m.cs, ut).unwrap();
    let mut src = Src(data);
    let mut steps = 0;
    while m.k.can_run() && steps < 4096 {
        steps += 1;
        m.k.schedule();
        let t = m.k.current();
        let Some(op) = src.byte() else { break };
        match op {
            // The architecture's other ways into the kernel.
            240..=243 => {
                m.k.fault([Fault::Page, Fault::Instruction, Fault::Exception][op as usize % 3], 0xdead_0000, op as u64);
                continue;
            }
            244..=247 => {
                m.k.hal.now.set(m.k.hal.now.get() + 3_000_000 * (op as u64 - 243));
                m.k.on_timer();
                continue;
            }
            248..=251 => {
                m.k.hal.pending.set(op as u32 - 247);
                continue;
            }
            _ => {}
        }
        let mut n = op as u64 % 18;
        let mut a = [0u64; 6];
        for x in &mut a {
            let Some(v) = src.arg(ut) else { break };
            *x = v;
        }
        if t == m.root {
            // The root task is the supervisor that cleans up at the end: it
            // never waits, and it does not stop or restart itself.
            match S::from_u64(n) {
                Some(S::Call) => n = S::Send as u64,
                Some(S::Recv) => a[1] = 1,
                Some(S::ReplyRecv) => a[0] = 0,
                Some(S::ThreadStart | S::ThreadControl) if a[0] == SLOT_THREAD => continue,
                _ => {}
            }
            if n == S::Send as u64 {
                a[0] |= NONBLOCK;
            }
        }
        m.raw(n, a);
    }

    // The root task may have destroyed itself (it holds the capabilities to
    // do so); then there is nobody left to ask and the run ends here.
    let rv = m.k.thread_view(m.root);
    let root_ok = m.cap(SLOT_THREAD).is_some_and(|c| c.obj == m.root)
        && m.k.view(rv.slots[0]).is_some_and(|v| v.cap.obj == m.cs.obj && v.cap.bits == 12)
        && m.k.view(ut_addr).is_some_and(|v| v.cap.ty == T::Untyped && v.cap.obj == d.base)
        && m.cap(SLOT_BOOT_UNTYPED).is_some()
        && m.cap(SLOT_CAPSPACE).is_some_and(|c| c.obj == m.cs.obj && c.rights.0 & 2 != 0)
        && rv.state == 1;
    if !root_ok {
        return;
    }
    // Derivation list consistency for every capability in the root CapSpace.
    for i in 1..4096 {
        let s = m.k.slot_addr(&m.cs, i).unwrap();
        let Some(v) = m.k.view(s) else { continue };
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
    let v = m.k.view(ut_addr).unwrap();
    assert_eq!((v.watermark.0, v.next == 0 || m.k.view(v.next).unwrap().depth == 0), (0, true), "the revoked Untyped is not empty");
    let free = (1..4096).find(|&i| m.cap(i).is_none()).expect("a free slot");
    m.retype(ut, T::Frame, d.bits as u64, free, 1);
    assert!(m.k.can_run());
}
