//! Threads, the run queue and timeouts.
//!
//! M1 scheduling is one FIFO run queue with a fixed time slice; scheduling
//! contexts and the fair and deadline classes (component C06) replace it at M2.

use crate::cap::Slot;
use crate::hal::{Hal, NREGS};
use crate::kernel::Kernel;
use crate::mem::{KObj, Paddr};
use core::cell::Cell;
use core::mem::offset_of;
use silica_abi::Error;

/// Offset of the extended-state (FPU, SIMD) save area inside a Thread object.
pub const FPU_OFFSET: usize = 1024;
/// Time slice when more than one thread is runnable.
pub const SLICE_NS: u64 = 10_000_000;

/// Not runnable: new, suspended, or faulted with nobody to tell.
pub const INACTIVE: u64 = 0;
/// Running or in the run queue.
pub const RUNNABLE: u64 = 1;
/// Queued on an endpoint, waiting to send.
pub const SENDING: u64 = 2;
/// Queued on an endpoint or notification, waiting to receive.
pub const RECEIVING: u64 = 3;
/// Message delivered; waiting for the receiver's reply.
pub const REPLY: u64 = 4;

/// `ipc_flags`: the pending send is a call.
pub(crate) const IPC_CALL: u64 = 1;
/// `ipc_flags`: the message is a fault report built by the kernel.
pub(crate) const IPC_FAULT: u64 = 2;

/// A thread object. The first [`FPU_OFFSET`] bytes are this structure; the
/// rest of the 4 KiB object is the extended-state save area.
#[repr(C, align(64))]
pub struct Thread {
    /// Saved user registers, in the architecture's frame layout.
    pub regs: [Cell<u64>; NREGS],
    pub(crate) cspace: Slot,
    pub(crate) aspace: Slot,
    pub(crate) fault_ep: Slot,
    pub(crate) ipcbuf: Slot,
    pub(crate) state: Cell<u64>,
    pub(crate) qnext: Cell<Paddr>,
    pub(crate) qprev: Cell<Paddr>,
    /// The endpoint or notification queued on, or the thread owing a reply.
    pub(crate) blocked_on: Cell<Paddr>,
    /// The thread whose call this thread has received and not yet answered.
    pub(crate) caller: Cell<Paddr>,
    pub(crate) ipc_badge: Cell<u64>,
    pub(crate) ipc_flags: Cell<u64>,
    pub(crate) deadline: Cell<u64>,
    pub(crate) tnext: Cell<Paddr>,
    pub(crate) tprev: Cell<Paddr>,
    pub(crate) fault: [Cell<u64>; 3],
    pub(crate) reap_next: Cell<u64>,
}
impl KObj for Thread {}
const _: () = assert!(core::mem::size_of::<Thread>() <= FPU_OFFSET);

impl Thread {
    pub(crate) fn embedded_slots(t: Paddr) -> [Paddr; 4] {
        [
            t + offset_of!(Thread, cspace) as u64,
            t + offset_of!(Thread, aspace) as u64,
            t + offset_of!(Thread, fault_ep) as u64,
            t + offset_of!(Thread, ipcbuf) as u64,
        ]
    }
    /// The scheduling state, one of the constants of this module.
    pub fn state(&self) -> u64 {
        self.state.get()
    }
}

/// The waiting queue at the start of every Endpoint and Notification.
#[repr(C, align(64))]
pub struct Queue {
    pub(crate) head: Cell<Paddr>,
    pub(crate) tail: Cell<Paddr>,
    /// Endpoint: whether the queued threads are senders (1) or receivers (2).
    pub(crate) kind: Cell<u64>,
    /// Notification: pending badge bits.
    pub(crate) word: Cell<u64>,
}
impl KObj for Queue {}

impl<H: Hal> Kernel<H> {
    pub(crate) fn thread(&self, t: Paddr) -> &Thread {
        self.phys.obj(t)
    }
    pub(crate) fn queue(&self, q: Paddr) -> &Queue {
        self.phys.obj(q)
    }

    pub(crate) fn q_push(&self, head: &Cell<Paddr>, tail: &Cell<Paddr>, t: Paddr) {
        let th = self.thread(t);
        th.qnext.set(0);
        th.qprev.set(tail.get());
        if tail.get() != 0 {
            self.thread(tail.get()).qnext.set(t);
        } else {
            head.set(t);
        }
        tail.set(t);
    }

    pub(crate) fn q_remove(&self, head: &Cell<Paddr>, tail: &Cell<Paddr>, t: Paddr) {
        let th = self.thread(t);
        let (p, n) = (th.qprev.get(), th.qnext.get());
        if p != 0 {
            self.thread(p).qnext.set(n);
        } else {
            head.set(n);
        }
        if n != 0 {
            self.thread(n).qprev.set(p);
        } else {
            tail.set(p);
        }
        th.qnext.set(0);
        th.qprev.set(0);
    }

    pub(crate) fn q_pop(&self, head: &Cell<Paddr>, tail: &Cell<Paddr>) -> Paddr {
        let t = head.get();
        if t != 0 {
            self.q_remove(head, tail, t);
        }
        t
    }

    fn timer_remove(&self, t: Paddr) {
        let th = self.thread(t);
        if th.deadline.get() == 0 {
            return;
        }
        let (p, n) = (th.tprev.get(), th.tnext.get());
        if p != 0 {
            self.thread(p).tnext.set(n);
        } else {
            self.timers.set(n);
        }
        if n != 0 {
            self.thread(n).tprev.set(p);
        }
        th.deadline.set(0);
        th.tnext.set(0);
        th.tprev.set(0);
    }

    fn timer_insert(&self, t: Paddr, deadline: u64) {
        let th = self.thread(t);
        th.deadline.set(deadline);
        let (mut prev, mut at) = (0, self.timers.get());
        while at != 0 && self.thread(at).deadline.get() <= deadline {
            prev = at;
            at = self.thread(at).tnext.get();
        }
        th.tprev.set(prev);
        th.tnext.set(at);
        if prev != 0 {
            self.thread(prev).tnext.set(t);
        } else {
            self.timers.set(t);
        }
        if at != 0 {
            self.thread(at).tprev.set(t);
        }
    }

    /// Block the current thread `t` in `state` on `on`, with an optional
    /// absolute deadline (0: none).
    pub(crate) fn block(&self, t: Paddr, state: u64, on: Paddr, deadline: u64) {
        let th = self.thread(t);
        th.state.set(state);
        th.blocked_on.set(on);
        if deadline != 0 {
            self.timer_insert(t, deadline);
        }
    }

    /// Make a blocked or inactive thread runnable.
    pub(crate) fn make_runnable(&self, t: Paddr) {
        self.timer_remove(t);
        let th = self.thread(t);
        th.blocked_on.set(0);
        th.state.set(RUNNABLE);
        if t != self.cur.get() {
            self.q_push(&self.rq_head, &self.rq_tail, t);
        }
    }

    /// Finish a blocked thread's syscall with `status` and make it runnable.
    pub(crate) fn wake(&self, t: Paddr, status: Result<(), Error>) {
        self.thread(t).regs[H::SYS].set(status.map_or_else(|e| e as u64, |()| 0));
        self.make_runnable(t);
    }

    /// Take `t` off whatever it is blocked on or queued in. With `restart`,
    /// a syscall that had not yet taken effect is set up to run again.
    pub(crate) fn cancel(&self, t: Paddr, restart: bool) {
        let th = self.thread(t);
        self.timer_remove(t);
        let on = th.blocked_on.get();
        match th.state.get() {
            RUNNABLE if t != self.cur.get() => self.q_remove(&self.rq_head, &self.rq_tail, t),
            SENDING | RECEIVING => {
                let q = self.queue(on);
                self.q_remove(&q.head, &q.tail, t);
                if q.head.get() == 0 {
                    q.kind.set(0);
                }
                if th.ipc_flags.get() & IPC_FAULT != 0 {
                    // The fault was never delivered; it recurs when the thread runs.
                } else if restart {
                    th.regs[H::IP].set(th.regs[H::IP].get().wrapping_sub(H::SYSCALL_LEN));
                }
            }
            REPLY => {
                self.thread(on).caller.set(0);
                if th.ipc_flags.get() & IPC_FAULT == 0 {
                    th.regs[H::SYS].set(Error::Gone as u64);
                }
            }
            _ => {}
        }
        th.ipc_flags.set(0);
        th.blocked_on.set(0);
        th.state.set(INACTIVE);
    }

    /// Fail the thread waiting for `server`'s reply, if any.
    pub(crate) fn drop_caller(&self, server: Paddr) {
        let c = self.thread(server).caller.replace(0);
        if c == 0 {
            return;
        }
        let ct = self.thread(c);
        if ct.ipc_flags.replace(0) & IPC_FAULT != 0 {
            ct.blocked_on.set(0);
            ct.state.set(INACTIVE);
        } else {
            self.wake(c, Err(Error::Gone));
        }
    }

    /// The last capability to thread `t` is gone.
    pub(crate) fn finalise_thread(&self, t: Paddr) {
        self.cancel(t, false);
        self.drop_caller(t);
        if self.loaded.get() == t {
            self.loaded.set(0);
        }
    }

    /// The last capability to an endpoint or notification is gone: everyone
    /// waiting on it fails with `Gone`.
    pub(crate) fn finalise_queue(&self, q: Paddr) {
        let qu = self.queue(q);
        loop {
            let t = self.q_pop(&qu.head, &qu.tail);
            if t == 0 {
                break;
            }
            let th = self.thread(t);
            if th.ipc_flags.replace(0) & IPC_FAULT != 0 {
                self.timer_remove(t);
                th.blocked_on.set(0);
                th.state.set(INACTIVE);
            } else {
                self.wake(t, Err(Error::Gone));
            }
        }
        qu.kind.set(0);
    }

    /// The timer interrupt: note that the slice ended. Expired deadlines are
    /// handled by `schedule`.
    pub fn on_timer(&self) {
        if self.hal.now() >= self.slice_end.get() {
            self.resched.set(true);
        }
    }

    /// Choose the thread to return to. Called by the architecture on every
    /// way out of the kernel; when it returns, [`Kernel::current`] is
    /// runnable and loaded.
    pub fn schedule(&self) {
        loop {
            let now = self.hal.now();
            loop {
                let t = self.timers.get();
                if t == 0 || self.thread(t).deadline.get() > now {
                    break;
                }
                self.cancel(t, false);
                self.wake(t, Err(Error::Timeout));
            }
            let cur = self.cur.get();
            let cur_ok = cur != 0 && self.thread(cur).state.get() == RUNNABLE;
            if cur_ok && !(self.resched.get() && self.rq_head.get() != 0) {
                break;
            }
            let next = self.q_pop(&self.rq_head, &self.rq_tail);
            if next != 0 {
                if cur_ok {
                    self.q_push(&self.rq_head, &self.rq_tail, cur);
                }
                self.cur.set(next);
                self.slice_end.set(now + SLICE_NS);
                break;
            }
            self.cur.set(0);
            self.arm_timer();
            self.hal.idle();
        }
        self.resched.set(false);
        self.arm_timer();
        let cur = self.cur.get();
        let root = self.thread(cur).aspace.cap().map_or(0, |c| c.obj + 4096);
        if self.loaded.get() != cur || self.loaded_root.get() != root {
            self.hal
                .switch_thread(&self.phys, self.loaded.get(), cur, root);
            self.loaded.set(cur);
            self.loaded_root.set(root);
        }
    }

    fn arm_timer(&self) {
        let t = self.timers.get();
        let mut at = if t != 0 {
            self.thread(t).deadline.get()
        } else {
            u64::MAX
        };
        if self.cur.get() != 0 && self.rq_head.get() != 0 {
            at = at.min(self.slice_end.get());
        }
        if self.timer_at.replace(at) != at {
            self.hal.set_timer(at);
        }
    }
}
