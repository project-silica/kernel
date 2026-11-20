//! Threads, the per-processor run queues and timeouts.
//!
//! Every thread has a home processor and runs only there; its queue links,
//! deadline and state are protected by that processor's lock (`smp.rs`).
//! Scheduling is one FIFO run queue per processor with a fixed time slice;
//! scheduling contexts and the fair and deadline classes (component C06)
//! replace it later in M2.

use crate::cap::Slot;
use crate::hal::{Hal, NREGS};
use crate::kernel::Kernel;
use crate::lock::Word;
use crate::mem::{KObj, Paddr};
use crate::smp::Cpu;
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
/// `ipc_flags`: the capability used to send may pass capabilities.
pub(crate) const IPC_GRANT: u64 = 4;

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
    /// The processor this thread runs on.
    pub(crate) cpu: Cell<u64>,
    /// While waiting for a server's reply: the endpoint the call went
    /// through, on whose list of calls in progress this thread is linked.
    pub(crate) via: Cell<Paddr>,
    /// While receiving: the slot of a notification to park the next call on.
    pub(crate) park: Cell<u64>,
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
    /// The processor whose lock alone protects this queue, plus one (0:
    /// none yet). The processor of the thread that last waited to receive.
    pub(crate) owner: Word,
    /// Endpoint: calls delivered through it and not yet answered.
    pub(crate) pend_head: Cell<Paddr>,
    pub(crate) pend_tail: Cell<Paddr>,
}
impl KObj for Queue {}

impl<H: Hal> Kernel<H> {
    pub(crate) fn thread(&self, t: Paddr) -> &Thread {
        self.phys.obj(t)
    }
    pub(crate) fn queue(&self, q: Paddr) -> &Queue {
        self.phys.obj(q)
    }
    /// The processor thread `t` runs on.
    pub(crate) fn home(&self, t: Paddr) -> &Cpu {
        &self.cpus[self.thread(t).cpu.get() as usize]
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
            self.home(t).timers.set(n);
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
        let cpu = self.home(t);
        th.deadline.set(deadline);
        let (mut prev, mut at) = (0, cpu.timers.get());
        while at != 0 && self.thread(at).deadline.get() <= deadline {
            prev = at;
            at = self.thread(at).tnext.get();
        }
        th.tprev.set(prev);
        th.tnext.set(at);
        if prev != 0 {
            self.thread(prev).tnext.set(t);
        } else {
            cpu.timers.set(t);
        }
        if at != 0 {
            self.thread(at).tprev.set(t);
        }
    }

    /// Block thread `t` (current on its processor) in `state` on `on`, with
    /// an optional absolute deadline (0: none).
    pub(crate) fn block(&self, t: Paddr, state: u64, on: Paddr, deadline: u64) {
        let th = self.thread(t);
        th.state.set(state);
        th.blocked_on.set(on);
        self.home(t).dirty.set(true);
        if deadline != 0 {
            self.timer_insert(t, deadline);
        }
    }

    /// Make a blocked or inactive thread runnable, on its own processor.
    /// The caller holds that processor's lock.
    pub(crate) fn make_runnable(&self, t: Paddr) {
        self.timer_remove(t);
        let th = self.thread(t);
        let cpu = self.home(t);
        th.blocked_on.set(0);
        th.state.set(RUNNABLE);
        cpu.dirty.set(true);
        if t != cpu.cur.get() {
            self.q_push(&cpu.rq_head, &cpu.rq_tail, t);
        }
        let home = th.cpu.get() as usize;
        if home != self.hal.cpu() {
            // It may be idle, or running something that should now share.
            self.hal.kick(home, false);
        }
    }

    /// Finish a blocked thread's syscall with `status` and make it runnable.
    pub(crate) fn wake(&self, t: Paddr, status: Result<(), Error>) {
        self.thread(t).regs[H::SYS].set(status.map_or_else(|e| e as u64, |()| 0));
        self.make_runnable(t);
    }

    /// Take `t` off whatever it is blocked on or queued in and leave it
    /// inactive. With `restart`, a syscall that had not yet taken effect is
    /// set up to run again. If `t` is running on another processor, that
    /// processor is brought into the kernel first, so the thread's saved
    /// registers are final when this returns.
    pub(crate) fn cancel(&self, t: Paddr, restart: bool) {
        let th = self.thread(t);
        let cpu = self.home(t);
        let home = th.cpu.get() as usize;
        self.timer_remove(t);
        let on = th.blocked_on.get();
        match th.state.get() {
            RUNNABLE if t != cpu.cur.get() => self.q_remove(&cpu.rq_head, &cpu.rq_tail, t),
            RUNNABLE if home != self.hal.cpu() => self.hal.kick(home, true),
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
                if th.via.get() != 0 {
                    // Waiting for a server thread's reply.
                    self.unpend(t);
                    self.thread(on).caller.set(0);
                } else {
                    // Parked on a notification.
                    let q = self.queue(on);
                    self.q_remove(&q.head, &q.tail, t);
                    q.kind.set(0);
                }
                if th.ipc_flags.get() & IPC_FAULT == 0 {
                    th.regs[H::SYS].set(Error::Gone as u64);
                }
            }
            _ => {}
        }
        th.ipc_flags.set(0);
        th.blocked_on.set(0);
        th.state.set(INACTIVE);
        cpu.dirty.set(true);
    }

    /// Fail the thread waiting for `server`'s reply, if any.
    pub(crate) fn drop_caller(&self, server: Paddr) {
        let c = self.thread(server).caller.replace(0);
        if c == 0 {
            return;
        }
        self.unpend(c);
        let ct = self.thread(c);
        if ct.ipc_flags.replace(0) & IPC_FAULT != 0 {
            ct.blocked_on.set(0);
            ct.state.set(INACTIVE);
            self.home(c).dirty.set(true);
        } else {
            self.wake(c, Err(Error::Gone));
        }
    }

    /// The last capability to thread `t` is gone.
    pub(crate) fn finalise_thread(&self, t: Paddr) {
        self.cancel(t, false);
        self.drop_caller(t);
        let cpu = self.home(t);
        if cpu.loaded.get() == t {
            cpu.loaded.set(0);
        }
        if cpu.cur.get() == t {
            cpu.cur.set(0);
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
                self.home(t).dirty.set(true);
            } else {
                self.wake(t, Err(Error::Gone));
            }
        }
        qu.kind.set(0);
        // Calls delivered through this endpoint and not yet answered fail
        // too: the server's reply would find no caller.
        loop {
            let c = qu.pend_head.get();
            if c == 0 {
                break;
            }
            let server = self.thread(c).blocked_on.get();
            self.drop_caller(server);
        }
    }

    /// The timer interrupt on this processor. Deadlines and the time slice
    /// are looked at by `schedule`, on the way out.
    pub fn on_timer(&self) {
        let me = self.me();
        // The timer is one-shot: having fired, it is no longer armed. It may
        // have fired early: it is only ever reprogrammed to fire sooner.
        me.timer_at.set(u64::MAX);
        me.tick.set(true);
        me.dirty.set(true);
    }

    /// Wake every thread of this processor whose deadline has passed and
    /// note the end of the slice.
    fn expire(&self) {
        let me = self.me();
        let now = self.hal.now();
        loop {
            let t = me.timers.get();
            if t == 0 || self.thread(t).deadline.get() > now {
                break;
            }
            self.cancel(t, false);
            self.wake(t, Err(Error::Timeout));
        }
        if now >= me.slice_end.get() {
            me.resched.set(true);
        }
    }

    /// Hand this processor to `t` on the way out of this kernel entry,
    /// without a pass through the run queue: the receiver of a call, or the
    /// caller a server has just answered. A thread of another processor is
    /// simply made runnable there.
    pub(crate) fn hand_off(&self, t: Paddr) {
        let me = self.me();
        if self.thread(t).cpu.get() as usize != self.hal.cpu() || me.direct.get() != 0 || t == me.cur.get() {
            return self.make_runnable(t);
        }
        self.timer_remove(t);
        let th = self.thread(t);
        th.blocked_on.set(0);
        th.state.set(RUNNABLE);
        me.dirty.set(true);
        me.direct.set(t);
    }

    /// Choose the thread this processor returns to. Called by the
    /// architecture on every way out of the kernel, holding this
    /// processor's lock; when it returns, [`Kernel::current`] is runnable
    /// and loaded.
    pub fn schedule(&self) {
        let me = self.me();
        // Most kernel entries change nothing the scheduler cares about.
        if !me.dirty.get() {
            debug_assert!(self.settled(), "scheduling state changed without `dirty` being set");
            return;
        }
        loop {
            if me.tick.replace(false) {
                self.expire();
            }
            let cur = me.cur.get();
            let cur_ok = cur != 0 && self.thread(cur).state.get() == RUNNABLE;
            let direct = me.direct.replace(0);
            if direct != 0 {
                // A hand-off continues the giver's time slice.
                if cur_ok {
                    self.q_push(&me.rq_head, &me.rq_tail, cur);
                }
                me.cur.set(direct);
                break;
            }
            if cur_ok && !(me.resched.get() && me.rq_head.get() != 0) {
                break;
            }
            let next = self.q_pop(&me.rq_head, &me.rq_tail);
            if next != 0 {
                if cur_ok {
                    self.q_push(&me.rq_head, &me.rq_tail, cur);
                }
                me.cur.set(next);
                me.slice_end.set(self.hal.now() + SLICE_NS);
                break;
            }
            // Nothing to run. The timer is set exactly (or switched off) so
            // an idle processor wakes only when there is something to do,
            // and the lock is released while it waits so other processors
            // can hand it work.
            me.cur.set(0);
            self.arm_timer(true);
            self.hal.idle_begin();
            self.unlock(self.hal.cpu());
            self.hal.idle();
            self.enter();
            // The interrupt may have been the timer: look at deadlines, and
            // program the timer afresh when one is next needed.
            me.tick.set(true);
            me.timer_at.set(u64::MAX);
        }
        me.resched.set(false);
        self.arm_timer(false);
        let cur = me.cur.get();
        let root = self.thread(cur).aspace.cap().map_or(0, |c| c.obj + 4096);
        if me.loaded.get() != cur || me.loaded_root.get() != root {
            self.hal.switch_thread(&self.phys, me.loaded.get(), cur, root);
            me.loaded.set(cur);
            me.loaded_root.set(root);
        }
        me.dirty.set(false);
    }

    /// Whether the scheduler would decide what it decided last time: the
    /// claim `dirty == false` makes, checked in debug builds.
    fn settled(&self) -> bool {
        let me = self.me();
        let cur = me.cur.get();
        cur != 0
            && self.thread(cur).state.get() == RUNNABLE
            && !me.tick.get()
            && !me.resched.get()
            && me.direct.get() == 0
            && me.loaded.get() == cur
            && me.loaded_root.get() == self.thread(cur).aspace.cap().map_or(0, |c| c.obj + 4096)
            && (me.rq_head.get() == 0 || me.timer_at.get() <= me.slice_end.get())
            && (me.timers.get() == 0 || me.timer_at.get() <= self.thread(me.timers.get()).deadline.get())
    }

    /// Make sure this processor's timer fires no later than the next
    /// deadline or the end of the slice. Programming the timer is expensive
    /// (on a virtual machine it leaves the guest), so unless `exact` it is
    /// only ever moved earlier; a timer left too early fires once and is
    /// then set right.
    fn arm_timer(&self, exact: bool) {
        let me = self.me();
        let t = me.timers.get();
        let mut want = if t != 0 { self.thread(t).deadline.get() } else { u64::MAX };
        if me.cur.get() != 0 && me.rq_head.get() != 0 {
            want = want.min(me.slice_end.get());
        }
        let armed = me.timer_at.get();
        if want < armed || (exact && want != armed) {
            me.timer_at.set(want);
            self.hal.set_timer(want);
        }
    }
}
