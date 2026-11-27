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
use silica_abi::{Error, Syscall};

/// Offset of the extended-state (FPU, SIMD) save area inside a Thread object.
pub const FPU_OFFSET: usize = 1024;

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
/// Waiting to make again a system call that takes every processor's lock,
/// its processor having used its share of such time (`Kernel::throttled`).
/// Its registers are as the call left them: woken, it makes it again.
pub const THROTTLED: u64 = 5;

/// `ipc_flags`: the pending send is a call.
pub(crate) const IPC_CALL: u64 = 1;
/// `ipc_flags`: the message is a fault report built by the kernel.
pub(crate) const IPC_FAULT: u64 = 2;
/// `ipc_flags`: the capability used to send may pass capabilities.
pub(crate) const IPC_GRANT: u64 = 4;
/// `ipc_flags`: the call was delivered and waits parked on a notification
/// (`blocked_on`), not for a server thread's reply.
pub(crate) const IPC_PARKED: u64 = 8;

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
    /// Links in the queue the thread is on: an endpoint's or a
    /// notification's list, the deadline run queue (a list), or the fair
    /// run queue (a tree, `order.rs`: `qprev` and `qnext` its left and
    /// right child, `rq_up` its parent, `rq_meta` its height).
    pub(crate) qnext: Cell<Paddr>,
    pub(crate) qprev: Cell<Paddr>,
    pub(crate) rq_up: Cell<Paddr>,
    pub(crate) rq_meta: Cell<u64>,
    /// The endpoint or notification queued on, or the thread owing a reply.
    pub(crate) blocked_on: Cell<Paddr>,
    /// The thread whose call this thread has received and not yet answered.
    pub(crate) caller: Cell<Paddr>,
    pub(crate) ipc_badge: Cell<u64>,
    pub(crate) ipc_flags: Cell<u64>,
    pub(crate) deadline: Cell<u64>,
    /// Links in its processor's timer tree (`order.rs`): `tprev` and
    /// `tnext` its left and right child, `t_up` its parent, `t_meta` its
    /// height and place.
    pub(crate) tnext: Cell<Paddr>,
    pub(crate) tprev: Cell<Paddr>,
    pub(crate) t_up: Cell<Paddr>,
    pub(crate) t_meta: Cell<u64>,
    pub(crate) fault: [Cell<u64>; 3],
    pub(crate) reap_next: Cell<u64>,
    /// The processor this thread runs on.
    pub(crate) cpu: Cell<u64>,
    /// While a call is delivered and unanswered: the endpoint it went
    /// through, on whose list of calls in progress this thread is linked.
    pub(crate) via: Cell<Paddr>,
    /// The capability slot the thread's last send, call or receive went
    /// through (0: none, a fault report). When the capability leaves that
    /// slot while the thread still waits, the operation fails with `Gone`.
    pub(crate) via_slot: Cell<Paddr>,
    /// While receiving: the slot of a notification to park the next call on.
    pub(crate) park: Cell<u64>,
    /// The scheduling context it is bound to and runs on (0: none, and it
    /// cannot run), and its links in that context's list of threads.
    pub(crate) sc: Cell<Paddr>,
    pub(crate) sc_next: Cell<Paddr>,
    pub(crate) sc_prev: Cell<Paddr>,
    /// Fair class: virtual runtime (`sched.rs`).
    pub(crate) vr: Cell<u64>,
    /// The notification this thread last listened on while receiving on an
    /// endpoint (it is that notification's listener, `ipc.rs`), and the
    /// capability slot it named it through.
    pub(crate) listen: Cell<Paddr>,
    pub(crate) listen_slot: Cell<Paddr>,
    /// Console text the thread has written with more to follow
    /// (`Syscall::Console`), written out whole with the last piece.
    pub(crate) line: [Cell<u64>; LINE_WORDS],
    pub(crate) line_len: Cell<u64>,
    /// The ticket of its Unmap of a range that stopped part way, kept in
    /// that address space (`vm::Sweep`; 0: none). A thread made anew, or
    /// started on a new program (`ThreadStart`), has none, so it never
    /// goes on with another's.
    pub(crate) sweep_ticket: Cell<u64>,
    /// The slot (plus one) its `CapDelete` emptied while what that left to
    /// empty was not done: the call, made again for that, does not look
    /// at the slot again (where something else may be by then). 0: none.
    pub(crate) deleted: Cell<u64>,
}
impl KObj for Thread {}

/// Words of a thread's console line (`Thread::line`): 192 bytes.
pub const LINE_WORDS: usize = 24;
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
    /// Notification: how many interrupt lines signal it.
    pub(crate) bound: Cell<u64>,
}

impl Queue {
    /// Notification: the thread that last listened on it while receiving
    /// on an endpoint (0: none). Kept where an endpoint keeps the tail of
    /// its calls in progress, which a notification has none of.
    pub(crate) fn listener(&self) -> &Cell<Paddr> {
        &self.pend_tail
    }
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
        self.cpus.get(self.thread(t).cpu.get() as usize)
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
        crate::order::remove(&Timers(self, self.home(t)), t);
        th.deadline.set(0);
    }

    fn timer_insert(&self, t: Paddr, deadline: u64) {
        self.thread(t).deadline.set(deadline);
        crate::order::insert(&Timers(self, self.home(t)), t, false);
    }

    /// Block thread `t` (current on its processor) in `state` on `on`, with
    /// an optional absolute deadline (0 or `u64::MAX`: none). `u64::MAX` is
    /// the timer's "off" (`arm_timer`), so a thread with that deadline would
    /// be on a timer list the timer never fires for: it waits for ever.
    pub(crate) fn block(&self, t: Paddr, state: u64, on: Paddr, deadline: u64) {
        let th = self.thread(t);
        th.state.set(state);
        th.blocked_on.set(on);
        self.home(t).dirty.set(true);
        if deadline != 0 && deadline != u64::MAX {
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
            if self.dl_of(t).is_none() {
                self.place(cpu, t);
            }
            // Whether it takes the processor is decided by that processor,
            // with its own account charged there: a wake from elsewhere only
            // marks it (charging from here would pull the other processor's
            // running thread into this one's cache on every message).
            let local = th.cpu.get() as usize == self.hal.cpu();
            let preempts = local && self.preempts(cpu, t);
            self.enqueue_at(cpu, t, preempts);
            if preempts {
                cpu.resched.set(true);
            } else {
                // When the running thread is next looked at depends on the
                // queue's new head: work it out again on the way out.
                cpu.preempt_at.set(u64::MAX);
                if !local {
                    cpu.woken.set(true);
                }
            }
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
            RUNNABLE if t != cpu.cur.get() => self.dequeue(cpu, t),
            RUNNABLE if home != self.hal.cpu() => {
                self.hal.kick(home, true);
                // Its processor is in the kernel now. If a system call took
                // it there, that call has not taken effect (it waits for a
                // lock this one holds) and is made again when the thread
                // next runs; anything else saved the frame where the thread
                // goes on from.
                if th.regs[H::BY_SYSCALL].get() != 0 {
                    th.regs[H::IP].set(th.regs[H::IP].get().wrapping_sub(H::SYSCALL_LEN));
                }
            }
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
                    // A `ReplyRecv` waiting here has replied already: made
                    // again, it must only receive (with no deadline and
                    // nothing to park on, as its receive was).
                    if th.regs[H::SYS].get() == Syscall::ReplyRecv as u64 {
                        th.regs[H::SYS].set(Syscall::Recv as u64);
                        th.regs[H::ARG[1]].set(0);
                        th.regs[H::ARG[2]].set(0);
                    }
                    // A `SignalRecv` waiting here has signalled already:
                    // made again, it must only receive.
                    if th.regs[H::SYS].get() == Syscall::SignalRecv as u64 {
                        th.regs[H::SYS].set(Syscall::Recv as u64);
                    }
                }
            }
            REPLY => {
                self.unanswer(t);
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

    /// Take thread `t` off its processor, to carry on with what it does on
    /// another ([`Kernel::arrive`]): a thread goes where its scheduling
    /// context is (`Kernel::sched_bind`), and the move is invisible to it.
    /// A runnable one runs there next; one waiting for a reply keeps
    /// waiting and is answered there; one blocked in a send or receive
    /// (nothing delivered yet) makes the same system call again there; an
    /// inactive one stays inactive. Returns whether it is to be made
    /// runnable on arrival. The caller holds every lock.
    pub(crate) fn depart(&self, t: Paddr) -> bool {
        let state = self.thread(t).state.get();
        let again = state != INACTIVE && state != REPLY;
        if again {
            // Off its run queue or wait queue (a running thread's processor
            // is first brought into the kernel), with an undelivered system
            // call set up to run again.
            self.cancel(t, true);
        }
        let old = self.home(t);
        if old.account.get() == t {
            self.charge(old, self.hal.now());
            old.account.set(0);
        }
        for c in [&old.loaded, &old.cur, &old.direct] {
            if c.get() == t {
                c.set(0);
            }
        }
        old.dirty.set(true);
        again
    }

    /// Make processor `to` the home of thread `t`, which left its last
    /// ([`Kernel::depart`], which said whether it runs `again`).
    pub(crate) fn arrive(&self, t: Paddr, to: usize, again: bool) {
        let th = self.thread(t);
        // A call waiting for its answer keeps its time limit, among the new
        // processor's timers.
        let deadline = th.deadline.get();
        if deadline != 0 {
            self.timer_remove(t);
        }
        th.cpu.set(to as u64);
        if deadline != 0 {
            self.timer_insert(t, deadline);
        }
        if again {
            self.make_runnable(t);
        }
    }

    /// Unlink `c`, whose call was delivered and is unanswered, from the
    /// endpoint's list of calls in progress and from whoever owes the answer.
    pub(crate) fn unanswer(&self, c: Paddr) {
        let ct = self.thread(c);
        self.unpend(c);
        let on = ct.blocked_on.get();
        if ct.ipc_flags.get() & IPC_PARKED != 0 {
            let q = self.queue(on);
            q.head.set(0);
            q.tail.set(0);
            q.kind.set(0);
        } else {
            self.thread(on).caller.set(0);
        }
    }

    /// Fail the delivered, unanswered call of `c` with `Gone`. (A thread
    /// whose fault report is the call stays stopped.)
    pub(crate) fn abort_call(&self, c: Paddr) {
        self.unanswer(c);
        let ct = self.thread(c);
        if ct.ipc_flags.replace(0) & IPC_FAULT != 0 {
            ct.blocked_on.set(0);
            ct.state.set(INACTIVE);
            self.home(c).dirty.set(true);
        } else {
            self.wake(c, Err(Error::Gone));
        }
    }

    /// Fail the thread waiting for `server`'s reply, if any.
    pub(crate) fn drop_caller(&self, server: Paddr) {
        let c = self.thread(server).caller.get();
        if c != 0 {
            self.abort_call(c);
        }
    }

    /// The capability in `slot`, to the endpoint or notification `q`, is
    /// leaving the slot while the object lives on: every operation made
    /// through that capability and still waiting fails with `Gone` (a
    /// queued send or call, a receive, a call delivered and unanswered).
    pub(crate) fn revoke_waiters(&self, slot: Paddr, q: Paddr) {
        let qu = self.queue(q);
        if qu.kind.get() != crate::ipc::PARKED {
            let mut t = qu.head.get();
            while t != 0 {
                let th = self.thread(t);
                let next = th.qnext.get();
                if th.via_slot.get() == slot {
                    self.q_remove(&qu.head, &qu.tail, t);
                    th.ipc_flags.set(0);
                    self.wake(t, Err(Error::Gone));
                }
                t = next;
            }
            if qu.head.get() == 0 {
                qu.kind.set(0);
            }
        }
        let mut c = qu.pend_head.get();
        while c != 0 {
            let ct = self.thread(c);
            let next = ct.qnext.get();
            if ct.via_slot.get() == slot {
                self.abort_call(c);
            }
            c = next;
        }
    }

    /// The last capability to thread `t` is gone.
    pub(crate) fn finalise_thread(&self, t: Paddr) {
        self.cancel(t, false);
        // No notification goes on naming it as its listener.
        let n = self.thread(t).listen.replace(0);
        if n != 0 && self.queue(n).listener().get() == t {
            self.queue(n).listener().set(0);
        }
        self.drop_caller(t);
        self.sched_unlink(t);
        let cpu = self.home(t);
        if cpu.account.get() == t {
            cpu.account.set(0);
        }
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
        if qu.kind.get() == crate::ipc::PARKED {
            // The call parked here can no longer be answered.
            self.abort_call(qu.head.get());
        }
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
        // Interrupt lines that signalled this notification go quiet.
        if qu.bound.replace(0) != 0 {
            for (line, irq) in self.irqs.iter().enumerate() {
                if irq.0.get() == q {
                    self.unwatch_line(line);
                    self.irq_stat[line].held.store(false, core::sync::atomic::Ordering::Relaxed);
                    irq.0.set(0);
                    self.hal.irq_mask(line, true);
                }
            }
        }
        // Calls delivered through this endpoint and not yet answered fail
        // too: the server's reply would find no caller.
        loop {
            let c = qu.pend_head.get();
            if c == 0 {
                break;
            }
            self.abort_call(c);
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
    fn expire(&self, now: u64) {
        let me = self.me();
        let mut global = false;
        // The thread this entry took every lock for, whose share was asked
        // before it did (the wait for the locks is its to pay for).
        let mut entered_for = 0;
        loop {
            let t = me.timers.get();
            if t == 0 || self.thread(t).deadline.get() > now {
                break;
            }
            // The links of a queue belong to the processor that owns it. A
            // waiter of this processor in a queue another one owns (several
            // processors receive on it) is taken out holding every lock,
            // and the deadlines are looked at afresh.
            // A call waiting for its answer past its time limit is unlinked from
            // whoever owes it, which may be any processor's, likewise.
            let th = self.thread(t);
            let elsewhere = match th.state.get() {
                SENDING | RECEIVING => self.queue(th.blocked_on.get()).owner.get() != self.hal.cpu() as u64 + 1,
                REPLY => true,
                _ => false,
            };
            if self.ncpu() > 1 && elsewhere {
                // Taking every lock for it is its context's to pay for, as
                // a system call that does is, each one's for itself: one
                // that has used its share has its timeout delivered when
                // the share is back (before, only the first of an entry was
                // asked and paid for every other: the review of
                // audit-kernel-stalls).
                if t != entered_for
                    && let Some(wait) = self.share_back_in(me, t, now)
                {
                    self.timer_remove(t);
                    self.timer_insert(t, now.saturating_add(wait));
                    continue;
                }
                if !global {
                    self.go_global();
                    global = true;
                    entered_for = t;
                    continue;
                }
                entered_for = 0;
                self.cancel(t, false);
                self.wake(t, Err(Error::Timeout));
                self.spent_global(me, t);
                continue;
            }
            if th.state.get() == THROTTLED {
                // Its processor's share has come back: it makes its call
                // again, its registers as the call left them.
                self.make_runnable(t);
                continue;
            }
            self.cancel(t, false);
            self.wake(t, Err(Error::Timeout));
        }
        if global {
            self.go_local();
        }
        if now >= me.preempt_at.get() {
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
    /// and loaded. In line in each way out (two on x86_64: the system call's,
    /// which never returns and so saves no registers, and the rest's).
    #[inline(always)]
    pub fn schedule(&self) {
        let me = self.me();
        // Most kernel entries change nothing the scheduler cares about.
        if !me.dirty.get() {
            debug_assert!(self.settled(), "scheduling state changed without `dirty` being set");
            return;
        }
        loop {
            // The clock is read at most once per pass.
            let mut clock = 0;
            if me.tick.replace(false) {
                clock = self.hal.now();
                self.expire(clock);
            }
            let cur = me.cur.get();
            let cur_ok = cur != 0 && self.thread(cur).state.get() == RUNNABLE;
            let direct = me.direct.replace(0);
            if direct != 0 {
                // A hand-off: whoever was paying for this processor's time
                // still is (`sched.rs`), and no clock is read. A thread
                // another processor woke meanwhile is weighed against the
                // thread handed to, in one more pass.
                if cur_ok {
                    self.enqueue(me, cur);
                }
                me.cur.set(direct);
                if me.woken.get() {
                    continue;
                }
                break;
            }
            // A thread another processor woke: does the first one waiting
            // now come before the running one?
            if me.woken.get() && cur_ok {
                me.woken.set(false);
                let first = if me.dl_head.get() != 0 { me.dl_head.get() } else { self.fair_first(me) };
                if first != 0 && self.preempts(me, first) {
                    me.resched.set(true);
                }
            }
            let others = me.fq_first.get() != 0 || me.dl_head.get() != 0;
            if cur_ok && !me.resched.get() {
                // Keep running; a queue that has just stopped being empty
                // needs a time at which to look again.
                if others && me.preempt_at.get() == u64::MAX {
                    me.preempt_at.set(self.preempt_at(me, self.hal.now()));
                }
                break;
            }
            // Asked to look again (a yield, the timer, a waking thread):
            // charge what ran. A thread alone keeps the processor unless its
            // deadline budget is spent.
            let now = if clock != 0 { clock } else { self.hal.now() };
            self.charge(me, now);
            if cur_ok && !others && !self.spent(me, now) {
                me.yielded.set(false);
                me.preempt_at.set(self.preempt_at(me, now));
                break;
            }
            // Choose afresh: queue the running thread (behind its equals if
            // it yielded) and pick.
            let mut next = 0;
            if cur_ok {
                if me.yielded.replace(false) && me.fq_first.get() != 0 && self.dl_of(cur).is_none() {
                    next = self.yield_within(me, cur);
                    if next == 0 {
                        self.requeue_yielded(me, cur);
                    }
                } else {
                    self.enqueue(me, cur);
                }
            }
            if next == 0 {
                next = self.pick(me, now);
            }
            if next != 0 {
                me.cur.set(next);
                me.account.set(next);
                me.preempt_at.set(self.preempt_at(me, now));
                break;
            }
            me.account.set(0);
            // Nothing to run. The timer is set exactly (or switched off) so
            // an idle processor wakes only when there is something to do,
            // and the lock is released while it waits so other processors
            // can hand it work.
            me.cur.set(0);
            // Wake for the next deadline, or for a spent deadline context's next period.
            let t = me.timers.get();
            let until = if t != 0 { self.thread(t).deadline.get() } else { u64::MAX };
            let until = until.min(self.preempt_at(me, now));
            me.preempt_at.set(u64::MAX);
            // The program's root stays loaded (no switch on the way to idle
            // and back): an address space destroyed meanwhile is left first
            // (`Kernel::evict`, answered while this waits for its lock).
            self.wait_idle(until);
            // The interrupt may have been the timer: look at deadlines. A
            // timer that fired is armed afresh when one is next needed; one
            // still armed fires no later than `until`, so it stays armed
            // (moving it later would leave a virtual machine for nothing,
            // on every wait for a device).
            me.tick.set(true);
            me.timer_at.set(if self.hal.timer_armed() { me.timer_at.get().min(until) } else { u64::MAX });
        }
        me.resched.set(false);
        self.arm_timer();
        let cur = me.cur.get();
        let root = self.thread(cur).aspace.cap().map_or(0, |c| c.obj + 4096);
        if me.loaded.get() != cur || me.loaded_root.get() != root {
            // Into or out of an address space isolated against speculation:
            // looked at, out of line, once the machine has one.
            if self.isolating.load(core::sync::atomic::Ordering::Relaxed) {
                self.isolate(me, cur, root);
            }
            self.hal.switch_thread(&self.phys, me.loaded.get(), cur, root);
            me.loaded.set(cur);
            me.loaded_root.set(root);
        }
        me.dirty.set(false);
    }

    /// This processor is about to run `cur` in address space `root`: if
    /// that is a change of root into or out of an address space isolated
    /// against speculation, nothing either side trained may steer the other
    /// (`Hal::isolate`).
    #[cold]
    #[inline(never)]
    fn isolate(&self, me: &Cpu, cur: Paddr, root: Paddr) {
        let flags = self.thread(cur).aspace.cap().map_or(0, |c| c.flags) & crate::cap::FLAG_ISOLATED;
        if me.loaded_root.get() != root && flags | me.isolated.get() != 0 {
            self.hal.isolate(flags != 0);
            me.isolated.set(flags);
        }
    }

    /// Nothing to run: wait for work, or until `until`, with this
    /// processor's lock let go meanwhile. Returns holding it again.
    #[cold]
    #[inline(never)]
    fn wait_idle(&self, until: u64) {
        let me = self.me();
        let mut words = [crate::hal::Watched::NONE; crate::watch::WATCHED_PER_CPU];
        let watched = self.begin_watches(&mut words);
        self.hal.idle_begin();
        me.lock.unlock();
        let halted = self.hal.idle(until, Some(&words[..watched]));
        self.enter_from_idle();
        if watched == 0 {
            return;
        }
        // The devices interrupt again, and what they wrote meanwhile is
        // delivered (`watch.rs`).
        self.end_watches();
        if halted || me.fq_first.get() != 0 || me.dl_head.get() != 0 || me.direct.get() != 0 || self.hal.now() >= until {
            return;
        }
        // The watch ran its length with nothing: halt, without watching again.
        self.hal.idle_begin();
        me.lock.unlock();
        self.hal.idle(until, None);
        self.enter_from_idle();
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
            && me.timer_at.get() <= me.preempt_at.get()
            && (me.timers.get() == 0 || me.timer_at.get() <= self.thread(me.timers.get()).deadline.get())
    }

    /// Make sure this processor's timer fires no later than the next
    /// deadline or the end of the slice. Programming the timer is expensive
    /// (on a virtual machine it leaves the guest), so it is only ever moved
    /// earlier; a timer left too early fires once and is then set right.
    /// (An idle processor follows the same rule, through [`Hal::idle`].)
    fn arm_timer(&self) {
        let me = self.me();
        let t = me.timers.get();
        let mut want = if t != 0 { self.thread(t).deadline.get() } else { u64::MAX };
        if me.cur.get() != 0 {
            want = want.min(me.preempt_at.get());
        }
        let armed = me.timer_at.get();
        if want < armed {
            me.timer_at.set(want);
            self.hal.set_timer(want);
        }
    }
}

/// A processor's timers as `order.rs` sees them: threads in order of
/// deadline, linked through `tprev` (left), `tnext` (right) and `t_up`.
pub(crate) struct Timers<'a, H: Hal>(pub(crate) &'a Kernel<H>, pub(crate) &'a Cpu);

impl<H: Hal> crate::order::Tree for Timers<'_, H> {
    #[inline(always)]
    fn node(&self, t: Paddr) -> crate::order::Node<'_> {
        let th = self.0.thread(t);
        crate::order::Node { left: &th.tprev, right: &th.tnext, up: &th.t_up, meta: &th.t_meta }
    }
    #[inline(always)]
    fn key(&self, t: Paddr) -> u64 {
        self.0.thread(t).deadline.get()
    }
    fn root(&self) -> &crate::order::Root {
        &self.1.timer_tree
    }
    fn first(&self) -> &Cell<Paddr> {
        &self.1.timers
    }
    fn last(&self) -> &Cell<Paddr> {
        &self.1.timers_last
    }
}
