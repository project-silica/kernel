//! Scheduling: scheduling contexts, the fair and deadline classes, and
//! accounting (`spec/ARCHITECTURE.md`, Scheduler; component C06).
//!
//! Each processor has two run queues of its own threads:
//!
//! - *Deadline*: threads bound to a deadline context with budget left run
//!   first, earliest deadline first. A context's budget is spent by every
//!   thread bound to it and refilled each period (a constant-bandwidth
//!   server); once spent, its threads wait for the next period. Admission
//!   control keeps the reservations on a processor below [`DEADLINE_CAP`].
//! - *Fair*: every other runnable thread, in order of virtual runtime, which
//!   advances by the real time it ran scaled by the default weight over its
//!   weight. The one that has run least (relative to its weight) runs; it is
//!   preempted once it is [`GRANULE`] ahead of the next, and a waking thread
//!   that is well behind the running one preempts it.
//!
//! Time is charged to the processor's *account*: the thread last picked
//! from a run queue. A direct hand-off between threads (a call, a reply)
//! does not change it, so a server running on a client's call spends the
//! client's share or budget: scheduling contexts are donated through IPC,
//! and the fast path reads no clock.
//!
//! Everything here is protected by the home processor's lock, like the
//! threads it schedules.

use crate::hal::Hal;
use crate::kernel::Kernel;
use crate::mem::{KObj, Paddr};
use crate::smp::Cpu;
use crate::thread::RUNNABLE;
use core::cell::Cell;
use silica_abi::{Error, SCHED_DEADLINE, SCHED_DEFAULT_WEIGHT, SCHED_FAIR, SCHED_MAX_WEIGHT};

/// How far a fair thread may run ahead of the one that has run least before
/// it is preempted, in default-weight nanoseconds.
pub(crate) const GRANULE: u64 = 3_000_000;
/// How far behind the one that has run least a waking fair thread is
/// placed at most: a thread that slept long gets this much credit, no more.
const SLEEPER_CREDIT: u64 = 3_000_000;
/// How far behind the running fair thread a waking one must be to take the
/// processor at once: less than its sleeper credit, so a thread that slept
/// runs promptly, more than nothing, so a thread that only paused does not
/// make the processor switch for every message.
const WAKE_GRANULE: u64 = 1_000_000;
/// The shortest time a fair thread runs before it can be preempted again.
const MIN_RUN: u64 = 100_000;
/// The share of a processor deadline reservations may take, in parts of 2^20.
pub(crate) const DEADLINE_CAP: u64 = ((1 << 20) * 9u64).div_ceil(10);
/// The shortest deadline period.
const MIN_PERIOD: u64 = 100_000;

/// A scheduling context.
#[repr(C, align(128))]
pub struct SchedCtx {
    /// `SCHED_FAIR` or `SCHED_DEADLINE`.
    pub(crate) class: Cell<u64>,
    /// Fair: the weight.
    pub(crate) weight: Cell<u64>,
    /// Deadline: budget and period in nanoseconds, and the processor (plus
    /// one; 0 none yet) whose reservation it is.
    pub(crate) budget: Cell<u64>,
    pub(crate) period: Cell<u64>,
    pub(crate) cpu: Cell<u64>,
    /// Deadline: what is left of this period's budget, and when the period ends.
    pub(crate) remaining: Cell<u64>,
    pub(crate) deadline: Cell<u64>,
    /// Nanoseconds charged to it, ever.
    pub(crate) consumed: Cell<u64>,
    /// The threads bound to it (linked through `Thread::sc_next`).
    pub(crate) threads: Cell<Paddr>,
}
impl KObj for SchedCtx {}

impl<H: Hal> Kernel<H> {
    pub(crate) fn sched(&self, s: Paddr) -> &SchedCtx {
        self.phys.obj(s)
    }

    /// A new context: fair, with the default weight.
    pub(crate) fn init_sched(&self, s: Paddr) {
        self.sched(s).weight.set(SCHED_DEFAULT_WEIGHT);
    }

    /// The deadline context thread `t` runs on, if it is bound to one.
    pub(crate) fn dl_of(&self, t: Paddr) -> Option<&SchedCtx> {
        let s = self.thread(t).sc.get();
        (s != 0 && self.sched(s).class.get() == SCHED_DEADLINE).then(|| self.sched(s))
    }

    fn weight_of(&self, t: Paddr) -> u64 {
        match self.thread(t).sc.get() {
            0 => SCHED_DEFAULT_WEIGHT,
            s => self.sched(s).weight.get(),
        }
    }

    /// Put runnable thread `t` (not running) on its processor's run queue:
    /// the deadline queue, or the fair queue in order of virtual runtime,
    /// after any equal (so equals take turns), or with `first` before them
    /// (a thread whose waking took the processor).
    pub(crate) fn enqueue_at(&self, cpu: &Cpu, t: Paddr, first: bool) {
        if self.dl_of(t).is_some() {
            return self.q_push(&cpu.dl_head, &cpu.dl_tail, t);
        }
        let vr = self.thread(t).vr.get();
        // From the back: a thread that yields, or has just run, goes last.
        let mut at = cpu.rq_tail.get();
        while at != 0 && (self.thread(at).vr.get() > vr || (first && self.thread(at).vr.get() == vr)) {
            at = self.thread(at).qprev.get();
        }
        let th = self.thread(t);
        let next = if at == 0 { cpu.rq_head.get() } else { self.thread(at).qnext.get() };
        th.qprev.set(at);
        th.qnext.set(next);
        if at == 0 {
            cpu.rq_head.set(t);
        } else {
            self.thread(at).qnext.set(t);
        }
        if next == 0 {
            cpu.rq_tail.set(t);
        } else {
            self.thread(next).qprev.set(t);
        }
    }

    /// [`Kernel::enqueue_at`] after any equals.
    pub(crate) fn enqueue(&self, cpu: &Cpu, t: Paddr) {
        self.enqueue_at(cpu, t, false);
    }

    /// Take runnable thread `t` (queued, not running) off its run queue.
    pub(crate) fn dequeue(&self, cpu: &Cpu, t: Paddr) {
        if self.dl_of(t).is_some() {
            self.q_remove(&cpu.dl_head, &cpu.dl_tail, t);
        } else {
            self.q_remove(&cpu.rq_head, &cpu.rq_tail, t);
        }
    }

    /// A fair thread that becomes runnable is placed no further back than
    /// the least-run thread minus a bounded credit, so a long sleep does not
    /// buy a long monopoly. A thread that has never run (virtual runtime 0)
    /// gets no credit: it starts level with the least-run thread, so a burst
    /// of new threads does not push aside the ones already waiting.
    pub(crate) fn place(&self, cpu: &Cpu, t: Paddr) {
        // The least-run fair time (only grows): of the running thread and
        // the head of the queue.
        let mut least = u64::MAX;
        let cur = cpu.cur.get();
        if cur != 0 && self.dl_of(cur).is_none() {
            least = self.thread(cur).vr.get();
        }
        let head = cpu.rq_head.get();
        if head != 0 {
            least = least.min(self.thread(head).vr.get());
        }
        if least != u64::MAX {
            cpu.min_vr.set(cpu.min_vr.get().max(least));
        }
        let th = self.thread(t);
        let floor = if th.vr.get() == 0 { cpu.min_vr.get() } else { cpu.min_vr.get().saturating_sub(SLEEPER_CREDIT) };
        th.vr.set(th.vr.get().max(floor).max(1));
    }

    /// Whether `t`, just made runnable on processor `cpu`, should take the
    /// processor from the thread running there now.
    pub(crate) fn preempts(&self, cpu: &Cpu, t: Paddr) -> bool {
        let cur = cpu.cur.get();
        if cur == 0 {
            return false; // an idle processor picks it anyway
        }
        // The running thread is charged only when it stops: bring its
        // account up to now before comparing with it (whoever makes a
        // thread runnable holds that processor's lock).
        self.charge(cpu, self.hal.now());
        match (self.dl_of(t), self.dl_of(cur)) {
            (Some(n), Some(c)) => n.remaining.get() > 0 && n.deadline.get() < c.deadline.get(),
            (Some(n), None) => n.remaining.get() > 0 || n.deadline.get() <= self.hal.now(),
            (None, Some(_)) => false,
            (None, None) => self.thread(t).vr.get().saturating_add(WAKE_GRANULE) <= self.thread(cur).vr.get(),
        }
    }

    /// Charge the time since the last charge to processor `cpu`'s account.
    pub(crate) fn charge(&self, cpu: &Cpu, now: u64) {
        let ran = now.saturating_sub(cpu.since.get());
        cpu.since.set(now);
        let a = cpu.account.get();
        if a == 0 {
            return;
        }
        let th = self.thread(a);
        let s = th.sc.get();
        // The common case: the default share. (Virtual runtime counts
        // nanoseconds: 2^64 of them are 584 years.)
        if s == 0 {
            th.vr.set(th.vr.get().wrapping_add(ran));
            return;
        }
        {
            let sc = self.sched(s);
            sc.consumed.set(sc.consumed.get().saturating_add(ran));
            if sc.class.get() == SCHED_DEADLINE {
                sc.remaining.set(sc.remaining.get().saturating_sub(ran));
                return;
            }
        }
        let w = self.weight_of(a);
        let delta = if w == SCHED_DEFAULT_WEIGHT { ran } else { (ran as u128 * SCHED_DEFAULT_WEIGHT as u128 / w as u128) as u64 };
        th.vr.set(th.vr.get().saturating_add(delta));
    }

    /// A deadline context whose period has ended gets its budget back.
    fn replenish(&self, sc: &SchedCtx, now: u64) {
        if now >= sc.deadline.get() {
            sc.remaining.set(sc.budget.get());
            sc.deadline.set(now.saturating_add(sc.period.get()));
        }
    }

    /// Whether processor `cpu`'s account has spent its deadline budget for
    /// this period (after any refill that is due).
    pub(crate) fn spent(&self, cpu: &Cpu, now: u64) -> bool {
        let a = cpu.account.get();
        if a == 0 {
            return false;
        }
        match self.dl_of(a) {
            Some(sc) => {
                self.replenish(sc, now);
                sc.remaining.get() == 0
            }
            None => false,
        }
    }

    /// The thread processor `cpu` runs next, taken off its queue: the
    /// earliest deadline with budget left, else the least-run fair thread.
    /// 0 when nothing can run.
    #[inline]
    pub(crate) fn pick(&self, cpu: &Cpu, now: u64) -> Paddr {
        // The common case: no deadline work on this processor.
        if cpu.dl_head.get() == 0 {
            return self.q_pop(&cpu.rq_head, &cpu.rq_tail);
        }
        self.pick_deadline(cpu, now)
    }

    #[inline(never)]
    fn pick_deadline(&self, cpu: &Cpu, now: u64) -> Paddr {
        let (mut best, mut best_d) = (0, u64::MAX);
        let mut t = cpu.dl_head.get();
        while t != 0 {
            if let Some(sc) = self.dl_of(t) {
                self.replenish(sc, now);
                if sc.remaining.get() > 0 && sc.deadline.get() < best_d {
                    (best, best_d) = (t, sc.deadline.get());
                }
            }
            t = self.thread(t).qnext.get();
        }
        if best != 0 {
            self.q_remove(&cpu.dl_head, &cpu.dl_tail, best);
            return best;
        }
        self.q_pop(&cpu.rq_head, &cpu.rq_tail)
    }

    /// When processor `cpu`, running its current thread, must next look at
    /// its queues: the current deadline thread's budget running out, the
    /// current fair thread getting [`GRANULE`] ahead of the next one, a
    /// spent deadline context's next period. `u64::MAX`: never.
    #[inline]
    pub(crate) fn preempt_at(&self, cpu: &Cpu, now: u64) -> u64 {
        // The common case: no deadline work and a thread with the default
        // share. The one picked has run least, so it cannot be GRANULE ahead
        // of the next sooner than GRANULE from now: look again then (and
        // the pick decides who runs).
        let a = cpu.account.get();
        if cpu.dl_head.get() == 0 && cpu.cur.get() != 0 && a != 0 && self.thread(a).sc.get() == 0 {
            return if cpu.rq_head.get() == 0 { u64::MAX } else { now.wrapping_add(GRANULE) };
        }
        self.preempt_at_exactly(cpu, now)
    }

    #[inline(never)]
    fn preempt_at_exactly(&self, cpu: &Cpu, now: u64) -> u64 {
        let cur = cpu.cur.get();
        let mut at = u64::MAX;
        // A spent deadline context comes back at its next period. (A queued
        // one with budget left is queued because the running thread comes
        // first; it needs no timer.)
        let mut t = cpu.dl_head.get();
        while t != 0 {
            if let Some(sc) = self.dl_of(t)
                && sc.remaining.get() == 0
            {
                at = at.min(sc.deadline.get());
            }
            t = self.thread(t).qnext.get();
        }
        if cur == 0 {
            return at;
        }
        // The account pays for what runs, so its budget or share decides.
        let a = match cpu.account.get() {
            0 => cur,
            a => a,
        };
        if let Some(sc) = self.dl_of(a) {
            return at.min(now.saturating_add(sc.remaining.get().max(1)));
        }
        let head = cpu.rq_head.get();
        if head != 0 {
            let ahead = (self.thread(head).vr.get() + GRANULE).saturating_sub(self.thread(a).vr.get());
            let w = self.weight_of(a);
            let real = (ahead as u128 * w as u128 / SCHED_DEFAULT_WEIGHT as u128) as u64;
            at = at.min(now.saturating_add(real.max(MIN_RUN)));
        }
        at
    }

    /// `SchedConfigure` on the context at `s`.
    pub(crate) fn sched_configure(&self, s: Paddr, class: u64, a: u64, b: u64, cpu: u64) -> Result<(), Error> {
        let sc = self.sched(s);
        let bound = sc.threads.get() != 0;
        match class {
            SCHED_FAIR => {
                if !(1..=SCHED_MAX_WEIGHT).contains(&a) {
                    return Err(Error::BadArg);
                }
                if bound && sc.class.get() != SCHED_FAIR {
                    return Err(Error::InUse);
                }
                self.release_reservation(sc);
                sc.class.set(SCHED_FAIR);
                sc.weight.set(a);
                Ok(())
            }
            SCHED_DEADLINE => {
                if a == 0 || b < MIN_PERIOD || a > b || cpu >= self.ncpu.get() as u64 {
                    return Err(Error::BadArg);
                }
                if bound && (sc.class.get() != SCHED_DEADLINE || sc.cpu.get() != cpu + 1) {
                    return Err(Error::InUse);
                }
                let share = |budget: u64, period: u64| ((budget as u128) << 20).div_ceil(period as u128) as u64;
                let home = self.cpus.get(cpu as usize);
                let mine =
                    if sc.class.get() == SCHED_DEADLINE && sc.cpu.get() == cpu + 1 { share(sc.budget.get(), sc.period.get()) } else { 0 };
                let total = home.reserved.get() - mine + share(a, b);
                if total > DEADLINE_CAP {
                    return Err(Error::Overload);
                }
                self.release_reservation(sc);
                home.reserved.set(home.reserved.get() + share(a, b));
                sc.class.set(SCHED_DEADLINE);
                sc.budget.set(a);
                sc.period.set(b);
                sc.cpu.set(cpu + 1);
                sc.remaining.set(sc.remaining.get().min(a));
                Ok(())
            }
            _ => Err(Error::BadArg),
        }
    }

    /// Give back the processor share a deadline context reserved.
    fn release_reservation(&self, sc: &SchedCtx) {
        if sc.class.get() == SCHED_DEADLINE && sc.cpu.get() != 0 {
            let home = self.cpus.get(sc.cpu.get() as usize - 1);
            let share = ((sc.budget.get() as u128) << 20).div_ceil(sc.period.get() as u128) as u64;
            home.reserved.set(home.reserved.get().saturating_sub(share));
            sc.cpu.set(0);
        }
    }

    /// Bind thread `t` to the context at `s` (0: none). The caller holds
    /// every lock, or the thread's processor's lock.
    pub(crate) fn sched_bind(&self, t: Paddr, s: Paddr) -> Result<(), Error> {
        let th = self.thread(t);
        let home = self.home(t);
        if s != 0 {
            let sc = self.sched(s);
            if sc.class.get() == SCHED_DEADLINE && sc.cpu.get() != th.cpu.get() + 1 {
                return Err(Error::BadArg);
            }
        }
        let queued = th.state.get() == RUNNABLE && home.cur.get() != t && home.direct.get() != t;
        if queued {
            self.dequeue(home, t);
        }
        self.sched_unlink(t);
        if s != 0 {
            let sc = self.sched(s);
            th.sc_prev.set(0);
            th.sc_next.set(sc.threads.get());
            if sc.threads.get() != 0 {
                self.thread(sc.threads.get()).sc_prev.set(t);
            }
            sc.threads.set(t);
        }
        th.sc.set(s);
        if queued {
            self.enqueue(home, t);
        }
        home.dirty.set(true);
        Ok(())
    }

    /// Take thread `t` off its context's list of bound threads.
    pub(crate) fn sched_unlink(&self, t: Paddr) {
        let th = self.thread(t);
        let s = th.sc.get();
        if s == 0 {
            return;
        }
        let (p, n) = (th.sc_prev.get(), th.sc_next.get());
        if p != 0 {
            self.thread(p).sc_next.set(n);
        } else {
            self.sched(s).threads.set(n);
        }
        if n != 0 {
            self.thread(n).sc_prev.set(p);
        }
        th.sc.set(0);
        th.sc_next.set(0);
        th.sc_prev.set(0);
    }

    /// The last capability to the context at `s` is gone: its threads go
    /// back to the default fair share and its reservation is given back.
    /// The caller holds every lock.
    pub(crate) fn finalise_sched(&self, s: Paddr) {
        loop {
            let t = self.sched(s).threads.get();
            if t == 0 {
                break;
            }
            let _ = self.sched_bind(t, 0);
        }
        self.release_reservation(self.sched(s));
    }
}

impl<H: Hal> Kernel<H> {
    /// Thread `t`'s virtual runtime, and its processor's least virtual
    /// runtime and running thread: the scheduler's view, for tests and the
    /// debug console.
    pub fn sched_view(&self, t: Paddr) -> (u64, u64, Paddr) {
        let cpu = self.home(t);
        (self.thread(t).vr.get(), cpu.min_vr.get(), cpu.cur.get())
    }
}
