//! Scheduling: scheduling contexts, the fair and deadline classes, and
//! accounting (`spec/ARCHITECTURE.md`, Scheduler; component C06).
//!
//! Processor time is a capability: a thread runs only on the scheduling
//! context it is bound to, and a context is time on one processor because
//! a holder of that processor's time made it so. Each processor has two
//! run queues of its own threads:
//!
//! - *Deadline*: threads bound to a deadline context with budget left run
//!   first, earliest deadline first. A context's budget is spent by every
//!   thread bound to it and refilled each period (a constant-bandwidth
//!   server); once spent, its threads wait for the next period. Admission
//!   control keeps the reservations on a processor below `DEADLINE_CAP`.
//! - *Fair*: the fair contexts with threads waiting, in order of virtual
//!   runtime, which advances by the real time the context's threads ran
//!   scaled by the default weight over its weight; and in each context its
//!   waiting threads in order of their own, which share it equally. The
//!   context that has run least runs its thread that has run least; it is
//!   preempted once it is `GRANULE` ahead of the next, and a waking thread
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
pub(crate) const GRANULE: u64 = crate::tuned(option_env!("SILICA_TUNE_GRANULE"), 3_000_000);
/// How far behind the one that has run least a waking fair thread is
/// placed at most: a thread that slept long gets this much credit, no more.
const SLEEPER_CREDIT: u64 = crate::tuned(option_env!("SILICA_TUNE_SLEEPER_CREDIT"), 3_000_000);
/// How far behind the running fair thread a waking one must be to take the
/// processor at once: less than its sleeper credit, so a thread that slept
/// runs promptly, more than nothing, so a thread that only paused does not
/// make the processor switch for every message.
const WAKE_GRANULE: u64 = crate::tuned(option_env!("SILICA_TUNE_WAKE_GRANULE"), 1_000_000);
/// The shortest time a fair thread runs before it can be preempted again.
const MIN_RUN: u64 = crate::tuned(option_env!("SILICA_TUNE_MIN_RUN"), 100_000);

/// `ran` nanoseconds of real time in the virtual time of weight `w`.
#[inline(always)]
fn scaled(ran: u64, w: u64) -> u64 {
    if w == SCHED_DEFAULT_WEIGHT { ran } else { (ran as u128 * SCHED_DEFAULT_WEIGHT as u128 / w.max(1) as u128) as u64 }
}
/// How long a processor with nothing to run watches for work before it
/// halts (`Presence::watch`): while it watches, a processor with work for
/// it says so with one store instead of an interrupt, and a deadline that
/// falls inside the watch needs no timer. 0: halt at once. Spinning is
/// power spent on the chance that work arrives soon, so the value is a
/// measured trade (`tools/bench.sh --sweep IDLE_WATCH=...`): about what a
/// halt and its wake-up cost (2 us each in a virtual machine), so a watch
/// that ends unused wastes no more than halting at once would have saved.
/// Swept on the reference host, cross-processor round trips per second on
/// one pair: 0 ns 218,000; 1 us 472,000; 2 us 1,021,000; 5 us 1,041,000;
/// 20 us 1,036,000; 200 us 1,018,000; nothing else moved. Longer only
/// helps a partner slower than a ping; that is decided with a workload
/// that has one.
pub const IDLE_WATCH_NS: u64 = crate::tuned(option_env!("SILICA_TUNE_IDLE_WATCH"), 5_000);
/// The longest an idle processor watches ([`Watch`]): what Linux's guest
/// halt polling allows by default (`guest_halt_poll_ns`), so the two are
/// given the same technique (`spec/TESTING.md` section 4.1).
pub const IDLE_WATCH_MAX_NS: u64 = crate::tuned(option_env!("SILICA_TUNE_IDLE_WATCH_MAX"), 200_000);

/// How long this processor watches for work before it halts, learned from
/// what came of its halts: a halt ended by work within
/// [`IDLE_WATCH_MAX_NS`] means watching that long would have caught it, so
/// the watch grows to cover it next time; a halt that lasted longer means
/// the work it waits for is not coming soon, and the watch shrinks back
/// towards [`IDLE_WATCH_NS`]. A processor waiting on a device (a disk read
/// takes some 20 us) or a slower partner then keeps watching through the
/// wait instead of halting for it, and a processor with nothing to do
/// returns to the short watch within a few long halts. A halt that a
/// deadline ended teaches nothing: the watch already stops at a deadline.
/// One per processor, used by the architecture's idle wait only.
pub struct Watch(Cell<u64>);

impl Watch {
    ctor! {
        /// The short watch.
        pub fn new() -> Watch {
            Watch(Cell::new(IDLE_WATCH_NS))
        }
    }

    /// How long to watch now, in nanoseconds.
    pub fn length(&self) -> u64 {
        self.0.get()
    }

    /// A halt that followed the watch was ended by work (an interrupt, a
    /// poke turned interrupt) after `slept` nanoseconds.
    pub fn woken_after(&self, slept: u64) {
        let now = self.0.get();
        self.0.set(if slept <= IDLE_WATCH_MAX_NS {
            // Enough to have caught it, with a quarter to spare, and at
            // least double: a wait that varies is covered in a step or two.
            now.saturating_mul(2).max(slept + slept / 4).min(IDLE_WATCH_MAX_NS)
        } else {
            (now / 2).max(IDLE_WATCH_NS)
        });
    }
}

impl Default for Watch {
    fn default() -> Watch {
        Watch::new()
    }
}

/// The share of a processor deadline reservations may take, in parts of 2^20.
pub(crate) const DEADLINE_CAP: u64 = ((1 << 20) * 9u64).div_ceil(10);
/// The shortest deadline period.
const MIN_PERIOD: u64 = 100_000;

/// A scheduling context: time on one processor, given by whoever holds
/// that processor's time (`SchedConfigure`). Processor time is a
/// capability: a thread runs only on the context it is bound to, and a
/// context nobody configured is no time at all. A fair context is one
/// share of its processor, divided equally among the threads bound to it;
/// a deadline context is a reservation they spend together.
#[repr(C, align(128))]
pub struct SchedCtx {
    /// `SCHED_FAIR` or `SCHED_DEADLINE`.
    pub(crate) class: Cell<u64>,
    /// Fair: the weight.
    pub(crate) weight: Cell<u64>,
    /// Deadline: budget and period in nanoseconds.
    pub(crate) budget: Cell<u64>,
    pub(crate) period: Cell<u64>,
    /// The processor (plus one) whose time it is; 0: nobody's yet.
    pub(crate) cpu: Cell<u64>,
    /// Deadline: what is left of this period's budget, and when the period ends.
    pub(crate) remaining: Cell<u64>,
    pub(crate) deadline: Cell<u64>,
    /// Nanoseconds charged to it, ever.
    pub(crate) consumed: Cell<u64>,
    /// Time its threads may still spend holding every lock, in nanoseconds
    /// (negative: overdrawn), and when that was last refilled
    /// (`Kernel::throttled`). Both zero in a new context: its first such
    /// call starts it with the burst.
    pub(crate) global_budget: Cell<i64>,
    pub(crate) global_at: Cell<u64>,
    /// The threads bound to it, linked through `Thread::sc_next`.
    pub(crate) threads: Cell<Paddr>,
    /// Fair: its place on its processor.
    pub(crate) share: Share,
}
impl KObj for SchedCtx {}
const _: () = assert!(core::mem::size_of::<SchedCtx>() <= 1 << silica_abi::SCHED_BITS);

/// A fair context's place on its processor.
///
/// Sharing a processor between threads, each its own share, let a program
/// take as much of it as it had memory for threads: a service beside a
/// program of 100 threads had 12 ms of each second and waited 300 ms for
/// its turn (the 2026-11-24 audit; `hostile.rs`). So the fair run queue
/// has two levels: the processor's contexts that have threads waiting, in
/// order of the context's virtual runtime (which advances by the time any
/// of its threads ran, scaled by its weight), and in each its waiting
/// threads in order of their own. The context that has run least runs its
/// thread that has run least. A context is in its processor's tree exactly
/// while it has threads waiting.
#[repr(C)]
pub struct Share {
    /// Links in its processor's tree of contexts (`order.rs`).
    pub(crate) left: Cell<Paddr>,
    pub(crate) right: Cell<Paddr>,
    pub(crate) up: Cell<Paddr>,
    pub(crate) meta: Cell<u64>,
    /// Its virtual runtime.
    pub(crate) vr: Cell<u64>,
    /// The least virtual runtime among its threads (only grows): where a
    /// waking one is placed.
    pub(crate) min_vr: Cell<u64>,
    /// Its waiting threads, a tree in order of virtual runtime: the tree,
    /// its first and last.
    pub(crate) top: crate::order::Root,
    pub(crate) first: Cell<Paddr>,
    pub(crate) last: Cell<Paddr>,
}

impl<H: Hal> Kernel<H> {
    pub(crate) fn sched(&self, s: Paddr) -> &SchedCtx {
        self.phys.obj(s)
    }

    /// A new context: no processor's time yet (`SchedCtx::cpu` is 0).
    pub(crate) fn init_sched(&self, s: Paddr) {
        self.sched(s).weight.set(SCHED_DEFAULT_WEIGHT);
    }

    /// The deadline context thread `t` runs on, if it is bound to one.
    pub(crate) fn dl_of(&self, t: Paddr) -> Option<&SchedCtx> {
        let s = self.thread(t).sc.get();
        (s != 0 && self.sched(s).class.get() == SCHED_DEADLINE).then(|| self.sched(s))
    }

    /// Put runnable thread `t` (not running) on its processor's run queue:
    /// the deadline queue, or among its context's waiting threads in order
    /// of virtual runtime (its context joining the processor's if none of
    /// its threads waited), after any equal (so equals take turns), or with
    /// `first` before them (a thread whose waking took the processor).
    pub(crate) fn enqueue_at(&self, cpu: &Cpu, t: Paddr, first: bool) {
        let s = self.thread(t).sc.get();
        let sc = self.sched(s);
        if sc.class.get() == SCHED_DEADLINE {
            return self.q_push(&cpu.dl_head, &cpu.dl_tail, t);
        }
        let joins = sc.share.first.get() == 0;
        crate::order::insert(&Members(self, &sc.share), t, first);
        if joins {
            crate::order::insert(&Shares(self, cpu), s, first);
        }
    }

    /// Queue fair thread `t`, running on `cpu` and yielding, behind every
    /// thread of its context waiting, and its context behind every context
    /// waiting (some is: the caller saw one), without comparing.
    #[inline(always)]
    pub(crate) fn requeue_yielded(&self, cpu: &Cpu, t: Paddr) {
        let th = self.thread(t);
        let s = th.sc.get();
        let sh = &self.sched(s).share;
        if sh.last.get() != 0 {
            th.vr.set(th.vr.get().max(self.thread(sh.last.get()).vr.get()));
        }
        let waiting = sh.first.get() != 0;
        crate::order::push_last(&Members(self, sh), t);
        // Alone among the contexts waiting, it is in place already.
        let last = cpu.fq_last.get();
        if waiting && last == s && cpu.fq_first.get() == s {
            return;
        }
        if last != s {
            sh.vr.set(sh.vr.get().max(self.sched(last).share.vr.get()));
        }
        let shares = Shares(self, cpu);
        if waiting {
            crate::order::remove(&shares, s);
        }
        crate::order::push_last(&shares, s);
    }

    /// Fair thread `t`, running on `cpu`, yields while its context is the
    /// only one with threads waiting there and no deadline thread waits:
    /// it goes behind every thread of its context and the first of them
    /// runs, which this returns. 0 when that is not the case; then
    /// [`Kernel::requeue_yielded`] and [`Kernel::pick`] decide.
    ///
    /// The same order as queuing `t` last and taking the first, done the
    /// other way round: the first is taken from the n waiting and `t` put
    /// after the n - 1 left, which with one waiting touches no tree at all
    /// (a thread yielding to one other, on every such switch).
    #[inline(always)]
    pub(crate) fn yield_within(&self, cpu: &Cpu, t: Paddr) -> Paddr {
        let th = self.thread(t);
        let s = th.sc.get();
        if cpu.dl_head.get() != 0 || cpu.fq_first.get() != s || cpu.fq_last.get() != s {
            return 0;
        }
        let sh = &self.sched(s).share;
        // In the processor's tree, so it has threads waiting: behind the last.
        th.vr.set(th.vr.get().max(self.thread(sh.last.get()).vr.get()));
        let members = Members(self, sh);
        let next = crate::order::pop(&members);
        crate::order::push_last(&members, t);
        next
    }

    /// The first thread of the least-run context waiting (0: none).
    pub(crate) fn fair_first(&self, cpu: &Cpu) -> Paddr {
        match cpu.fq_first.get() {
            0 => 0,
            s => self.sched(s).share.first.get(),
        }
    }

    /// [`Kernel::enqueue_at`] after any equals.
    pub(crate) fn enqueue(&self, cpu: &Cpu, t: Paddr) {
        self.enqueue_at(cpu, t, false);
    }

    /// Take runnable thread `t` (queued, not running) off its run queue.
    pub(crate) fn dequeue(&self, cpu: &Cpu, t: Paddr) {
        let s = self.thread(t).sc.get();
        let sc = self.sched(s);
        if sc.class.get() == SCHED_DEADLINE {
            return self.q_remove(&cpu.dl_head, &cpu.dl_tail, t);
        }
        crate::order::remove(&Members(self, &sc.share), t);
        if sc.share.first.get() == 0 {
            crate::order::remove(&Shares(self, cpu), s);
        }
    }

    /// A fair thread that becomes runnable is placed no further back than
    /// the least-run thread of its context minus a bounded credit, so a
    /// long sleep does not buy a long monopoly; and a context none of whose
    /// threads waited or ran, no further back than the least-run context
    /// likewise. A thread or context that has never run (virtual runtime
    /// 0) gets no credit: it starts level with the least-run one, so a
    /// burst of new threads does not push aside the ones already waiting.
    pub(crate) fn place(&self, cpu: &Cpu, t: Paddr) {
        let floor = |vr: u64, min: u64| vr.max(if vr == 0 { min } else { min.saturating_sub(SLEEPER_CREDIT) }).max(1);
        let th = self.thread(t);
        let s = th.sc.get();
        let sh = &self.sched(s).share;
        let cur = cpu.cur.get();
        // The running thread's context, when it is a fair one.
        let cur_s = if cur != 0 && self.dl_of(cur).is_none() { self.thread(cur).sc.get() } else { 0 };
        // Within its context: the least-run time (only grows) of the
        // context's running thread and its first waiting one.
        let mut least = if cur_s == s { self.thread(cur).vr.get() } else { u64::MAX };
        if sh.first.get() != 0 {
            least = least.min(self.thread(sh.first.get()).vr.get());
        }
        if least != u64::MAX {
            sh.min_vr.set(sh.min_vr.get().max(least));
        }
        th.vr.set(floor(th.vr.get(), sh.min_vr.get()));
        // Among the contexts, when this one comes back to the processor.
        if sh.first.get() == 0 && cur_s != s {
            let mut least = if cur_s != 0 { self.sched(cur_s).share.vr.get() } else { u64::MAX };
            if cpu.fq_first.get() != 0 {
                least = least.min(self.sched(cpu.fq_first.get()).share.vr.get());
            }
            if least != u64::MAX {
                cpu.min_vr.set(cpu.min_vr.get().max(least));
            }
            sh.vr.set(floor(sh.vr.get(), cpu.min_vr.get()));
        }
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
            (None, None) => {
                // Within one context, by the threads' standing; else by the
                // contexts'.
                let (st, sc) = (self.thread(t).sc.get(), self.thread(cur).sc.get());
                let (a, b) = if st == sc {
                    (self.thread(t).vr.get(), self.thread(cur).vr.get())
                } else {
                    (self.sched(st).share.vr.get(), self.sched(sc).share.vr.get())
                };
                a.saturating_add(WAKE_GRANULE) <= b
            }
        }
    }

    /// Charge the time since the last charge to processor `cpu`'s account:
    /// to its context, and (fair) to its own standing among the context's
    /// threads, which share equally.
    pub(crate) fn charge(&self, cpu: &Cpu, now: u64) {
        let ran = now.saturating_sub(cpu.since.get());
        cpu.since.set(now);
        let a = cpu.account.get();
        if a == 0 {
            return;
        }
        let th = self.thread(a);
        let s = th.sc.get();
        let sc = self.sched(s);
        // (Nanoseconds: 2^64 of them are 584 years.)
        sc.consumed.set(sc.consumed.get().wrapping_add(ran));
        if sc.class.get() == SCHED_DEADLINE {
            sc.remaining.set(sc.remaining.get().saturating_sub(ran));
            return;
        }
        th.vr.set(th.vr.get().wrapping_add(ran));
        let sh = &sc.share;
        let delta = scaled(ran, sc.weight.get());
        // A context in the processor's tree moves to its new place (alone
        // there, it is in place whatever its key).
        if sh.first.get() != 0 && !(cpu.fq_first.get() == s && cpu.fq_last.get() == s) {
            let shares = Shares(self, cpu);
            crate::order::remove(&shares, s);
            sh.vr.set(sh.vr.get().saturating_add(delta));
            crate::order::insert(&shares, s, false);
        } else {
            sh.vr.set(sh.vr.get().wrapping_add(delta));
        }
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
            return self.pick_fair(cpu);
        }
        self.pick_deadline(cpu, now)
    }

    /// The first waiting thread of the least-run context, taken off the
    /// queue (the context with it, if it was its last). 0: none.
    #[inline(always)]
    fn pick_fair(&self, cpu: &Cpu) -> Paddr {
        let s = cpu.fq_first.get();
        if s == 0 {
            return 0;
        }
        let sh = &self.sched(s).share;
        let t = crate::order::pop(&Members(self, sh));
        if sh.first.get() == 0 {
            crate::order::pop(&Shares(self, cpu));
        }
        t
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
        self.pick_fair(cpu)
    }

    /// When processor `cpu`, running its current thread, must next look at
    /// its queues: the current deadline thread's budget running out, the
    /// current fair thread getting [`GRANULE`] ahead of the next one, a
    /// spent deadline context's next period. `u64::MAX`: never.
    #[inline]
    pub(crate) fn preempt_at(&self, cpu: &Cpu, now: u64) -> u64 {
        // The common case: no deadline work and a thread on a fair context
        // of the default weight. The one picked has run least, in its
        // context and its context among the contexts, so neither can be
        // GRANULE ahead of the next sooner than GRANULE from now: look
        // again then (and the pick decides who runs).
        let a = cpu.account.get();
        if cpu.dl_head.get() == 0 && cpu.cur.get() != 0 && a != 0 {
            let sc = self.sched(self.thread(a).sc.get());
            if sc.class.get() == SCHED_FAIR && sc.weight.get() == SCHED_DEFAULT_WEIGHT {
                return if cpu.fq_first.get() == 0 { u64::MAX } else { now.wrapping_add(GRANULE) };
            }
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
        // Within its context: GRANULE ahead of its first waiting thread.
        let s = self.thread(a).sc.get();
        let sc = self.sched(s);
        let head = sc.share.first.get();
        if head != 0 {
            let ahead = (self.thread(head).vr.get() + GRANULE).saturating_sub(self.thread(a).vr.get());
            at = at.min(now.saturating_add(ahead.max(MIN_RUN)));
        }
        // Among the contexts: its own GRANULE ahead of the least-run other one.
        let other = if cpu.fq_first.get() == s { crate::order::next(&Shares(self, cpu), s) } else { cpu.fq_first.get() };
        if other != 0 {
            let ahead = (self.sched(other).share.vr.get() + GRANULE).saturating_sub(sc.share.vr.get());
            let real = (ahead as u128 * sc.weight.get() as u128 / SCHED_DEFAULT_WEIGHT as u128) as u64;
            at = at.min(now.saturating_add(real.max(MIN_RUN)));
        }
        at
    }

    /// `SchedConfigure` on the context at `s`, by a caller with authority
    /// over the time of processors `time` (empty: none). Processor time is
    /// a capability like any other resource: a context is time on
    /// processor `cpu` only because a holder of that processor's time made
    /// it so, and memory alone buys none.
    pub(crate) fn sched_configure(&self, s: Paddr, class: u64, a: u64, b: u64, cpu: u64, time: core::ops::Range<u64>) -> Result<(), Error> {
        let sc = self.sched(s);
        let bound = sc.threads.get() != 0;
        let ok = match class {
            SCHED_FAIR => (1..=SCHED_MAX_WEIGHT).contains(&a),
            SCHED_DEADLINE => a != 0 && b >= MIN_PERIOD && a <= b,
            _ => false,
        };
        if !ok || cpu >= self.ncpu() as u64 {
            return Err(Error::BadArg);
        }
        if !time.contains(&cpu) {
            return Err(Error::NoRights);
        }
        // Its threads are where it is and queued as what it is.
        if bound && (sc.class.get() != class || sc.cpu.get() != cpu + 1) {
            return Err(Error::InUse);
        }
        let share = |budget: u64, period: u64| ((budget as u128) << 20).div_ceil(period as u128) as u64;
        if class == SCHED_DEADLINE {
            let home = self.cpus.get(cpu as usize);
            let mine =
                if sc.class.get() == SCHED_DEADLINE && sc.cpu.get() == cpu + 1 { share(sc.budget.get(), sc.period.get()) } else { 0 };
            if home.reserved.get() - mine + share(a, b) > DEADLINE_CAP {
                return Err(Error::Overload);
            }
            self.release_reservation(sc);
            home.reserved.set(home.reserved.get() + share(a, b));
            sc.budget.set(a);
            sc.period.set(b);
            sc.remaining.set(sc.remaining.get().min(a));
        } else {
            // A new weight changes its pace, not its place.
            self.release_reservation(sc);
            sc.weight.set(a);
        }
        sc.class.set(class);
        sc.cpu.set(cpu + 1);
        Ok(())
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

    /// Bind thread `t` to the context at `s`: from now on it runs on that
    /// context's time, on that context's processor (holding the context's
    /// capability is the authority to take it there), carrying its standing
    /// over to the new context's scale. A context nobody gave a processor's
    /// time is none (`NoTime`). The caller holds every lock.
    pub(crate) fn sched_bind(&self, t: Paddr, s: Paddr) -> Result<(), Error> {
        let sc = self.sched(s);
        if sc.cpu.get() == 0 {
            return Err(Error::NoTime);
        }
        let th = self.thread(t);
        let was = th.sc.get();
        if was == s {
            return Ok(());
        }
        let to = sc.cpu.get() as usize - 1;
        let home = self.home(t);
        let moving = to as u64 != th.cpu.get();
        // Off the run queue it waits on (its place there is its context's);
        // bound for another processor, off whatever it does on this one.
        let again = if moving {
            self.depart(t)
        } else {
            // What it ran until now is the old context's to pay.
            if home.account.get() == t {
                self.charge(home, self.hal.now());
            }
            let queued = th.state.get() == RUNNABLE && home.cur.get() != t && home.direct.get() != t;
            if queued {
                self.dequeue(home, t);
            }
            queued
        };
        let floor = |s: Paddr| if s != 0 && self.sched(s).class.get() == SCHED_FAIR { self.sched(s).share.min_vr.get() } else { 0 };
        let (from, onto) = (floor(was), floor(s));
        self.sched_unlink(t);
        th.sc_next.set(sc.threads.get());
        if sc.threads.get() != 0 {
            self.thread(sc.threads.get()).sc_prev.set(t);
        }
        sc.threads.set(t);
        th.sc.set(s);
        if th.vr.get() != 0 {
            th.vr.set(th.vr.get().saturating_sub(from).saturating_add(onto).max(1));
        }
        if moving {
            self.arrive(t, to, again);
        } else if again {
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

    /// The last capability to the context at `s` is gone, and with it the
    /// time its threads ran on: each stops where it is (suspended, a system
    /// call that had not taken effect set up to be made again) and runs
    /// again once it is bound to other time and resumed. Its reservation
    /// is given back. The caller holds every lock.
    pub(crate) fn finalise_sched(&self, s: Paddr) {
        loop {
            let t = self.sched(s).threads.get();
            if t == 0 {
                break;
            }
            self.cancel(t, true);
            let home = self.home(t);
            if home.account.get() == t {
                self.charge(home, self.hal.now());
                home.account.set(0);
                home.resched.set(true);
            }
            for c in [&home.cur, &home.direct] {
                if c.get() == t {
                    c.set(0);
                }
            }
            self.sched_unlink(t);
        }
        self.release_reservation(self.sched(s));
    }
}

impl<H: Hal> Kernel<H> {
    /// Thread `t`'s virtual runtime and its fair context's, and its
    /// processor's running thread: the scheduler's view, for tests and the
    /// debug console.
    pub fn sched_view(&self, t: Paddr) -> (u64, u64, Paddr) {
        let s = self.thread(t).sc.get();
        (self.thread(t).vr.get(), if s == 0 { 0 } else { self.sched(s).share.vr.get() }, self.home(t).cur.get())
    }
}

/// A processor's fair contexts with threads waiting, as `order.rs` sees
/// them: in order of the context's virtual runtime, linked through the
/// context's own links.
pub(crate) struct Shares<'a, H: Hal>(pub(crate) &'a Kernel<H>, pub(crate) &'a Cpu);

impl<H: Hal> crate::order::Tree for Shares<'_, H> {
    #[inline(always)]
    fn node(&self, s: Paddr) -> crate::order::Node<'_> {
        let sh = &self.0.sched(s).share;
        crate::order::Node { left: &sh.left, right: &sh.right, up: &sh.up, meta: &sh.meta }
    }
    #[inline(always)]
    fn key(&self, s: Paddr) -> u64 {
        self.0.sched(s).share.vr.get()
    }
    fn root(&self) -> &crate::order::Root {
        &self.1.fq_tree
    }
    fn first(&self) -> &Cell<Paddr> {
        &self.1.fq_first
    }
    fn last(&self) -> &Cell<Paddr> {
        &self.1.fq_last
    }
}

/// A fair context's waiting threads, as `order.rs` sees them: in order of
/// virtual runtime, linked through `qprev` (left), `qnext` (right) and
/// `rq_up`.
pub(crate) struct Members<'a, H: Hal>(pub(crate) &'a Kernel<H>, pub(crate) &'a Share);

impl<H: Hal> crate::order::Tree for Members<'_, H> {
    #[inline(always)]
    fn node(&self, t: Paddr) -> crate::order::Node<'_> {
        let th = self.0.thread(t);
        crate::order::Node { left: &th.qprev, right: &th.qnext, up: &th.rq_up, meta: &th.rq_meta }
    }
    #[inline(always)]
    fn key(&self, t: Paddr) -> u64 {
        self.0.thread(t).vr.get()
    }
    fn root(&self) -> &crate::order::Root {
        &self.1.top
    }
    fn first(&self) -> &Cell<Paddr> {
        &self.1.first
    }
    fn last(&self) -> &Cell<Paddr> {
        &self.1.last
    }
}
