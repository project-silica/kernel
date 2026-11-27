//! Processors and their locks.
//!
//! Kernel state is protected by one lock per processor, used in two ways:
//!
//! - *Local.* A kernel entry takes its own processor's lock. Holding it, it
//!   may change that processor's scheduling state, the threads whose home
//!   it is, and the endpoints and notifications it owns, and it may read
//!   any capability slot or object. Calls between threads on one processor
//!   run entirely like this and never write memory another processor uses.
//! - *Structure.* `Map` adds to an address space and marks a frame
//!   capability as mapped. Nothing a local operation uses changes, so it
//!   keeps its own processor's lock and adds one lock shared by all mapping
//!   (`Kernel::structure`); a fault resolved by a pager never stops the
//!   other processors.
//! - *Pair.* A message between threads of two processors, or through a
//!   queue the other one owns, takes the other processor's lock as well
//!   (in index order) and then checks that everything the operation will
//!   touch belongs to one of the two. Pairs of processors talking to each
//!   other do not wait for the rest of the machine.
//! - *Global.* An operation that changes anything else (capability slots,
//!   mappings being removed, threads or queues of more than two processors) first takes every
//!   processor's lock, in index order, having let go of its own. Holding
//!   them all it is the only code in the kernel, so nothing it changes can
//!   be in use; before it reuses memory it brings every processor that
//!   might still be running on that memory in user mode into the kernel
//!   ([`Hal::kick`] with `wait`). An address-space root is the exception
//!   where a root maps the kernel too ([`Hal::KERNEL_IN_ROOT`], x86_64): a
//!   processor runs the kernel through the root it has loaded, also while
//!   idle or waiting for its lock, so each one on a root being destroyed is
//!   asked to leave it and answers while it waits ([`crate::lock::Leave`],
//!   `Kernel::evict`).
//!
//! A processor waiting for its lock has interrupts off and has already saved
//! its thread's registers, so the holder may change that thread. When it
//! gets the lock it looks at what it was told in the meantime (`Cpu::flush`
//! and the scheduler's `dirty`), and its current thread may no longer exist.
//!
//! The locks are taken and released by the architecture's entry and exit
//! code through [`Kernel::enter`] and [`Kernel::leave`]. The primitives are
//! in `lock.rs`; `tests/loom_models.rs` runs them under every interleaving.

use crate::hal::Hal;
use crate::kernel::Kernel;
pub use crate::lock::{Lock, Word};
use crate::mem::Paddr;
use core::cell::Cell;

/// The time a scheduling context's threads may spend holding every
/// processor's lock, in parts per thousand, divided among the processors
/// (so that a context on each, all flooding, take this much of the
/// machine's time together: `Kernel::throttled`).
pub const GLOBAL_SHARE: u64 = crate::tuned(option_env!("SILICA_TUNE_GLOBAL_SHARE"), 100);
/// What a context may spend holding every lock at once, in nanoseconds,
/// before its share holds it back: a program's start, the boot.
pub const GLOBAL_BURST_NS: u64 = crate::tuned(option_env!("SILICA_TUNE_GLOBAL_BURST"), 5_000_000);

/// Most processors the kernel will run.
pub const MAX_CPUS: usize = 256;

/// How long one processor waits for another (its lock, its entry into the
/// kernel, its leaving a root) before the machine stops, naming it. Every
/// wait between processors is bounded: a processor that never answers (an
/// interrupt lost, a firmware SMI that does not return, a core hung in
/// hardware) would otherwise hold the whole machine with every lock taken
/// and say nothing. Ten seconds is far beyond any wait the kernel makes
/// itself (a global entry is bounded in its work and its share of time).
pub const STALL_NS: u64 = 10_000_000_000;

/// One wait for another processor, timed from its first turn: what
/// [`STALL_NS`] bounds. Reads the clock only while it waits.
pub struct Stall {
    since: u64,
}

impl Stall {
    /// A wait not yet begun.
    pub const fn new() -> Stall {
        Stall { since: 0 }
    }

    /// One turn of waiting, at `now`, for processor `cpu` to do `what`.
    /// Past [`STALL_NS`] the machine stops (see [`stalled`]).
    #[inline]
    pub fn turn(&mut self, now: u64, cpu: usize, what: &str) {
        if self.over(now) {
            stalled(cpu, what);
        }
    }

    /// One turn of waiting, at `now`: whether it has gone past [`STALL_NS`].
    #[inline]
    pub fn over(&mut self, now: u64) -> bool {
        if self.since == 0 {
            self.since = now.max(1);
            return false;
        }
        now.saturating_sub(self.since) > STALL_NS
    }

    /// Whether the wait has gone on longer than `ns` (for one retry before
    /// giving up, as a kick sent again).
    pub fn longer_than(&self, now: u64, ns: u64) -> bool {
        self.since != 0 && now.saturating_sub(self.since) > ns
    }
}

impl Default for Stall {
    fn default() -> Stall {
        Stall::new()
    }
}

/// Processor `cpu` did not `what` within [`STALL_NS`]: stop the machine,
/// naming it, rather than hang with every lock held.
#[cold]
#[inline(never)]
pub fn stalled(cpu: usize, what: &str) -> ! {
    panic!("processor {cpu} has not {what} in 10 s: the machine stops")
}
const _: () = assert!(MAX_CPUS.is_power_of_two());

/// Every processor's [`Cpu`], laid out so that no processor's state shares
/// page offsets with the busy part of a thread object.
///
/// A thread object is page aligned and its first `size_of::<Thread>()`
/// bytes (registers, embedded capability slots, scheduling links) are
/// written on every kernel entry; per-processor state is read and written
/// on every entry too. On x86 a load whose address matches an earlier
/// store's in its low 12 bits waits for that store ("4K aliasing"): when a
/// processor's state sat at the page offsets of the register frame, every
/// system call on that one processor stalled (measured: one processor of
/// eight ran IPC 30% slower, with 5.6 million aliased loads where the others
/// had none). So each page of this table holds processors' state only in
/// its upper half, above the thread object's size.
#[repr(C, align(4096))]
pub struct CpuTable([Cpu; CPU_SLOTS]);

const PER_PAGE: usize = 4096 / core::mem::size_of::<Cpu>();
/// The upper half of each page: a power of two, so finding a processor's
/// state (on every kernel entry, many times) is shifts and masks.
const USED: usize = PER_PAGE / 2;
const SKIP: usize = PER_PAGE - USED;
const CPU_SLOTS: usize = MAX_CPUS.div_ceil(USED) * PER_PAGE;
const _: () = assert!(
    CPU_SLOTS.is_power_of_two()
        && 4096 % core::mem::size_of::<Cpu>() == 0
        && USED.is_power_of_two()
        && SKIP * core::mem::size_of::<Cpu>() >= core::mem::size_of::<crate::thread::Thread>()
);

/// Where processor `i`'s state is in the table: what an architecture
/// keeps in its per-processor data for [`Hal::cpu_slot`].
pub const fn cpu_slot(i: usize) -> usize {
    i / USED * PER_PAGE + SKIP + i % USED
}

impl CpuTable {
    pub(crate) const fn new() -> CpuTable {
        CpuTable([const { Cpu::new() }; CPU_SLOTS])
    }
    /// Processor `i`'s state.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> &Cpu {
        &self.0[cpu_slot(i)]
    }
    /// The state at slot `s` (from [`cpu_slot`]); the slot count is a power
    /// of two, so masking replaces the bounds check.
    #[inline]
    pub(crate) fn at_slot(&self, s: usize) -> &Cpu {
        &self.0[s & (CPU_SLOTS - 1)]
    }
    /// The first `n` processors' states, in order. By slot, masked
    /// ([`CpuTable::at_slot`]): every global transition walks this, and a
    /// bounds check on each processor cost it two instructions apiece.
    pub(crate) fn iter(&self, n: usize) -> impl Iterator<Item = &Cpu> {
        (0..n).map(move |i| self.at_slot(cpu_slot(i)))
    }
}

/// One processor's lock and scheduling state (protected by the lock). Each
/// has its own cache lines: processors working independently must not make
/// each other's memory bounce. Its size is a power of two that divides a
/// page ([`CpuTable`]): 512 bytes since the scheduling groups and the share
/// of time holding every lock took it past 256.
#[repr(align(512))]
pub struct Cpu {
    pub(crate) lock: Lock,
    /// The thread it runs (0 while idle).
    pub(crate) cur: Cell<Paddr>,
    /// The thread whose registers and extended state are loaded.
    pub(crate) loaded: Cell<Paddr>,
    /// The address-space root that is loaded.
    pub(crate) loaded_root: Cell<Paddr>,
    /// `FLAG_ISOLATED` while the last address space it ran is isolated
    /// against speculation (`ASPACE_ISOLATED`), else 0. Kept when the root
    /// is forgotten (`forget_root`): the next switch, whatever it loads,
    /// still leaves the isolated space's predictions behind.
    pub(crate) isolated: Cell<u8>,
    /// The fair run queue: the fair contexts with threads waiting, a tree
    /// in order of the context's virtual runtime (`order.rs`,
    /// `sched::Share`): its first (least-run) and last, and the tree.
    pub(crate) fq_first: Cell<Paddr>,
    pub(crate) fq_last: Cell<Paddr>,
    pub(crate) fq_tree: crate::order::Root,
    /// Threads with deadlines, a tree in deadline order: the soonest, the
    /// latest, and the tree.
    pub(crate) timers: Cell<Paddr>,
    pub(crate) timers_last: Cell<Paddr>,
    pub(crate) timer_tree: crate::order::Root,
    /// When the timer is set to fire (`u64::MAX`: not armed).
    pub(crate) timer_at: Cell<u64>,
    /// When the running thread must next be looked at again (its share or
    /// budget used up; `u64::MAX`: never), see `sched.rs`.
    pub(crate) preempt_at: Cell<u64>,
    /// Runnable threads bound to deadline contexts.
    pub(crate) dl_head: Cell<Paddr>,
    pub(crate) dl_tail: Cell<Paddr>,
    /// The least virtual runtime among this processor's fair contexts (only grows).
    pub(crate) min_vr: Cell<u64>,
    /// When time was last charged, and to which thread (0: none).
    pub(crate) since: Cell<u64>,
    pub(crate) account: Cell<Paddr>,
    /// Deadline reservations on this processor, in parts of 2^20.
    pub(crate) reserved: Cell<u64>,
    /// The running thread yielded: it goes behind its equals.
    pub(crate) yielded: Cell<bool>,
    /// Another processor made a thread of this one runnable: whether it
    /// should take the processor is decided here, on the way out.
    pub(crate) woken: Cell<bool>,
    pub(crate) resched: Cell<bool>,
    /// The timer fired since the scheduler last looked at deadlines.
    pub(crate) tick: Cell<bool>,
    /// The thread to switch to directly at the end of this kernel entry.
    pub(crate) direct: Cell<Paddr>,
    /// Something changed that the scheduler must look at on the way out.
    pub(crate) dirty: Cell<bool>,
    /// Another processor changed mappings of the loaded address space:
    /// drop cached translations before running user code again.
    pub(crate) flush: Cell<bool>,
    /// The other processor whose lock this kernel entry also holds, plus
    /// one (0: none).
    pub(crate) pair: Cell<u64>,
    /// How many of this processor's kernel entries took a second lock.
    pub(crate) pairs: Cell<u64>,
    /// When the global entry under way took the locks ([`Kernel::throttled`]).
    pub(crate) global_from: Cell<u64>,
    /// A global entry's request that this processor stop running on a root
    /// it is about to reuse ([`Kernel::evict`]), answered while this
    /// processor waits for a lock without holding its own.
    /// Last: read only when a lock is contended, so it moves nothing the
    /// entry paths use.
    pub(crate) leave: crate::lock::Leave,
    /// When the lock wait under way began ([`Kernel::lock_wait_turn`]).
    /// Touched only by this processor, also while it does not hold its own
    /// lock; after `leave` for the same reason.
    pub(crate) wait_since: Cell<u64>,
    /// Turns this processor has spent waiting for a lock, ever (counted
    /// only while it waits, so an entry that finds its lock free pays
    /// nothing). A long operation holding every lock reads the others':
    /// one that has moved since it last looked is a processor waiting for
    /// it ([`Kernel::must_pause`]).
    pub(crate) spins: core::sync::atomic::AtomicU64,
    /// [`Kernel::must_pause`]'s view of the others' `spins`: the global
    /// entry it was taken in (`globals`) and their sum then.
    pub(crate) pause_seen: Cell<(u64, u64)>,
    /// An address space whose mappings this global entry removed: the
    /// other processors are told once, when it lets go of every lock
    /// ([`Kernel::go_local`]), not once a mapping (0: none).
    pub(crate) shoot: Cell<Paddr>,
    /// A capability to an address space went in this global entry: every
    /// other processor looks again at what it should have loaded, told
    /// once when the entry lets go of every lock.
    pub(crate) reload: Cell<bool>,
}

const _: () = assert!(core::mem::size_of::<Cpu>() == 512);

/// The processors whose locks a kernel entry holds.
#[derive(Clone, Copy)]
pub(crate) struct Held {
    me: u64,
    /// The second processor plus one (0: none).
    pair: u64,
    all: bool,
}

impl Held {
    /// Whether processor `cpu`'s lock is among them.
    pub(crate) fn has(self, cpu: u64) -> bool {
        cpu == self.me || cpu.wrapping_add(1) == self.pair || self.all
    }
}

impl Cpu {
    pub(crate) const fn new() -> Cpu {
        Cpu {
            lock: Lock::new(),
            cur: Cell::new(0),
            loaded: Cell::new(0),
            loaded_root: Cell::new(0),
            isolated: Cell::new(0),
            fq_first: Cell::new(0),
            fq_last: Cell::new(0),
            fq_tree: crate::order::Root::new(),
            timers: Cell::new(0),
            timers_last: Cell::new(0),
            timer_tree: crate::order::Root::new(),
            timer_at: Cell::new(u64::MAX),
            preempt_at: Cell::new(u64::MAX),
            dl_head: Cell::new(0),
            dl_tail: Cell::new(0),
            min_vr: Cell::new(0),
            since: Cell::new(0),
            account: Cell::new(0),
            reserved: Cell::new(0),
            yielded: Cell::new(false),
            woken: Cell::new(false),
            resched: Cell::new(false),
            tick: Cell::new(false),
            direct: Cell::new(0),
            dirty: Cell::new(true),
            flush: Cell::new(false),
            pair: Cell::new(0),
            pairs: Cell::new(0),
            global_from: Cell::new(0),
            leave: crate::lock::Leave::new(),
            wait_since: Cell::new(0),
            spins: core::sync::atomic::AtomicU64::new(0),
            pause_seen: Cell::new((0, 0)),
            shoot: Cell::new(0),
            reload: Cell::new(false),
        }
    }

    /// Whether it has address-space root `root` loaded.
    pub(crate) fn runs_on(&self, root: Paddr) -> bool {
        self.loaded_root.get() == root
    }

    /// It no longer has a root loaded that it may keep using (the root was
    /// destroyed, or its translations flushed): the next switch loads one.
    /// Whether the last one was isolated is kept (`isolated`).
    pub(crate) fn forget_root(&self) {
        self.loaded_root.set(0);
    }
}

impl<H: Hal> Kernel<H> {
    /// This processor's state.
    pub(crate) fn me(&self) -> &Cpu {
        // On every path through the kernel, many times: one load where the
        // architecture keeps the address ([`Hal::cpu_state`]), else the
        // slot masked rather than bounds-checked (a processor's index is
        // below MAX_CPUS, a power of two, by construction).
        let me = self.hal.cpu_state(&self.cpus);
        debug_assert!(core::ptr::eq(me, self.cpus.get(self.hal.cpu())));
        me
    }

    /// Processor `cpu`'s count of turns spent waiting for a lock: what a
    /// long operation holding every lock looks at to know that another
    /// processor waits for it (`Kernel::must_pause`). For tests, whose
    /// processors do not run at once and so count for one that waits.
    pub fn lock_wait_turns(&self, cpu: usize) -> &core::sync::atomic::AtomicU64 {
        &self.cpus.get(cpu).spins
    }

    /// Processor `i`'s state, for an architecture to record beside the
    /// processor's index when it starts ([`Hal::cpu_state`]).
    pub fn cpu_state_of(&self, i: usize) -> &Cpu {
        self.cpus.get(i)
    }

    /// Number of processors the machine has (set once at boot).
    pub fn cpu_count(&self) -> usize {
        self.ncpu()
    }

    /// Tell the kernel how many processors there are. Called by the boot
    /// processor during boot only: with what the firmware describes before
    /// the others start, and with how many arrived before the root task
    /// starts (`Admission`), so a secondary may read either while it waits
    /// for work; both are counts of processors whose state is initialised.
    pub fn set_cpu_count(&self, n: usize) {
        self.ncpu.store(n.clamp(1, MAX_CPUS), core::sync::atomic::Ordering::Release);
    }

    /// The number of processors, as last set (`set_cpu_count`). A plain
    /// load on every architecture: as cheap as the field it was.
    #[inline(always)]
    pub(crate) fn ncpu(&self) -> usize {
        self.ncpu.load(core::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn unlock(&self, cpu: usize) {
        self.cpus.get(cpu).lock.unlock();
    }

    /// Enter the kernel on this processor: take its lock and act on what
    /// other processors asked of it while it was away.
    pub fn enter(&self) {
        // `me` found once: the answer may load a root (a write the compiler
        // cannot see past), and finding it again after the wait cost every
        // entry an instruction.
        let me = self.me();
        me.lock.lock_while(|began| self.lock_wait_turn(me, &me.lock, began));
        self.catch_up(me);
    }

    /// [`Kernel::enter`] for the scheduler coming back from idle: out of
    /// line, so that the way out of the kernel, which does not idle, is not
    /// laid out around it (`bench.sh --only insns`).
    #[cold]
    #[inline(never)]
    pub(crate) fn enter_from_idle(&self) {
        self.enter();
    }

    /// Answer a global entry's request to stop running on a root
    /// ([`Kernel::evict`]): called on every turn of a wait for a lock while
    /// this processor does not hold its own. Only where a root maps the
    /// kernel ([`Hal::KERNEL_IN_ROOT`]) is one ever made.
    fn answer_leave(&self, me: &Cpu) {
        if H::KERNEL_IN_ROOT {
            let leave = &me.leave;
            let root = leave.asked();
            if root != 0 {
                self.hal.leave_aspace(root);
                leave.answer();
            }
        }
    }

    /// One turn of this processor's wait for lock `l` (its own, entering
    /// the kernel, or another's, taking more locks than its own:
    /// [`crate::lock::go_global`], [`crate::lock::go_pair`]): answer what
    /// other processors ask of it ([`Kernel::answer_leave`]), and once the
    /// lock has not been seen free for [`STALL_NS`] stop the machine,
    /// naming the processor whose lock it is (`began`: the lock was just
    /// found held, [`Lock::lock_while`]).
    ///
    /// The wait is timed in the processor's own record, not on the stack,
    /// so a transition that finds every lock free sets nothing up, and the
    /// wait's only call is one that does not return: the entry paths keep
    /// the frame they had without the bound (a bound kept on the stack, or
    /// a call that returns, cost a page fault 16 to 23 instructions in
    /// saved registers).
    #[inline(always)]
    fn lock_wait_turn(&self, me: &Cpu, l: &Lock, began: bool) {
        self.answer_leave(me);
        // Only this processor writes it: no locked instruction.
        me.spins.store(me.spins.load(core::sync::atomic::Ordering::Relaxed).wrapping_add(1), core::sync::atomic::Ordering::Relaxed);
        let now = self.hal.now();
        if began {
            me.wait_since.set(now);
        } else if now.saturating_sub(me.wait_since.get()) > STALL_NS {
            self.stalled_on(l);
        }
    }

    /// The machine stops: lock `l` was not seen free within [`STALL_NS`].
    #[cold]
    #[inline(never)]
    fn stalled_on(&self, l: &Lock) -> ! {
        let cpu = (0..self.ncpu()).find(|&i| core::ptr::eq(&self.cpus.get(i).lock, l)).unwrap_or(usize::MAX);
        if cpu == self.hal.cpu() {
            // Its own lock, held by a processor taking every lock (which
            // one is not recorded): that processor is the one stalled.
            stalled(cpu, "had its lock back from a processor holding every lock")
        }
        stalled(cpu, "let go of its lock")
    }

    /// Enter the kernel on this processor, whose lock the architecture's
    /// entry has already taken (`cpu_lock`): act on what other processors
    /// asked of it while it was away.
    pub fn entered(&self) {
        self.caught_up();
    }

    /// Processor `cpu`'s lock, for an architecture's entry code to take
    /// before it saves anything (then calling [`Kernel::entered`]).
    pub fn cpu_lock(&self, cpu: usize) -> &Lock {
        &self.cpus.get(cpu).lock
    }

    fn caught_up(&self) {
        self.catch_up(self.me());
    }

    fn catch_up(&self, me: &Cpu) {
        if me.flush.replace(false) {
            self.hal.tlb_reset();
            me.forget_root();
            me.dirty.set(true);
        }
    }

    /// Leave the kernel on this processor.
    pub fn leave(&self) {
        self.me().lock.unlock();
    }

    /// Whether a long operation holding every lock (emptying, revoking,
    /// unmapping) should stop here and be made again: an interrupt is
    /// pending on this processor, or another processor has been waiting for
    /// a lock since this entry last looked (its `spins` moved). Before, only
    /// the interrupt was asked, and a processor that had none (one with no
    /// timer armed: nothing else interrupts it) held every lock until the
    /// whole operation was done, for as long as the memory a program had
    /// paid for let it make that (the 2026-11-27 audit: H4, N1). Asked every
    /// so many steps; the first time in a global entry it only takes note.
    pub(crate) fn must_pause(&self) -> bool {
        if self.hal.irq_pending() {
            return true;
        }
        if self.ncpu() == 1 || !self.global.get() {
            return false;
        }
        let me = self.me();
        let sum = self.others().fold(0u64, |s, (_, c)| s.wrapping_add(c.spins.load(core::sync::atomic::Ordering::Relaxed)));
        let (entry, seen) = me.pause_seen.get();
        let now = self.globals.get();
        me.pause_seen.set((now, sum));
        entry == now && seen != sum
    }

    /// From holding this processor's lock to holding every processor's.
    /// Other global operations may run in between: the caller re-examines
    /// everything it looked at before.
    pub(crate) fn go_global(&self) {
        let me = self.me();
        crate::lock::go_global(&me.lock, self.cpus.iter(self.ncpu()).map(|c| &c.lock), |l, began| self.lock_wait_turn(me, l, began));
        self.global.set(true);
        self.globals.set(self.globals.get() + 1);
        self.catch_up(me);
    }

    /// Whether system call `t` made, about to take every lock (holding only
    /// this processor's), must wait for its scheduling context's share of
    /// the time spent so: then it is set up to be made again and waits, and
    /// this returns true. Taking every lock stops the whole machine, so a
    /// program on processors of its own that did nothing else (capability
    /// copies and deletes, maps and unmaps, inspections) took two thirds
    /// of the round trips of programs on the others (the 2026-11-24 audit;
    /// `smp_global_flood`). The time spent holding every lock is a resource
    /// like processor time, and whoever spends it pays: each scheduling
    /// context may hold them for [`GLOBAL_SHARE`] parts per thousand of the
    /// time, divided among the processors, with [`GLOBAL_BURST_NS`] in hand
    /// for bursts (a program's start, the boot). The calls of a context
    /// that has used its share wait until it has some again; nobody else's
    /// do, on its processor or any other. In line: two loads and a branch
    /// unless the share is used up.
    #[inline(always)]
    pub(crate) fn throttled(&self, me: &Cpu, t: Paddr) -> bool {
        let sc = self.sched(self.thread(t).sc.get());
        if sc.global_budget.get() > 0 {
            me.global_from.set(self.hal.now());
            return false;
        }
        self.throttle(me, sc, t)
    }

    #[cold]
    #[inline(never)]
    fn throttle(&self, me: &Cpu, sc: &crate::sched::SchedCtx, t: Paddr) -> bool {
        self.throttle_as(me, sc, t, true)
    }

    /// [`Kernel::throttled`] for a fault that is to take every lock (its
    /// fault endpoint another processor's): a thread whose context has
    /// used its share waits for it and then faults again where it was, as
    /// a system call is made again. Faults cost a program nothing to make,
    /// and each took every lock unthrottled (the 2026-11-27 audit: M10).
    pub(crate) fn throttled_fault(&self, me: &Cpu, t: Paddr) -> bool {
        let sc = self.sched(self.thread(t).sc.get());
        if sc.global_budget.get() > 0 {
            me.global_from.set(self.hal.now());
            return false;
        }
        self.throttle_as(me, sc, t, false)
    }

    /// For the timeouts of thread `t`, waiting in a queue another
    /// processor owns, which are delivered holding every lock (`expire`):
    /// when its context has used its share of that time, how long until it
    /// is back (the timeout is delivered then); else `None`, the entry
    /// about to take every lock timed from `now`, to be charged to it
    /// ([`Kernel::spent_global`]). Each such timeout took every lock
    /// unthrottled, as many as a program cared to arrange (the 2026-11-27
    /// audit: M10).
    pub(crate) fn share_back_in(&self, me: &Cpu, t: Paddr, now: u64) -> Option<u64> {
        let s = self.thread(t).sc.get();
        if s == 0 {
            return None;
        }
        let sc = self.sched(s);
        if sc.global_budget.get() > 0 {
            me.global_from.set(now);
            return None;
        }
        self.global_share(me, sc, now)
    }

    /// The share of time holding every lock left to context `sc`, brought
    /// up to `now`: `None` when some is left (the global entry about to
    /// begin is timed from `now`), else how long until it is back.
    fn global_share(&self, me: &Cpu, sc: &crate::sched::SchedCtx, now: u64) -> Option<u64> {
        let n = self.ncpu() as u128;
        // A context never charged starts with the burst; configuring it
        // again does not refill it.
        let earned = if sc.global_at.get() == 0 {
            GLOBAL_BURST_NS as u128
        } else {
            (now.saturating_sub(sc.global_at.get()) as u128 * GLOBAL_SHARE as u128 / (1000 * n)).min(GLOBAL_BURST_NS as u128)
        };
        let budget = (sc.global_budget.get() + earned as i64).min(GLOBAL_BURST_NS as i64);
        sc.global_at.set(now);
        sc.global_budget.set(budget);
        if budget > 0 {
            me.global_from.set(now);
            return None;
        }
        Some(((-budget) as u128 * 1000 * n / GLOBAL_SHARE as u128) as u64 + 1)
    }

    #[cold]
    #[inline(never)]
    fn throttle_as(&self, me: &Cpu, sc: &crate::sched::SchedCtx, t: Paddr, syscall: bool) -> bool {
        let now = self.hal.now();
        let Some(wait) = self.global_share(me, sc, now) else { return false };
        // Until the share is back: then the call is made again (a fault
        // needs nothing: the thread faults again where it was).
        if syscall {
            let r = &self.thread(t).regs;
            r[H::IP].set(r[H::IP].get().wrapping_sub(H::SYSCALL_LEN));
        }
        self.block(t, crate::thread::THROTTLED, 0, now.saturating_add(wait));
        true
    }

    /// The global entry under way, which [`Kernel::throttled`] let through
    /// for thread `t`, is over: what it cost comes out of the share of the
    /// context `t` runs on now. Its length, from before it took the first
    /// lock (waiting for the others it held some), is what it cost each
    /// other processor that had a thread to run; an idle one loses nothing
    /// to its lock being held. So it is charged whole when any other
    /// processor had work, and not at all on a machine where none did (a
    /// pager alone, mapping and unmapping on every fault). Not scaled by
    /// how many had work: a pair of programs calling each other across two
    /// processors has one running at a time, and every call of the pair
    /// waits out the whole of it. Read holding every lock: what each
    /// processor runs is a fact, not an estimate.
    #[inline(always)]
    pub(crate) fn spent_global(&self, me: &Cpu, t: Paddr) {
        let s = self.thread(t).sc.get();
        // (A call may have taken the caller's own context away.)
        if s == 0 || !self.others().any(|(_, c)| c.cur.get() != 0) {
            return;
        }
        let held = self.hal.now().saturating_sub(me.global_from.get());
        let cost = held.min(i64::MAX as u64) as i64;
        let sc = self.sched(s);
        sc.global_budget.set(sc.global_budget.get().saturating_sub(cost));
    }

    /// How many times a kernel entry has taken every lock (a statistic
    /// for tests and the debug console).
    pub fn global_entries(&self) -> u64 {
        self.globals.get()
    }

    /// From holding this processor's lock to holding processor `other`'s as
    /// well. This one's may have been let go on the way: the caller
    /// re-examines everything it looked at before.
    pub(crate) fn go_pair(&self, other: usize) {
        let cpu = self.me();
        crate::lock::go_pair(&cpu.lock, &self.cpus.get(other).lock, other < self.hal.cpu(), |l, began| self.lock_wait_turn(cpu, l, began));
        cpu.pair.set(other as u64 + 1);
        cpu.pairs.set(cpu.pairs.get() + 1);
        self.catch_up(cpu);
    }

    /// Let go of the second lock taken by [`Kernel::go_pair`].
    pub(crate) fn leave_pair(&self) {
        let other = self.me().pair.replace(0);
        if other != 0 {
            self.unlock(other as usize - 1);
        }
    }

    /// The locks this kernel entry holds, read once for a series of checks.
    pub(crate) fn held(&self) -> Held {
        Held { me: self.hal.cpu() as u64, pair: self.me().pair.get(), all: self.global.get() }
    }

    /// Whether this kernel entry holds processor `cpu`'s lock.
    pub(crate) fn holds(&self, cpu: u64) -> bool {
        self.held().has(cpu)
    }

    /// How many times a kernel entry has taken a second processor's lock
    /// (a statistic for tests and the debug console).
    /// Only for tests: call it while no processor is in the kernel.
    pub fn pair_entries(&self) -> u64 {
        self.cpus.iter(self.ncpu()).map(|c| c.pairs.get()).sum()
    }

    /// Back to holding only this processor's lock.
    pub(crate) fn go_local(&self) {
        // What this entry unmapped, told to the others before they can
        // run on it again.
        self.shoot_pending();
        let cpu = self.hal.cpu();
        self.global.set(false);
        crate::lock::go_local(cpu, self.cpus.iter(self.ncpu()).map(|c| &c.lock));
    }

    /// Every processor other than this one, with its index. Only for code
    /// holding all locks.
    pub(crate) fn others(&self) -> impl Iterator<Item = (usize, &Cpu)> {
        let me = self.hal.cpu();
        debug_assert!(self.global.get() || self.ncpu() == 1);
        self.cpus.iter(self.ncpu()).enumerate().filter(move |(i, _)| *i != me)
    }
}

/// Which processors joined the machine at boot. The firmware's table says
/// which processors exist; a processor counts once it has arrived. Each
/// secondary takes the next index as it arrives ([`Admission::admit`]), so
/// the indices are contiguous whichever processors never start, and the
/// boot processor closes the count ([`Admission::close`]) before the root
/// task is told how many there are: no thread is ever placed on a
/// processor that is not running. One that arrives after the count is
/// closed is turned away and stays stopped.
pub struct Admission(core::sync::atomic::AtomicU64);

const CLOSED: u64 = 1 << 63;

impl Admission {
    /// No secondary has arrived.
    pub const fn new() -> Admission {
        Admission(core::sync::atomic::AtomicU64::new(0))
    }

    /// A secondary processor arriving: its index (from 1), or `None` once
    /// the count is closed (it must stop, touching nothing shared).
    pub fn admit(&self) -> Option<usize> {
        use core::sync::atomic::Ordering::{AcqRel, Acquire};
        self.0.try_update(AcqRel, Acquire, |v| (v & CLOSED == 0).then_some(v + 1)).ok().map(|v| v as usize + 1)
    }

    /// How many secondaries have arrived so far.
    pub fn arrived(&self) -> usize {
        (self.0.load(core::sync::atomic::Ordering::Acquire) & !CLOSED) as usize
    }

    /// Admit no more: the number of processors, the boot one included.
    pub fn close(&self) -> usize {
        (self.0.fetch_or(CLOSED, core::sync::atomic::Ordering::AcqRel) & !CLOSED) as usize + 1
    }
}

impl Default for Admission {
    fn default() -> Admission {
        Admission::new()
    }
}
