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

/// Most processors the kernel will run.
pub const MAX_CPUS: usize = 256;
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
    /// The first `n` processors' states, in order.
    pub(crate) fn iter(&self, n: usize) -> impl Iterator<Item = &Cpu> {
        (0..n).map(move |i| self.get(i))
    }
}

/// One processor's lock and scheduling state (protected by the lock). Each
/// has its own cache lines: processors working independently must not make
/// each other's memory bounce.
#[repr(align(128))]
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
    /// The fair run queue's groups with threads waiting, a tree in order
    /// of the group's virtual runtime (`order.rs`, `sched::Group`): its
    /// first (least-run) and last groups, and the tree.
    pub(crate) gq_first: Cell<Paddr>,
    pub(crate) gq_last: Cell<Paddr>,
    pub(crate) gq_tree: crate::order::Root,
    /// This processor's own group: its threads given no group.
    pub(crate) unbound: crate::sched::Group,
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
    /// The least virtual runtime among this processor's groups (only grows).
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
    /// A global entry's request that this processor stop running on a root
    /// it is about to reuse ([`Kernel::evict`]), answered while this
    /// processor waits for a lock without holding its own.
    /// Last: read only when a lock is contended, so it moves nothing the
    /// entry paths use.
    pub(crate) leave: crate::lock::Leave,
}

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
            gq_first: Cell::new(0),
            gq_last: Cell::new(0),
            gq_tree: crate::order::Root::new(),
            unbound: crate::sched::Group::new(),
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
            leave: crate::lock::Leave::new(),
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
        // Masked rather than bounds-checked: this is on every path through
        // the kernel, and a processor's index is below MAX_CPUS (a power of
        // two) by construction.
        let s = self.hal.cpu_slot();
        debug_assert!(s == cpu_slot(self.hal.cpu()));
        self.cpus.at_slot(s)
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
        me.lock.lock_while(|| self.answer_leave(me));
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

    /// From holding this processor's lock to holding every processor's.
    /// Other global operations may run in between: the caller re-examines
    /// everything it looked at before.
    pub(crate) fn go_global(&self) {
        let me = self.me();
        crate::lock::go_global(&me.lock, self.cpus.iter(self.ncpu()).map(|c| &c.lock), || self.answer_leave(me));
        self.global.set(true);
        self.globals.set(self.globals.get() + 1);
        self.catch_up(me);
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
        crate::lock::go_pair(&cpu.lock, &self.cpus.get(other).lock, other < self.hal.cpu(), || self.answer_leave(cpu));
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
