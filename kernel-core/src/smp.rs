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
//!   ([`Hal::kick`] with `wait`).
//!
//! A processor waiting for its lock has interrupts off and has already saved
//! its thread's registers, so the holder may change that thread. When it
//! gets the lock it looks at what it was told in the meantime ([`Cpu::flush`]
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
    pub(crate) rq_head: Cell<Paddr>,
    pub(crate) rq_tail: Cell<Paddr>,
    /// Threads with deadlines, soonest first.
    pub(crate) timers: Cell<Paddr>,
    /// When the timer is set to fire (`u64::MAX`: not armed).
    pub(crate) timer_at: Cell<u64>,
    pub(crate) slice_end: Cell<u64>,
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
            rq_head: Cell::new(0),
            rq_tail: Cell::new(0),
            timers: Cell::new(0),
            timer_at: Cell::new(u64::MAX),
            slice_end: Cell::new(0),
            resched: Cell::new(false),
            tick: Cell::new(false),
            direct: Cell::new(0),
            dirty: Cell::new(true),
            flush: Cell::new(false),
            pair: Cell::new(0),
            pairs: Cell::new(0),
        }
    }
}

impl<H: Hal> Kernel<H> {
    /// This processor's state.
    pub(crate) fn me(&self) -> &Cpu {
        // Masked rather than bounds-checked: this is on every path through
        // the kernel, and a processor's index is below MAX_CPUS (a power of
        // two) by construction.
        let i = self.hal.cpu();
        debug_assert!(i < MAX_CPUS);
        &self.cpus[i & (MAX_CPUS - 1)]
    }

    /// Number of processors the machine has (set once at boot).
    pub fn cpu_count(&self) -> usize {
        self.ncpu.get()
    }

    /// Tell the kernel how many processors there are. Called once by the
    /// boot processor before any other processor enters the kernel.
    pub fn set_cpu_count(&self, n: usize) {
        self.ncpu.set(n.clamp(1, MAX_CPUS));
    }

    pub(crate) fn unlock(&self, cpu: usize) {
        self.cpus[cpu].lock.unlock();
    }

    /// Enter the kernel on this processor: take its lock and act on what
    /// other processors asked of it while it was away.
    pub fn enter(&self) {
        self.me().lock.lock();
        self.caught_up();
    }

    fn caught_up(&self) {
        let me = self.me();
        if me.flush.replace(false) {
            self.hal.tlb_reset();
            me.loaded_root.set(0);
            me.dirty.set(true);
        }
    }

    /// Leave the kernel on this processor.
    pub fn leave(&self) {
        self.unlock(self.hal.cpu());
    }

    /// From holding this processor's lock to holding every processor's.
    /// Other global operations may run in between: the caller re-examines
    /// everything it looked at before.
    pub(crate) fn go_global(&self) {
        crate::lock::go_global(&self.me().lock, self.cpus[..self.ncpu.get()].iter().map(|c| &c.lock));
        self.global.set(true);
        self.globals.set(self.globals.get() + 1);
        self.caught_up();
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
        let me = self.hal.cpu();
        crate::lock::go_pair(&self.cpus[me].lock, &self.cpus[other].lock, other < me);
        let cpu = self.me();
        cpu.pair.set(other as u64 + 1);
        cpu.pairs.set(cpu.pairs.get() + 1);
        self.caught_up();
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
        self.cpus[..self.ncpu.get()].iter().map(|c| c.pairs.get()).sum()
    }

    /// Back to holding only this processor's lock.
    pub(crate) fn go_local(&self) {
        let cpu = self.hal.cpu();
        self.global.set(false);
        crate::lock::go_local(cpu, self.cpus[..self.ncpu.get()].iter().map(|c| &c.lock));
    }

    /// Every processor other than this one, with its index. Only for code
    /// holding all locks.
    pub(crate) fn others(&self) -> impl Iterator<Item = (usize, &Cpu)> {
        let me = self.hal.cpu();
        debug_assert!(self.global.get() || self.ncpu.get() == 1);
        self.cpus[..self.ncpu.get()].iter().enumerate().filter(move |(i, _)| *i != me)
    }
}
