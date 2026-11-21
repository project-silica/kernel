//! The lock protocol of `src/smp.rs` under every interleaving (loom).
//!
//! `src/lock.rs` is compiled here against loom's atomics, so these models
//! run the kernel's own lock, global-acquisition and presence code. What a
//! model adds is the surrounding order of operations, written as the
//! architecture's entry and exit code performs it; `loom::cell::UnsafeCell`
//! stands for the memory the protocol protects and fails the model on any
//! concurrent access.
//!
//! Run by `tools/test.sh --full` (suite kernel.loom), or:
//!     RUSTFLAGS="--cfg loom" cargo test --release -p kernel-core --test loom_models
#![cfg(loom)]

/// Defines a constructor for `lock.rs`: a plain function here.
macro_rules! ctor {
    ($(#[$m:meta])* $v:vis fn $name:ident() -> $t:ty $body:block) => {
        $(#[$m])* $v fn $name() -> $t $body
    };
}

mod sync {
    pub(crate) use loom::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
    pub(crate) use loom::thread::yield_now as spin;
}
#[allow(dead_code)]
#[path = "../src/lock.rs"]
mod lock;

use lock::{Lock, Presence, go_global, go_local, go_pair};
use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, Ordering};
use loom::thread;

struct Machine {
    locks: [Lock; 2],
    /// Each processor's own scheduling state.
    cpu: [UnsafeCell<u32>; 2],
    /// State only a holder of every lock may touch (capability slots).
    shared: UnsafeCell<u32>,
}

fn machine() -> Arc<Machine> {
    Arc::new(Machine { locks: [Lock::new(), Lock::new()], cpu: [UnsafeCell::new(0), UnsafeCell::new(0)], shared: UnsafeCell::new(0) })
}

impl Machine {
    /// A kernel entry that stays local: only this processor's state.
    fn local(&self, me: usize) {
        self.locks[me].lock();
        self.cpu[me].with_mut(|p| unsafe { *p += 1 });
        self.locks[me].unlock();
    }
    /// A kernel entry that goes global: everybody's state.
    fn global(&self, me: usize) {
        self.locks[me].lock();
        go_global(&self.locks[me], self.locks.iter());
        self.shared.with_mut(|p| unsafe { *p += 1 });
        for c in &self.cpu {
            c.with_mut(|p| unsafe { *p += 10 });
        }
        go_local(me, self.locks.iter());
        self.cpu[me].with_mut(|p| unsafe { *p += 1 });
        self.locks[me].unlock();
    }
}

#[test]
fn a_local_entry_and_a_global_entry_never_touch_the_same_state_at_once() {
    loom::model(|| {
        let m = machine();
        let a = {
            let m = m.clone();
            thread::spawn(move || {
                m.local(0);
                m.local(0);
            })
        };
        m.global(1);
        a.join().unwrap();
        assert_eq!(m.cpu[0].with(|p| unsafe { *p }), 12);
        assert_eq!(m.cpu[1].with(|p| unsafe { *p }), 11);
    });
}

#[test]
fn two_global_entries_exclude_each_other_and_do_not_deadlock() {
    loom::model(|| {
        let m = machine();
        let a = {
            let m = m.clone();
            thread::spawn(move || m.global(0))
        };
        m.global(1);
        a.join().unwrap();
        assert_eq!(m.shared.with(|p| unsafe { *p }), 2);
        assert_eq!(m.cpu[0].with(|p| unsafe { *p }), 21);
    });
}

/// Three processors. A kernel entry of processor `me` that takes processor
/// `other`'s lock as well (a message between the two) changes the state of
/// both and of nobody else.
struct Three {
    locks: [Lock; 3],
    cpu: [UnsafeCell<u32>; 3],
}

impl Three {
    fn new() -> Arc<Three> {
        Arc::new(Three {
            locks: [Lock::new(), Lock::new(), Lock::new()],
            cpu: [UnsafeCell::new(0), UnsafeCell::new(0), UnsafeCell::new(0)],
        })
    }
    fn pair(&self, me: usize, other: usize) {
        self.locks[me].lock();
        go_pair(&self.locks[me], &self.locks[other], other < me);
        for c in [me, other] {
            self.cpu[c].with_mut(|p| unsafe { *p += 1 });
        }
        self.locks[other].unlock();
        self.locks[me].unlock();
    }
    fn global(&self, me: usize) {
        self.locks[me].lock();
        go_global(&self.locks[me], self.locks.iter());
        for c in &self.cpu {
            c.with_mut(|p| unsafe { *p += 10 });
        }
        go_local(me, self.locks.iter());
        self.locks[me].unlock();
    }
}

#[test]
fn two_processors_taking_each_others_locks_do_not_deadlock_or_overlap() {
    loom::model(|| {
        let m = Three::new();
        let a = {
            let m = m.clone();
            thread::spawn(move || m.pair(0, 1))
        };
        m.pair(1, 0);
        a.join().unwrap();
        assert_eq!((m.cpu[0].with(|p| unsafe { *p }), m.cpu[1].with(|p| unsafe { *p })), (2, 2));
    });
}

#[test]
fn a_pair_entry_and_a_global_entry_exclude_each_other() {
    loom::model(|| {
        let m = Three::new();
        let a = {
            let m = m.clone();
            thread::spawn(move || m.pair(2, 1))
        };
        m.global(0);
        a.join().unwrap();
        assert_eq!((m.cpu[0].with(|p| unsafe { *p }), m.cpu[1].with(|p| unsafe { *p }), m.cpu[2].with(|p| unsafe { *p })), (10, 11, 11));
    });
}

/// A queue is protected by the lock of the processor that owns it, and its
/// owner changes only while the old owner's lock is held. Processor 0,
/// holding both locks, takes the queue over from processor 1; processor 1,
/// holding its own lock, uses the queue only if it still owns it. Neither
/// ever uses it while the other does.
#[test]
fn a_queue_changing_owner_is_never_used_under_the_old_owners_lock_alone() {
    loom::model(|| {
        struct S {
            locks: [Lock; 2],
            /// The owning processor plus one, as in a queue's `owner` word.
            owner: loom::sync::atomic::AtomicU64,
            queue: UnsafeCell<u32>,
        }
        let s = Arc::new(S { locks: [Lock::new(), Lock::new()], owner: loom::sync::atomic::AtomicU64::new(2), queue: UnsafeCell::new(0) });
        let old_owner = {
            let s = s.clone();
            thread::spawn(move || {
                for _ in 0..2 {
                    s.locks[1].lock();
                    if s.owner.load(Ordering::Relaxed) == 2 {
                        s.queue.with_mut(|p| unsafe { *p += 1 });
                    }
                    s.locks[1].unlock();
                }
            })
        };
        // Processor 0: a receive on a queue processor 1 owns.
        s.locks[0].lock();
        if s.owner.load(Ordering::Relaxed) == 2 {
            go_pair(&s.locks[0], &s.locks[1], false);
            // Looked at again with both locks held.
            if s.owner.load(Ordering::Relaxed) == 2 {
                s.queue.with_mut(|p| unsafe { *p += 10 });
                s.owner.store(1, Ordering::Relaxed);
            }
            s.locks[1].unlock();
        }
        // From here on the queue is processor 0's alone.
        if s.owner.load(Ordering::Relaxed) == 1 {
            s.queue.with_mut(|p| unsafe { *p += 10 });
        }
        s.locks[0].unlock();
        old_owner.join().unwrap();
        assert!(s.queue.with(|p| unsafe { *p }) >= 20);
    });
}

/// Processor 0 runs a thread; processor 1, holding processor 0's lock,
/// stops that thread and changes its saved registers (ThreadStart, or
/// destroying it and reusing its memory). The registers are written by
/// processor 0's entry code without any lock, so processor 1 may touch them
/// only once processor 0 is known to be past that point: it interrupts it
/// and waits until its presence is no longer "user".
#[test]
fn another_processors_thread_is_changed_only_after_that_processor_left_user_mode() {
    loom::model(|| {
        struct S {
            lock: Lock,
            presence: Presence,
            regs: UnsafeCell<u32>,
            stopped: AtomicBool,
        }
        let s = Arc::new(S { lock: Lock::new(), presence: Presence::new(), regs: UnsafeCell::new(0), stopped: AtomicBool::new(false) });

        // Processor 0: the first exit to user mode, then kernel entries (a
        // system call, then the interrupt that the kick guarantees).
        let owner = {
            let s = s.clone();
            thread::spawn(move || {
                // The scheduler, on every exit, runs a thread only if it has
                // not been stopped.
                s.lock.lock();
                let mut in_user = !s.stopped.load(Ordering::SeqCst);
                if in_user {
                    s.regs.with(|r| unsafe { *r });
                    s.presence.leave_to_user();
                }
                s.lock.unlock();
                for _ in 0..2 {
                    if !in_user {
                        break;
                    }
                    // Entry stub: save the registers, then announce, then lock.
                    s.regs.with_mut(|r| unsafe { *r += 1 });
                    s.presence.enter_kernel();
                    s.lock.lock();
                    if s.stopped.load(Ordering::SeqCst) {
                        // The thread was stopped while this processor waited:
                        // its syscall is dropped and it does not return to it.
                        in_user = false;
                    } else {
                        // Exit: restore the registers, announce, unlock.
                        s.regs.with(|r| unsafe { *r });
                        s.presence.leave_to_user();
                    }
                    s.lock.unlock();
                }
                // However it got here, an interrupt eventually brings a
                // processor in user mode into the kernel.
                if in_user {
                    s.regs.with_mut(|r| unsafe { *r += 1 });
                    s.presence.enter_kernel();
                }
            })
        };

        // Processor 1, holding processor 0's lock as part of holding all.
        s.lock.lock();
        while s.presence.is_user() {
            thread::yield_now(); // the kick has been sent; wait for it to land
        }
        s.stopped.store(true, Ordering::SeqCst);
        s.regs.with_mut(|r| unsafe { *r = 1000 });
        s.lock.unlock();

        owner.join().unwrap();
        // Nothing processor 0 did after being stopped overwrote the new registers.
        assert_eq!(s.regs.with(|r| unsafe { *r }), 1000);
    });
}

/// The system-call entry first tries its own lock, before saving anything:
/// when that works, nobody can look at the registers while they are saved,
/// so they may be saved before "kernel" is announced; when another processor
/// holds the lock, the entry falls back to saving, announcing, then waiting.
/// Either way processor 1, holding the lock and waiting for processor 0 to
/// leave user mode, never meets a half-saved frame.
#[test]
fn a_system_call_that_takes_its_lock_first_is_never_seen_half_saved() {
    loom::model(|| {
        struct S {
            lock: Lock,
            presence: Presence,
            regs: UnsafeCell<u32>,
            stopped: AtomicBool,
        }
        let s = Arc::new(S { lock: Lock::new(), presence: Presence::new(), regs: UnsafeCell::new(0), stopped: AtomicBool::new(false) });

        let owner = {
            let s = s.clone();
            thread::spawn(move || {
                s.lock.lock();
                let mut in_user = !s.stopped.load(Ordering::SeqCst);
                if in_user {
                    s.regs.with(|r| unsafe { *r });
                    s.presence.leave_to_user();
                }
                s.lock.unlock();
                for _ in 0..2 {
                    if !in_user {
                        break;
                    }
                    if s.lock.try_lock() {
                        // Entry stub, lock first: save, then announce.
                        s.regs.with_mut(|r| unsafe { *r += 1 });
                        s.presence.enter_kernel();
                    } else {
                        s.regs.with_mut(|r| unsafe { *r += 1 });
                        s.presence.enter_kernel();
                        s.lock.lock();
                    }
                    if s.stopped.load(Ordering::SeqCst) {
                        in_user = false;
                    } else {
                        s.regs.with(|r| unsafe { *r });
                        s.presence.leave_to_user();
                    }
                    s.lock.unlock();
                }
                if in_user {
                    s.regs.with_mut(|r| unsafe { *r += 1 });
                    s.presence.enter_kernel();
                }
            })
        };

        s.lock.lock();
        while s.presence.is_user() {
            thread::yield_now();
        }
        s.stopped.store(true, Ordering::SeqCst);
        s.regs.with_mut(|r| unsafe { *r = 1000 });
        s.lock.unlock();

        owner.join().unwrap();
        assert_eq!(s.regs.with(|r| unsafe { *r }), 1000);
    });
}

/// A processor about to idle announces it before letting go of its lock, so
/// a processor that queues work for it and finds it "idle" interrupts it,
/// and one that finds it "in the kernel" knows it will see the work.
#[test]
fn work_queued_for_an_idling_processor_is_never_missed() {
    loom::model(|| {
        struct S {
            lock: Lock,
            presence: Presence,
            queue: UnsafeCell<u32>,
            /// The interrupt line: a notification stays pending until taken,
            /// as an interrupt does until the processor enables interrupts.
            interrupt: loom::sync::Notify,
        }
        let s =
            Arc::new(S { lock: Lock::new(), presence: Presence::new(), queue: UnsafeCell::new(0), interrupt: loom::sync::Notify::new() });
        let idler = {
            let s = s.clone();
            thread::spawn(move || {
                // In the scheduler, holding the lock: nothing to run.
                s.lock.lock();
                let mut ran = 0;
                loop {
                    ran += s.queue.with_mut(|q| unsafe { core::mem::take(&mut *q) });
                    if ran > 0 {
                        break;
                    }
                    s.presence.idle();
                    s.lock.unlock();
                    // Halt until an interrupt (one already pending ends it at once).
                    s.interrupt.wait();
                    s.presence.enter_kernel();
                    s.lock.lock();
                }
                s.lock.unlock();
                ran
            })
        };
        // Another processor makes a thread of the idler runnable.
        s.lock.lock();
        s.queue.with_mut(|q| unsafe { *q += 1 });
        if !s.presence.in_kernel() {
            s.interrupt.notify();
        }
        s.lock.unlock();
        assert_eq!(idler.join().unwrap(), 1);
    });
}

/// The same with a processor that watches before it halts: whoever queues
/// work for it pokes its presence word, and interrupts it only when the
/// poke comes too late. The watch opens a window for interrupts on every
/// turn, and an interrupt taken there is gone: it does not end a later
/// halt. So the processor must never be seen as "idle, interrupt me"
/// before it has stopped watching; it says "watching" while it still holds
/// its lock. The watcher halts only if it stopped watching before any poke,
/// and then the interrupt is still to come: the work is never missed.
#[test]
fn work_queued_for_a_watching_processor_is_never_missed() {
    loom::model(|| {
        struct S {
            lock: Lock,
            presence: Presence,
            queue: UnsafeCell<u32>,
            /// An interrupt has been sent and not yet taken.
            pending: AtomicBool,
            interrupt: loom::sync::Notify,
        }
        let s = Arc::new(S {
            lock: Lock::new(),
            presence: Presence::new(),
            queue: UnsafeCell::new(0),
            pending: AtomicBool::new(false),
            interrupt: loom::sync::Notify::new(),
        });
        let idler = {
            let s = s.clone();
            thread::spawn(move || {
                s.lock.lock();
                let mut ran = 0;
                loop {
                    ran += s.queue.with_mut(|q| unsafe { core::mem::take(&mut *q) });
                    if ran > 0 {
                        break;
                    }
                    s.presence.watch();
                    s.lock.unlock();
                    // One turn of the watch (a real one takes many; the
                    // model needs only the ways it can end): the window, in
                    // which a pending interrupt is taken and forgotten,
                    // then a look at the word.
                    s.pending.swap(false, Ordering::AcqRel);
                    if s.presence.watching() {
                        thread::yield_now();
                    }
                    if s.presence.stop_watching() {
                        // Halt until an interrupt (any interrupt: a halt
                        // may end for one that has nothing to do with
                        // this work, and the processor then watches again).
                        while !s.pending.swap(false, Ordering::AcqRel) {
                            s.interrupt.wait();
                        }
                    }
                    s.presence.enter_kernel();
                    s.lock.lock();
                }
                s.lock.unlock();
                ran
            })
        };
        s.lock.lock();
        s.queue.with_mut(|q| unsafe { *q += 1 });
        if !s.presence.in_kernel() && !s.presence.poke() {
            // (A swap, not a store: the model checker orders a plain store
            // against the idler's swaps more loosely than any machine does.)
            s.pending.swap(true, Ordering::AcqRel);
            s.interrupt.notify();
        }
        s.lock.unlock();
        // A wake that was lost would leave the idler halted for ever: the
        // model would report a deadlock.
        assert_eq!(idler.join().unwrap(), 1);
    });
}
