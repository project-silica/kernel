//! The primitives of the lock protocol described in `smp.rs`: the spin lock,
//! the shared word, a processor's presence flag, and the two transitions
//! between holding one lock and holding all.
//!
//! This file is compiled into the kernel with the core atomics and into
//! `tests/loom_models.rs` with loom's, which runs the protocol under every
//! interleaving. It must therefore name atomics only through `crate::sync`,
//! and define constructors with `ctor!` (a `const fn` in the kernel; loom's
//! atomics have no constant constructors).

use crate::sync::{AtomicBool, AtomicU8, AtomicU64, Ordering, spin};

/// A spin lock.
pub struct Lock(AtomicBool);

impl Lock {
    ctor! {
        /// An unlocked lock.
        pub fn new() -> Lock {
            Lock(AtomicBool::new(false))
        }
    }
    /// Take the lock, waiting for it if need be.
    pub fn lock(&self) {
        while self.0.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            while self.0.load(Ordering::Relaxed) {
                spin();
            }
        }
    }
    /// Release the lock.
    pub fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

/// From holding `mine` to holding every lock in `all` (which includes
/// `mine`), taken in order so two processors doing this cannot deadlock.
pub fn go_global<'a>(mine: &Lock, all: impl Iterator<Item = &'a Lock>) {
    mine.unlock();
    for l in all {
        l.lock();
    }
}

/// From holding `mine` to holding `mine` and `other`, taken in index order
/// (the order [`go_global`] uses), so no two of these transitions deadlock.
/// When `other` comes first `mine` is let go on the way: the caller then
/// re-examines everything it looked at before.
pub fn go_pair(mine: &Lock, other: &Lock, other_first: bool) {
    if other_first {
        mine.unlock();
        other.lock();
        mine.lock();
    } else {
        other.lock();
    }
}

/// From holding every lock in `all` back to holding only number `mine`.
pub fn go_local<'a>(mine: usize, all: impl Iterator<Item = &'a Lock>) {
    for (i, l) in all.enumerate() {
        if i != mine {
            l.unlock();
        }
    }
}

/// A word that one processor may write while others read it: a capability
/// slot's fields. The lock rules decide who may write; a reader that is
/// only checking "is this what I expect" may look without a lock and get the
/// old or the new value, never a torn one and never undefined behaviour.
#[repr(transparent)]
pub struct Word(AtomicU64);

impl Word {
    /// The current value.
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
    /// Store a value.
    pub fn set(&self, v: u64) {
        self.0.store(v, Ordering::Relaxed);
    }
    /// Make the word `v` if it is 0. Returns whether it is `v` afterwards.
    pub fn claim(&self, v: u64) -> bool {
        match self.0.compare_exchange(0, v, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => true,
            Err(now) => now == v,
        }
    }
}

/// Where a processor is, as other processors need to know it.
///
/// The owner calls [`Presence::enter_kernel`] on every way into the kernel
/// *after* it has saved its thread's registers and *before* it takes its
/// lock, and [`Presence::leave_to_user`] on the way out *while it still
/// holds its lock*. So a processor holding this one's lock sees "user" only
/// while this one really is in user mode (or about to be, having finished
/// with the kernel), and once it sees anything else, this processor will not
/// touch its thread's registers or run user code again before it has taken
/// its lock.
pub struct Presence(AtomicU8);

const USER: u8 = 0;
const KERNEL: u8 = 1;
const IDLE: u8 = 2;

impl Presence {
    ctor! {
        /// A processor that is in the kernel (as every processor is when it starts).
        pub fn new() -> Presence {
            Presence(AtomicU8::new(KERNEL))
        }
    }
    /// The processor came into the kernel (registers saved, lock not yet taken),
    /// or woke from idling.
    pub fn enter_kernel(&self) {
        self.0.store(KERNEL, Ordering::SeqCst);
    }
    /// The processor is about to run user code (it still holds its lock).
    pub fn leave_to_user(&self) {
        self.0.store(USER, Ordering::SeqCst);
    }
    /// The processor is about to halt until an interrupt (it still holds its
    /// lock, and lets go of it next).
    pub fn idle(&self) {
        self.0.store(IDLE, Ordering::SeqCst);
    }
    /// Whether the processor is running user code.
    pub fn is_user(&self) -> bool {
        self.0.load(Ordering::SeqCst) == USER
    }
    /// Whether the processor is in the kernel: it will look at what changed
    /// before it leaves, so it need not be interrupted.
    pub fn in_kernel(&self) -> bool {
        self.0.load(Ordering::SeqCst) == KERNEL
    }
}

impl Default for Lock {
    fn default() -> Lock {
        Lock::new()
    }
}

impl Default for Presence {
    fn default() -> Presence {
        Presence::new()
    }
}
