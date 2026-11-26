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

/// A spin lock: one byte, 0 free and 1 held, so an architecture's entry
/// code may take it with an atomic exchange of its own.
#[repr(transparent)]
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
        self.lock_while(|_| {});
    }
    /// Take the lock, calling `waiting` on every turn of the wait: what a
    /// processor that does not hold its own lock must go on answering
    /// ([`Leave`]). `waiting(true)` begins a wait: the lock was just found
    /// held (again, when another processor took it first as it was let
    /// go); `waiting(false)` is a turn of the same wait. A bound on the
    /// wait is a bound on how long the lock is not seen free.
    pub fn lock_while(&self, mut waiting: impl FnMut(bool)) {
        while self.0.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            let mut began = true;
            while self.0.load(Ordering::Relaxed) {
                waiting(core::mem::take(&mut began));
                spin();
            }
        }
    }
    /// Take the lock if it is free, without waiting. This is what the x86_64
    /// system-call entry does with one XCHG before it saves the registers
    /// (`kernel/arch/x86_64/cpu.rs`); the loom models run it in its place.
    pub fn try_lock(&self) -> bool {
        !self.0.swap(true, Ordering::Acquire)
    }
    /// Release the lock.
    pub fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

/// From holding `mine` to holding every lock in `all` (which includes
/// `mine`), taken in order so two processors doing this cannot deadlock.
/// `waiting` is called on every turn of a wait with the lock waited for, as
/// in [`Lock::lock_while`]: `mine` is let go first.
pub fn go_global<'a>(mine: &Lock, all: impl Iterator<Item = &'a Lock>, mut waiting: impl FnMut(&Lock, bool)) {
    mine.unlock();
    for l in all {
        l.lock_while(|began| waiting(l, began));
    }
}

/// From holding `mine` to holding `mine` and `other`, taken in index order
/// (the order [`go_global`] uses), so no two of these transitions deadlock.
/// When `other` comes first `mine` is let go on the way: the caller then
/// re-examines everything it looked at before, and `waiting` is called on
/// every turn of the waits that follow ([`Lock::lock_while`]), given the lock
/// waited for.
pub fn go_pair(mine: &Lock, other: &Lock, other_first: bool, mut waiting: impl FnMut(&Lock, bool)) {
    if other_first {
        mine.unlock();
        other.lock_while(|began| waiting(other, began));
        mine.lock_while(|began| waiting(mine, began));
    } else {
        other.lock_while(|began| waiting(other, began));
    }
}

/// A request to one processor to stop running on an address-space root:
/// the root, or 0 for none. A global entry about to reuse a root's memory
/// asks each processor that has the root loaded ([`Leave::ask`]) and waits
/// until each has answered. The asked processor does not hold its lock
/// (the global entry does), so it is in user mode, idle or waiting for that
/// lock; once kicked it is waiting for it, and it answers there, on every
/// turn of the wait ([`Lock::lock_while`]): it loads the kernel's own root,
/// then [`Leave::answer`]s. The asker needs nothing from it but that, so
/// neither waits for the other's lock and they cannot deadlock.
pub struct Leave(AtomicU64);

impl Leave {
    ctor! {
        /// No request.
        pub fn new() -> Leave {
            Leave(AtomicU64::new(0))
        }
    }
    /// Ask the processor to leave `root` (not 0). The caller holds its lock.
    pub fn ask(&self, root: u64) {
        self.0.store(root, Ordering::Release);
    }
    /// The root the processor is asked to leave, or 0. Read by the asked
    /// processor while it waits for its lock.
    pub fn asked(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
    /// The asked processor no longer runs on the root: whatever the asker
    /// does after seeing [`Leave::answered`] comes after its switch.
    pub fn answer(&self) {
        self.0.store(0, Ordering::Release);
    }
    /// Whether the request was answered (or none was made).
    pub fn answered(&self) -> bool {
        self.0.load(Ordering::Acquire) == 0
    }
}

impl Default for Leave {
    fn default() -> Leave {
        Leave::new()
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
/// Idle, and watching this word for a while before halting: whoever has
/// work for the processor changes it to `POKED` instead of interrupting.
const WATCH: u8 = 3;
/// Idle no longer: a processor with work for this one said so here.
const POKED: u8 = 4;

impl Presence {
    ctor! {
        /// A processor that is in the kernel (as every processor is when it starts).
        pub fn new() -> Presence {
            Presence(AtomicU8::new(KERNEL))
        }
    }
    // Orderings. A reader holds the owner's lock. What it needs from a
    // value it reads is what the owner did before storing it: having saved
    // its thread's registers before "kernel" (release here, acquire in the
    // reader). Everything else is ordered by the lock itself: "user" and
    // "idle" are stored while the owner holds its lock, so a reader that
    // takes the lock afterwards sees them; a reader that holds the lock
    // while the owner stores "kernel" may see the old "user" and then
    // interrupts a processor that has already left, which costs an
    // interrupt and nothing else. So no store needs to be sequentially
    // consistent, and on x86 none is a locked instruction: two of the
    // three atomic read-modify-writes a system call made are gone.

    /// The processor came into the kernel (registers saved, lock not yet taken),
    /// or woke from idling.
    pub fn enter_kernel(&self) {
        self.0.store(KERNEL, Ordering::Release);
    }
    /// The processor is about to run user code (it still holds its lock).
    pub fn leave_to_user(&self) {
        self.0.store(USER, Ordering::Release);
    }
    /// The processor is about to halt until an interrupt (it still holds its
    /// lock, and lets go of it next).
    pub fn idle(&self) {
        self.0.store(IDLE, Ordering::Release);
    }
    /// The processor is about to idle (it still holds its lock, and lets go
    /// of it next), and will look at this word for a while before it halts.
    /// Stored under the lock, like "idle", and for the same reason: whoever
    /// then takes the lock and has work for this processor sees it. It must
    /// not be stored over "idle" later: a processor that saw "idle" has
    /// sent an interrupt, and a watch takes interrupts without ending.
    pub fn watch(&self) {
        self.0.store(WATCH, Ordering::Release);
    }
    /// Whether the watching processor may go on watching: nobody has poked
    /// it, and no interrupt it took in between has left it in another state.
    pub fn watching(&self) -> bool {
        self.0.load(Ordering::Acquire) == WATCH
    }
    /// The watching processor gives up and is about to halt. False when it
    /// must not: it was poked first (or an interrupt it took changed what it
    /// was doing), and has work to look for.
    pub fn stop_watching(&self) -> bool {
        self.0.compare_exchange(WATCH, IDLE, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }
    /// Tell a watching processor that there is work for it. True when it
    /// will find it without an interrupt: it was watching (it now stops and
    /// looks) or has been poked already. One of this and
    /// [`Presence::stop_watching`] succeeds, never both: either the
    /// processor does not halt, or the caller knows to interrupt it.
    pub fn poke(&self) -> bool {
        match self.0.compare_exchange(WATCH, POKED, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => true,
            Err(now) => now == POKED,
        }
    }
    /// Whether the processor is running user code.
    pub fn is_user(&self) -> bool {
        self.0.load(Ordering::Acquire) == USER
    }
    /// Whether the processor is in the kernel: it will look at what changed
    /// before it leaves, so it need not be interrupted.
    pub fn in_kernel(&self) -> bool {
        self.0.load(Ordering::Acquire) == KERNEL
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
