//! The scheduler (`src/sched.rs`) on the host machine: shares, reservations,
//! admission control, and what happens to bindings when things change.
//!
//! Time is simulated: each step runs the machine to its next timer and
//! charges the interval to whichever thread was running, which is exactly
//! when a tickless scheduler makes its decisions.
mod support;

use kernel_core::hal::Hal;
use silica_abi::root::{SLOT_CAPSPACE, SLOT_THREAD};
use silica_abi::{Error, ObjectType as T, SCHED_DEADLINE, SCHED_FAIR, Syscall as S};
use std::collections::HashMap;
use support::*;

const FREE: u64 = 2000;
const MS: u64 = 1_000_000;

/// A machine whose root task has suspended itself, leaving the threads it
/// made to share the processor.
fn machine() -> (Machine, u64) {
    let m = Machine::new(16);
    let (ut, _) = m.big_untyped();
    (m, ut)
}

/// A scheduling context in `slot`, configured as asked.
fn sched(m: &Machine, ut: u64, slot: u64, class: u64, a: u64, b: u64) {
    m.ok(S::Retype, [ut, T::SchedContext as u64, 0, SLOT_CAPSPACE, slot, 1]);
    m.ok(S::SchedConfigure, [slot, class, a, b, 0, 0]);
}

/// When the current processor's timer fires next (`u64::MAX`: not armed).
/// The test machine's timer stays set after it fires; this simulation ticks
/// at every timer it reaches, so one at or before now has fired already.
fn next_timer(m: &Machine) -> u64 {
    let t = m.k.hal.timer();
    if m.k.hal.armed[0].get() && t > m.k.hal.now.get() { t } else { u64::MAX }
}

/// Run until `until`, timer to timer; returns how long each thread ran.
fn run(m: &Machine, until: u64) -> HashMap<u64, u64> {
    let mut ran = HashMap::new();
    let mut steps = 0;
    while m.k.hal.now.get() < until {
        let now = m.k.hal.now.get();
        let next = next_timer(m).min(until);
        assert!(next > now || next == until, "the timer is armed in the past: {next} at {now}");
        *ran.entry(m.k.current()).or_insert(0) += next - now;
        m.k.hal.now.set(next);
        m.tick();
        steps += 1;
        assert!(steps < 100_000, "the scheduler never stops interrupting itself");
    }
    ran
}

fn suspend_root(m: &Machine) {
    let r = m.sys(S::ThreadControl, [SLOT_THREAD, 0, 0, 0, 0, 0]);
    assert!(r.blocked, "the root task suspending itself must stop it");
}

#[test]
fn weights_share_the_processor_in_proportion() {
    let (m, ut) = machine();
    let a = m.spawn(ut, FREE);
    let b = m.spawn(ut, FREE + 1);
    sched(&m, ut, FREE + 2, SCHED_FAIR, 3072, 0);
    m.ok(S::ThreadControl, [FREE + 1, 3, FREE + 2, 0, 0, 0]);
    suspend_root(&m);
    let ran = run(&m, 600 * MS);
    let (ta, tb) = (ran[&a] as f64, ran[&b] as f64);
    let ratio = tb / ta;
    assert!((2.7..=3.3).contains(&ratio), "weights 1024 and 3072 got {ta} and {tb} ns (ratio {ratio:.2})");
    // Charged to the context, and visible to its holder (thread `a` shares
    // the root's CapSpace): consumed is about what `b` ran.
    let r = m.sys_as(a, S::Inspect, [FREE + 2, 0, 0, 0, 0, 0]);
    assert_eq!(r.status, 0);
    let regs = m.k.regs(a);
    let (class, weight, consumed) = (regs[TestHal::ARG[0]].get(), regs[TestHal::ARG[1]].get(), regs[TestHal::ARG[4]].get());
    assert_eq!((class, weight), (SCHED_FAIR, 3072));
    assert!(consumed.abs_diff(ran[&b]) < 4 * MS, "the context was charged {consumed} ns for {} ns", ran[&b]);
}

#[test]
fn equal_threads_take_turns_promptly() {
    let (m, ut) = machine();
    let a = m.spawn(ut, FREE);
    let b = m.spawn(ut, FREE + 1);
    suspend_root(&m);
    // No thread waits longer than a few granules for its turn.
    let mut last_run = HashMap::from([(a, 0u64), (b, 0u64)]);
    let mut worst = 0;
    let mut steps = 0;
    while m.k.hal.now.get() < 200 * MS {
        let cur = m.k.current();
        let now = m.k.hal.now.get();
        for (&t, &since) in &last_run {
            if t != cur {
                worst = worst.max(now - since);
            }
        }
        let next = next_timer(&m).min(200 * MS);
        last_run.insert(cur, next);
        m.k.hal.now.set(next);
        m.tick();
        steps += 1;
        assert!(steps < 100_000);
    }
    assert!(worst <= 7 * MS, "a runnable thread waited {worst} ns for its turn");
}

#[test]
fn a_deadline_reservation_gets_its_budget_and_no_more() {
    let (m, ut) = machine();
    let hogs = [m.spawn(ut, FREE), m.spawn(ut, FREE + 1), m.spawn(ut, FREE + 2)];
    let dl = m.spawn(ut, FREE + 3);
    sched(&m, ut, FREE + 4, SCHED_DEADLINE, 2 * MS, 10 * MS);
    m.ok(S::ThreadControl, [FREE + 3, 3, FREE + 4, 0, 0, 0]);
    suspend_root(&m);
    let ran = run(&m, 1000 * MS);
    let share = ran[&dl] as f64 / (1000 * MS) as f64;
    // Three fair hogs would leave it a quarter; the reservation is a fifth,
    // and a hard one: no more, though it would run all the time.
    assert!((0.19..=0.21).contains(&share), "a 2 ms in 10 ms reservation ran {:.1}% of the time", share * 100.0);
    for h in hogs {
        let s = ran[&h] as f64 / (1000 * MS) as f64;
        assert!((0.24..=0.29).contains(&s), "a fair hog ran {:.1}% beside the reservation", s * 100.0);
    }
}

#[test]
fn admission_control_refuses_to_promise_more_than_the_processor_has() {
    let (m, ut) = machine();
    sched(&m, ut, FREE, SCHED_DEADLINE, 5 * MS, 10 * MS);
    m.ok(S::Retype, [ut, T::SchedContext as u64, 0, SLOT_CAPSPACE, FREE + 1, 1]);
    assert_eq!(m.err(S::SchedConfigure, [FREE + 1, SCHED_DEADLINE, 5 * MS, 10 * MS, 0, 0]), Error::Overload, "100% promised");
    m.ok(S::SchedConfigure, [FREE + 1, SCHED_DEADLINE, 4 * MS, 10 * MS, 0, 0]);
    // Changing a reservation counts what it held already.
    m.ok(S::SchedConfigure, [FREE + 1, SCHED_DEADLINE, 3 * MS, 10 * MS, 0, 0]);
    // Making it fair gives its share back.
    m.ok(S::SchedConfigure, [FREE, SCHED_FAIR, 1024, 0, 0, 0]);
    m.ok(S::SchedConfigure, [FREE + 1, SCHED_DEADLINE, 8 * MS, 10 * MS, 0, 0]);
    // Destroying one gives its share back too.
    m.ok(S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]);
    sched(&m, ut, FREE + 2, SCHED_DEADLINE, 9 * MS, 10 * MS);
    // Malformed configurations.
    m.ok(S::Retype, [ut, T::SchedContext as u64, 0, SLOT_CAPSPACE, FREE + 3, 1]);
    for a in [
        [FREE + 3, SCHED_FAIR, 0, 0, 0, 0],
        [FREE + 3, SCHED_FAIR, (1 << 20) + 1, 0, 0, 0],
        [FREE + 3, SCHED_DEADLINE, 0, 10 * MS, 0, 0],
        [FREE + 3, SCHED_DEADLINE, 11 * MS, 10 * MS, 0, 0],
        [FREE + 3, SCHED_DEADLINE, 1000, 50_000, 0, 0],
        [FREE + 3, SCHED_DEADLINE, MS, 10 * MS, 1, 0],
        [FREE + 3, 7, 1, 1, 0, 0],
    ] {
        assert_eq!(m.err(S::SchedConfigure, a), Error::BadArg, "{a:?}");
    }
    assert_eq!(m.err(S::SchedConfigure, [SLOT_THREAD, SCHED_FAIR, 1, 0, 0, 0]), Error::WrongType);
}

#[test]
fn bindings_follow_what_happens_to_contexts_and_threads() {
    let (m, ut) = machine();
    let t = m.spawn(ut, FREE);
    sched(&m, ut, FREE + 1, SCHED_DEADLINE, MS, 10 * MS);
    m.ok(S::ThreadControl, [FREE, 3, FREE + 1, 0, 0, 0]);
    // A bound deadline context keeps its class and processor.
    assert_eq!(m.err(S::SchedConfigure, [FREE + 1, SCHED_FAIR, 1024, 0, 0, 0]), Error::InUse);
    // A thread with a reservation on this processor cannot simply move.
    assert_eq!(m.err(S::ThreadControl, [FREE, 2, 0, 0, 0, 0]), Error::InUse);
    // Binding needs a scheduling context.
    assert_eq!(m.err(S::ThreadControl, [FREE, 3, SLOT_THREAD, 0, 0, 0]), Error::WrongType);
    // Destroying the context returns the thread to the default fair share,
    // still runnable, and gives the reservation back.
    m.ok(S::CapDelete, [FREE + 1, 0, 0, 0, 0, 0]);
    sched(&m, ut, FREE + 2, SCHED_DEADLINE, 9 * MS, 10 * MS);
    suspend_root(&m);
    let ran = run(&m, 50 * MS);
    assert!(ran.get(&t).copied().unwrap_or(0) >= 49 * MS, "the unbound thread did not run on its own");
    m.k.check_runqueues().unwrap();
}

#[test]
fn a_thread_waking_beside_busy_ones_runs_at_once() {
    let (m, ut) = machine();
    let a = m.spawn(ut, FREE);
    let b = m.spawn(ut, FREE + 1);
    let s = m.spawn(ut, FREE + 2);
    m.retype(ut, T::Notification, 0, FREE + 3, 1);
    suspend_root(&m);
    let mut worst = 0;
    for round in 0..50 {
        // The sleeper waits 1 ms for a signal that never comes.
        let at = m.k.hal.now.get() + MS;
        let r = m.sys_as(s, S::Recv, [FREE + 3, at, 0, 0, 0, 0]);
        assert!(r.blocked, "round {round}: the sleeper did not wait");
        // The two others run until it wakes.
        while m.k.current() != s {
            let now = m.k.hal.now.get();
            let next = next_timer(&m);
            assert!(next != u64::MAX, "round {round}: nothing will wake the sleeper");
            m.k.hal.now.set(next.max(now));
            m.tick();
            assert!(m.k.hal.now.get() < at + 50 * MS, "round {round}: the sleeper never ran");
        }
        worst = worst.max(m.k.hal.now.get() - at);
        let _ = (a, b);
    }
    assert!(worst <= MS / 10, "a waking thread ran {worst} ns after its deadline beside two busy ones");
}

/// The idle watch grows to cover halts that work ended soon, and shrinks
/// back after halts that lasted.
#[test]
fn the_idle_watch_learns_how_long_to_watch() {
    use kernel_core::sched::{IDLE_WATCH_MAX_NS, IDLE_WATCH_NS, Watch};
    let w = Watch::new();
    assert_eq!(w.length(), IDLE_WATCH_NS);
    // A disk read every 20 us: one halt, and the watch covers the next.
    w.woken_after(20_000);
    assert_eq!(w.length(), 25_000);
    // Never beyond the most a watch may cost.
    for _ in 0..20 {
        w.woken_after(IDLE_WATCH_MAX_NS);
    }
    assert_eq!(w.length(), IDLE_WATCH_MAX_NS);
    // Long halts: back to the short watch within a handful.
    for _ in 0..8 {
        w.woken_after(10_000_000);
    }
    assert_eq!(w.length(), IDLE_WATCH_NS);
    // Short wake-ups double it from there.
    w.woken_after(1_000);
    assert_eq!(w.length(), 2 * IDLE_WATCH_NS);
}
