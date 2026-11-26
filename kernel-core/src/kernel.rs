//! The kernel state for one processor and the syscall dispatcher.

use crate::cap::{Cap, FLAG_CONSOLE, FLAG_DEVICE, FLAG_DMA, FLAG_IO, FLAG_PORTS, FLAG_POWER};
use crate::hal::Hal;
use crate::mem::{Paddr, Phys};
use crate::smp::Held;
use crate::thread::{INACTIVE, IPC_CALL, IPC_GRANT, RUNNABLE, Thread};
use core::cell::Cell;
use silica_abi::{CALL_SLOT, CALL_TIMEOUT_SHIFT, Error, KEEP, MsgInfo, NONBLOCK, ObjectType, PAGE_BITS, Rights, Syscall};

/// How a syscall handler finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// Complete; the status register is set to success.
    Done,
    /// The thread is blocked; whoever wakes it sets its status.
    Blocked,
    /// An interrupt is pending; the syscall runs again from the start.
    Preempted,
}

/// The kernel: one instance for the machine, entered by every processor.
pub struct Kernel<H: Hal> {
    /// The architecture.
    pub hal: H,
    /// Physical memory.
    pub phys: Phys,
    pub(crate) cpus: crate::smp::CpuTable,
    pub(crate) ncpu: core::sync::atomic::AtomicUsize,
    /// An address space isolated against speculation (`ASPACE_ISOLATED`)
    /// has been made since boot: only then does a switch look at whether it
    /// enters or leaves one (`Kernel::isolate`), so a machine without one
    /// pays a load and a branch for it.
    pub(crate) isolating: core::sync::atomic::AtomicBool,
    /// This processor holds every lock (see `smp.rs`).
    pub(crate) global: Cell<bool>,
    /// Held, with the holder's own processor lock, while mapping.
    pub(crate) structure: crate::lock::Lock,
    pub(crate) globals: Cell<u64>,
    /// Objects waiting to be emptied (only used holding every lock).
    pub(crate) reap: Cell<u64>,
    /// The CapSpace being emptied, part done (0: none; `cap::drain_reap`).
    pub(crate) reaping: Cell<u64>,
    /// The notification and badge each interrupt line signals (changed
    /// only by a holder of every lock).
    pub(crate) irqs: [(Cell<Paddr>, Cell<u64>); silica_abi::IRQ_LINES as usize],
    /// Where each line is routed and how its deliveries went.
    pub(crate) irq_stat: [crate::ipc::IrqStat; silica_abi::IRQ_LINES as usize],
    /// Watched lines (`watch.rs`), and how many entries are in use.
    pub(crate) watches: [crate::watch::LineWatch; silica_abi::IRQ_WATCHES],
    pub(crate) nwatches: Cell<usize>,
    /// How often programs may make the kernel write a console line.
    pub(crate) log_rate: crate::log::Rate,
}

impl<H: Hal> Kernel<H> {
    /// A kernel with no threads, for a machine with one processor until
    /// [`Kernel::set_cpu_count`] says otherwise.
    pub const fn new(hal: H, phys: Phys) -> Kernel<H> {
        Kernel {
            hal,
            phys,
            cpus: crate::smp::CpuTable::new(),
            ncpu: core::sync::atomic::AtomicUsize::new(1),
            isolating: core::sync::atomic::AtomicBool::new(false),
            global: Cell::new(false),
            structure: crate::lock::Lock::new(),
            globals: Cell::new(0),
            reap: Cell::new(0),
            reaping: Cell::new(0),
            irqs: [const { (Cell::new(0), Cell::new(0)) }; silica_abi::IRQ_LINES as usize],
            irq_stat: [const { crate::ipc::IrqStat::new() }; silica_abi::IRQ_LINES as usize],
            watches: [const { crate::watch::LineWatch::new() }; silica_abi::IRQ_WATCHES],
            nwatches: Cell::new(0),
            log_rate: crate::log::Rate::new(),
        }
    }

    /// The thread this processor is running (0 while idle).
    pub fn current(&self) -> Paddr {
        self.me().cur.get()
    }

    /// The saved registers of thread `t`.
    pub fn regs(&self, t: Paddr) -> &[Cell<u64>; crate::hal::NREGS] {
        &self.thread(t).regs
    }

    /// The slot `index` of the current thread's CapSpace.
    /// On every system call: the CapSpace capability's type and size are
    /// read straight from its slot, with the results of `slot_in`.
    pub(crate) fn cslot(&self, t: Paddr, index: u64) -> Result<Paddr, Error> {
        let cs = &self.thread(t).cspace;
        let m = cs.meta.get();
        if m & 0xFF != ObjectType::CapSpace as u64 {
            return Err(if ObjectType::from_u64(m & 0xFF).is_some() { Error::WrongType } else { Error::BadCap });
        }
        let bits = (m >> 16) as u8 as u32;
        if index == 0 || 1u64.checked_shl(bits).is_some_and(|n| index >= n) {
            return Err(Error::BadCap);
        }
        Ok(cs.obj.get() + (index << silica_abi::SLOT_BITS))
    }

    /// The capability at `index`, which must be of type `ty` and carry `rights`.
    pub(crate) fn ccap(&self, t: Paddr, index: u64, ty: ObjectType, rights: Rights) -> Result<(Paddr, Cap), Error> {
        let s = self.cslot(t, index)?;
        let cap = self.slot(s).cap().ok_or(Error::BadCap)?;
        if cap.ty != ty {
            return Err(Error::WrongType);
        }
        if !cap.rights.has(rights) {
            return Err(Error::NoRights);
        }
        Ok((s, cap))
    }

    /// Whether thread `x` and the thread waiting for its reply, if any,
    /// belong to processors whose locks this entry holds. `x` itself must
    /// be readable under those locks: a thread of one of them, or one found
    /// in a queue one of them owns.
    fn here(&self, h: Held, x: Paddr) -> bool {
        let th = self.thread(x);
        h.has(th.cpu.get()) && (th.caller.get() == 0 || h.has(self.thread(th.caller.get()).cpu.get()))
    }

    /// Whether the queue's owner is a processor whose lock this entry holds.
    fn owned(&self, h: Held, q: Paddr) -> bool {
        let o = self.queue(q).owner.get();
        o != 0 && h.has(o - 1)
    }

    /// Whether an IPC operation of thread `t` on the queue behind `cap`
    /// touches only what the locks this entry holds protect.
    pub(crate) fn queue_is_local(&self, h: Held, t: Paddr, cap: &Cap) -> bool {
        if !matches!(cap.ty, ObjectType::Endpoint | ObjectType::Notification) {
            return true; // an error, reported without touching anything
        }
        if !self.owned(h, cap.obj) {
            return false;
        }
        let q = self.queue(cap.obj);
        if cap.ty == ObjectType::Notification && !self.listener_is_local(h, cap.obj) {
            return false;
        }
        let head = q.head.get();
        // Answering a parked call unlinks the caller from the endpoint
        // its call came through.
        if q.kind.get() == crate::ipc::PARKED && !self.owned(h, self.thread(head).via.get()) {
            return false;
        }
        // A receiver at the head may have asked for the call it gets to be
        // parked on a notification of its own.
        if q.kind.get() == crate::ipc::RECEIVERS && !self.park_is_local(h, head, self.thread(head).park.get()) {
            return false;
        }
        (head == 0 || (self.here(h, head) && MsgInfo(self.thread(head).regs[H::ARG[1]].get()).caps() == 0)) && self.here(h, t)
    }

    /// Whether what a signal to notification `n` (whose owner's lock is
    /// held), or a wait on it, may touch of its listener is under the locks
    /// this entry holds: the thread it names, and when that thread is
    /// listening now, the endpoint it waits on.
    fn listener_is_local(&self, h: Held, n: Paddr) -> bool {
        let l = self.queue(n).listener().get();
        if l == 0 {
            return true;
        }
        if !h.has(self.thread(l).cpu.get()) {
            return false;
        }
        self.listener(n) == 0 || self.owned(h, self.thread(l).blocked_on.get())
    }

    /// Whether the notification a receive of thread `t` listens on (named
    /// in its first argument `a0`) can be made this thread's under the
    /// locks this entry holds: nobody owns it, or a held lock's processor
    /// does, and its last listener and the notification this thread
    /// listened on before are under held locks too.
    fn listen_is_local(&self, h: Held, t: Paddr, a0: u64) -> bool {
        let index = crate::ipc::listen_slot(a0);
        if index == 0 {
            return true;
        }
        match self.cslot(t, index).ok().and_then(|s| self.slot(s).cap()) {
            Some(n) if n.ty == ObjectType::Notification => {
                let free = |o: u64| o == 0 || h.has(o - 1);
                let q = self.queue(n.obj);
                let (last, old) = (q.listener().get(), self.thread(t).listen.get());
                free(q.owner.get())
                    && (last == 0 || h.has(self.thread(last).cpu.get()))
                    && (old == 0 || old == n.obj || free(self.queue(old).owner.get()))
            }
            _ => true, // an error, reported without touching anything
        }
    }

    /// Whether the notification in slot `park` of thread `server`, on which
    /// the server wants its next call parked, can be used under the locks
    /// this entry holds: nobody owns it yet, or a processor whose lock is
    /// held does.
    fn park_is_local(&self, h: Held, server: Paddr, park: u64) -> bool {
        if park == 0 {
            return true;
        }
        match self.cslot(server, park).ok().and_then(|s| self.slot(s).cap()) {
            Some(n) if n.ty == ObjectType::Notification => {
                let o = self.queue(n.obj).owner.get();
                o == 0 || h.has(o - 1)
            }
            _ => true, // not parked at all
        }
    }

    /// Whether syscall `s` can run holding only the locks this entry holds:
    /// this processor's, and after [`Kernel::go_pair`] one other's.
    fn is_local(&self, s: Syscall, t: Paddr, a: &[u64; 6]) -> bool {
        if self.ncpu() == 1 {
            return true;
        }
        let h = self.held();
        let queue = |index: u64| match self.cslot(t, index).ok().and_then(|slot| self.slot(slot).cap()) {
            Some(cap) => self.queue_is_local(h, t, &cap),
            None => true, // BadCap, reported without touching anything
        };
        match s {
            // `Map` takes the structure lock itself. `IrqAck` reads the
            // line's binding (changed only holding every lock) and unmasks
            // it under the interrupt controller's own lock.
            Syscall::Yield | Syscall::PortIo | Syscall::Map | Syscall::Console | Syscall::PowerOff | Syscall::IrqAck => true,
            // Passing capabilities changes capability slots.
            Syscall::Call => MsgInfo(a[1]).caps() == 0 && queue(a[0] & CALL_SLOT),
            Syscall::Send => MsgInfo(a[1]).caps() == 0 && queue(a[0] & !NONBLOCK),
            Syscall::Recv => queue(crate::ipc::recv_slot(a[0])) && self.park_is_local(h, t, a[2]) && self.listen_is_local(h, t, a[0]),
            Syscall::SignalRecv => self.signal_recv_is_local(h, t, a),
            Syscall::ReplyRecv => {
                // The reply unlinks the caller from the endpoint its call came through.
                let c = self.thread(t).caller.get();
                let via = c == 0 || self.owned(h, self.thread(c).via.get());
                let from = crate::ipc::recv_slot(a[0]);
                MsgInfo(a[1]).caps() == 0 && via && self.here(h, t) && (from == 0 || (queue(from) && self.listen_is_local(h, t, a[0])))
            }
            _ => false,
        }
    }

    /// For [`Kernel::partner`], of the queue in slot `index`: its owner, or
    /// (the queue being this processor's) the home of the thread at its head
    /// or of that thread's caller.
    #[inline(always)]
    fn queue_partner(&self, t: Paddr, index: u64, me: u64) -> Option<usize> {
        let other = |cpu: u64| (cpu != me).then_some(cpu as usize);
        let cap = self.cslot(t, index).ok().and_then(|slot| self.slot(slot).cap())?;
        if !matches!(cap.ty, ObjectType::Endpoint | ObjectType::Notification) {
            return None;
        }
        let q = self.queue(cap.obj);
        match q.owner.get() {
            0 => return None,
            o if o != me + 1 => return Some(o as usize - 1),
            _ => {}
        }
        let head = q.head.get();
        if head == 0 {
            return None;
        }
        let th = self.thread(head);
        if th.cpu.get() != me {
            return other(th.cpu.get());
        }
        if q.kind.get() == crate::ipc::PARKED {
            return self.queue(th.via.get()).owner.get().checked_sub(1).and_then(other);
        }
        let c = th.caller.get();
        if c != 0 { other(self.thread(c).cpu.get()) } else { None }
    }

    // `SignalRecv`, out of line in each of the three places it is decided
    // and made: inlined, the dispatch's code around it cost every path
    // through it for the registers it took (16 instructions a page fault).

    /// [`Kernel::is_local`] of `SignalRecv`: `Recv`'s, and the
    /// notification it signals.
    #[inline(never)]
    fn signal_recv_is_local(&self, h: Held, t: Paddr, a: &[u64; 6]) -> bool {
        let queue = |index: u64| match self.cslot(t, index).ok().and_then(|slot| self.slot(slot).cap()) {
            Some(cap) => self.queue_is_local(h, t, &cap),
            None => true, // BadCap, reported without touching anything
        };
        queue(crate::ipc::recv_slot(a[0])) && self.park_is_local(h, t, a[2]) && self.listen_is_local(h, t, a[0]) && queue(a[3])
    }

    /// [`Kernel::partner`] of `SignalRecv`.
    #[inline(never)]
    fn signal_recv_partner(&self, t: Paddr, a: &[u64; 6]) -> Option<usize> {
        let me = self.hal.cpu() as u64;
        self.queue_partner(t, crate::ipc::recv_slot(a[0]), me).or_else(|| self.queue_partner(t, a[3], me))
    }

    /// `SignalRecv`: the signal, then the receive. The wake-up a waiter
    /// gives before it waits, in the same kernel entry, so the thread it
    /// wakes on this processor finds it gone, not still running.
    #[inline(never)]
    fn signal_recv(&self, t: Paddr, a: &[u64; 6]) -> Result<Exit, Error> {
        let (_, n) = self.ccap(t, a[3], ObjectType::Notification, Rights::WRITE)?;
        // A signal, never a message: a call parked on the notification is
        // not answered by it (this call's registers hold a deadline, not a
        // message), and the bits accumulate for whoever answers it.
        self.signal(n.obj, n.badge.max(1));
        let r = self.recv_any(t, a[0], a[1], a[2]);
        if r == Ok(Exit::Preempted) {
            // The signal is made; what runs again is the receive alone.
            self.thread(t).regs[H::SYS].set(Syscall::Recv as u64);
        }
        r
    }

    /// For a syscall that is not local: the one other processor whose lock
    /// would make it so, if what can be read under this processor's lock
    /// names one. A guess, checked by `is_local` once both locks are held.
    fn partner(&self, s: Syscall, t: Paddr, a: &[u64; 6]) -> Option<usize> {
        let me = self.hal.cpu() as u64;
        let other = |cpu: u64| (cpu != me).then_some(cpu as usize);
        let of_queue = |index: u64| self.queue_partner(t, index, me);
        match s {
            Syscall::Call if MsgInfo(a[1]).caps() == 0 => of_queue(a[0] & CALL_SLOT),
            Syscall::Send if MsgInfo(a[1]).caps() == 0 => of_queue(a[0] & !NONBLOCK),
            Syscall::Recv => of_queue(crate::ipc::recv_slot(a[0])),
            Syscall::SignalRecv => self.signal_recv_partner(t, a),
            Syscall::ReplyRecv if MsgInfo(a[1]).caps() == 0 => {
                let c = self.thread(t).caller.get();
                let of_caller = if c == 0 {
                    None
                } else {
                    other(self.thread(c).cpu.get())
                        .or_else(|| self.queue(self.thread(c).via.get()).owner.get().checked_sub(1).and_then(other))
                };
                of_caller.or_else(|| of_queue(crate::ipc::recv_slot(a[0])))
            }
            _ => None,
        }
    }

    /// A syscall `s` of thread `t` that is not local: if it involves one
    /// other processor, take that one's lock as well and look again.
    /// Returns whether the syscall still needs every lock, or `None` when
    /// the thread no longer runs (it was stopped while the locks were taken).
    /// Out of line: the path between threads of one processor is the hot
    /// one and must not pay for this one's registers.
    #[cold]
    #[inline(never)]
    fn try_pair(&self, s: Syscall, t: Paddr, a: &[u64; 6]) -> Option<bool> {
        let Some(other) = self.partner(s, t, a) else { return Some(true) };
        self.go_pair(other);
        let me = self.me();
        if me.cur.get() != t || self.thread(t).state.get() != RUNNABLE {
            me.dirty.set(true);
            self.leave_pair();
            return None;
        }
        let global = !self.is_local(s, t, a);
        if global {
            self.leave_pair();
        }
        Some(global)
    }

    /// Handle the syscall the current thread just made. In line in the
    /// architecture's system-call entry, which never returns: then nothing
    /// saves registers for a caller on the way.
    #[inline(always)]
    pub fn syscall(&self) {
        let me = self.me();
        let t = me.cur.get();
        // While this processor waited for its lock another one may have
        // stopped or destroyed the thread that made the call.
        if t == 0 || self.thread(t).state.get() != RUNNABLE {
            me.dirty.set(true);
            return;
        }
        let r = &self.thread(t).regs;
        let Some(s) = Syscall::from_u64(r[H::SYS].get()) else {
            r[H::SYS].set(Error::BadSyscall as u64);
            return;
        };
        match s {
            // Only this processor's scheduling state.
            Syscall::Yield => {
                me.resched.set(true);
                me.yielded.set(true);
                me.dirty.set(true);
                r[H::SYS].set(0);
            }
            Syscall::Call | Syscall::ReplyRecv => {
                let g = |i: usize| r[H::ARG[i]].get();
                let a = [g(0), g(1), g(2), g(3), g(4), g(5)];
                let fast = if s == Syscall::Call { self.fast_call(t, &a) } else { self.fast_reply_recv(t, &a) };
                // A fast path leaves the thread blocked: whoever wakes it
                // sets its status.
                if !fast {
                    self.syscall_general(s, t);
                }
            }
            _ => self.syscall_general(s, t),
        }
    }

    /// Any syscall, under whatever locks it needs. Out of line: the paths
    /// above must not pay for this one's registers and stack.
    #[inline(never)]
    fn syscall_general(&self, s: Syscall, t: Paddr) {
        let me = self.me();
        let r = &self.thread(t).regs;
        let g = |i: usize| r[H::ARG[i]].get();
        let a = [g(0), g(1), g(2), g(3), g(4), g(5)];
        let mut global = !self.is_local(s, t, &a);
        if global {
            match self.try_pair(s, t, &a) {
                Some(g) => global = g,
                None => return,
            }
        }
        if global {
            // Before any other processor's lock is taken: a processor that
            // has used its share of that time takes none of them.
            if self.throttled(me, t) {
                return;
            }
            self.go_global();
            if me.cur.get() != t || self.thread(t).state.get() != RUNNABLE {
                me.dirty.set(true);
                self.go_local();
                return;
            }
        }
        match self.dispatch(s, t, a) {
            Ok(Exit::Done) => r[H::SYS].set(0),
            Ok(Exit::Blocked) => {}
            Ok(Exit::Preempted) => r[H::IP].set(r[H::IP].get().wrapping_sub(H::SYSCALL_LEN)),
            Err(e) => r[H::SYS].set(e as u64),
        }
        if global {
            self.spent_global(me);
            self.go_local();
        }
        self.leave_pair();
    }

    fn dispatch(&self, s: Syscall, t: Paddr, a: [u64; 6]) -> Result<Exit, Error> {
        let grant = |cap: &Cap| if cap.rights.has(Rights::GRANT) { IPC_GRANT } else { 0 };
        // Emptying left over from an earlier entry (a large CapSpace,
        // `drain_reap`) is finished before any system call that holds every
        // lock does anything: its memory is reused only once it is done.
        if self.reap_pending() && (self.ncpu() == 1 || self.global.get()) && !self.finish_reap() {
            return Ok(Exit::Preempted);
        }
        match s {
            Syscall::Call => {
                if a[0] & NONBLOCK != 0 {
                    return Err(Error::BadArg);
                }
                let (slot, ep) = self.ccap(t, a[0] & CALL_SLOT, ObjectType::Endpoint, Rights::WRITE)?;
                self.thread(t).via_slot.set(slot);
                let us = a[0] >> CALL_TIMEOUT_SHIFT;
                let deadline = if us == 0 { 0 } else { self.hal.now().saturating_add(us * 1000) };
                self.send(t, ep.obj, ep.badge, IPC_CALL | grant(&ep), true, deadline)
            }
            Syscall::Send => {
                let slot = self.cslot(t, a[0] & !NONBLOCK)?;
                let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
                if !cap.rights.has(Rights::WRITE) {
                    return Err(Error::NoRights);
                }
                self.thread(t).via_slot.set(slot);
                match cap.ty {
                    ObjectType::Endpoint => self.send(t, cap.obj, cap.badge, grant(&cap), a[0] & NONBLOCK == 0, 0),
                    // A capability without a badge signals bit 0.
                    ObjectType::Notification => self.notify(t, cap.obj, cap.badge.max(1)),
                    _ => Err(Error::WrongType),
                }
            }
            Syscall::Recv => self.recv_any(t, a[0], a[1], a[2]),
            Syscall::SignalRecv => self.signal_recv(t, &a),
            Syscall::ReplyRecv => {
                let caller = self.thread(t).caller.get();
                let replied = self.reply(t)?;
                if crate::ipc::recv_slot(a[0]) == 0 {
                    return if replied { Ok(Exit::Done) } else { Err(Error::NoCaller) };
                }
                let r = self.recv_any(t, a[0], 0, 0);
                if r == Ok(Exit::Preempted) {
                    // The reply is made; what runs again is the receive
                    // alone (with no deadline and nothing to park on).
                    let regs = &self.thread(t).regs;
                    regs[H::SYS].set(Syscall::Recv as u64);
                    regs[H::ARG[1]].set(0);
                    regs[H::ARG[2]].set(0);
                }
                // The server now waits: the caller it answered runs next.
                let me = self.me();
                if replied
                    && r == Ok(Exit::Blocked)
                    && caller != t
                    && self.thread(caller).state.get() == RUNNABLE
                    && self.thread(caller).cpu.get() as usize == self.hal.cpu()
                {
                    self.dequeue(me, caller);
                    self.hand_off(caller);
                }
                r
            }
            Syscall::Yield => {
                self.me().resched.set(true);
                self.me().yielded.set(true);
                self.me().dirty.set(true);
                Ok(Exit::Done)
            }
            Syscall::Retype => {
                let us = self.cslot(t, a[0])?;
                let ty = ObjectType::from_u64(a[1]).ok_or(Error::BadArg)?;
                let (_, dst) = self.ccap(t, a[3], ObjectType::CapSpace, Rights::WRITE)?;
                self.retype(us, ty, a[2], &dst, a[4], a[5])
            }
            Syscall::CapCopy => {
                if a[5] > 1 {
                    return Err(Error::BadArg);
                }
                let (_, dst) = self.ccap(t, a[0], ObjectType::CapSpace, Rights::WRITE)?;
                let d = self.slot_in(&dst, a[1])?;
                let s = self.cslot(t, a[2])?;
                self.copy(s, d, Rights(a[3] as u8), a[4], a[5] == 1)?;
                Ok(Exit::Done)
            }
            Syscall::CapMove => {
                let (_, dst) = self.ccap(t, a[0], ObjectType::CapSpace, Rights::WRITE)?;
                let d = self.slot_in(&dst, a[1])?;
                // From the caller's own CapSpace, or (a[3]) from slot a[2]
                // of another it may write: how a parent takes back what it
                // gave a child before the child goes.
                let s = if a[3] == 0 {
                    self.cslot(t, a[2])?
                } else {
                    let (_, src) = self.ccap(t, a[3], ObjectType::CapSpace, Rights::WRITE)?;
                    self.slot_in(&src, a[2])?
                };
                if self.slot(s).is_empty() {
                    return Err(Error::BadCap);
                }
                if !self.slot(d).is_empty() {
                    return Err(Error::SlotFull);
                }
                self.move_slot(s, d);
                Ok(Exit::Done)
            }
            Syscall::CapDelete => {
                let s = self.cslot(t, a[0])?;
                if !self.revoke(s) {
                    return Ok(Exit::Preempted);
                }
                self.delete(s);
                // Made again until what it destroyed is emptied: the
                // deleter pays for it, not the next caller.
                Ok(if self.reap_pending() { Exit::Preempted } else { Exit::Done })
            }
            Syscall::CapRevoke => {
                let s = self.cslot(t, a[0])?;
                if self.slot(s).is_empty() {
                    return Err(Error::BadCap);
                }
                Ok(if self.revoke(s) { Exit::Done } else { Exit::Preempted })
            }
            Syscall::Map => {
                let (_, asp) = self.ccap(t, a[0], ObjectType::AddressSpace, Rights::WRITE)?;
                let (fs, _) = self.ccap(t, a[1], ObjectType::Frame, Rights::NONE)?;
                self.map(asp.obj, fs, a[2], Rights(a[3] as u8))?;
                Ok(Exit::Done)
            }
            Syscall::Unmap => {
                // One lookup: a frame's own mapping, or a range of a space.
                let s = self.cslot(t, a[0])?;
                let cap = self.slot(s).cap().ok_or(Error::BadCap)?;
                match cap.ty {
                    ObjectType::Frame => self.unmap_slot(s),
                    ObjectType::AddressSpace => {
                        if !cap.rights.has(Rights::WRITE) {
                            return Err(Error::NoRights);
                        }
                        let end = a[1].checked_add(a[2]).ok_or(Error::BadArg)?;
                        let n = self.unmap_range(cap.obj, a[1], end);
                        self.thread(t).regs[H::ARG[0]].set(n);
                    }
                    _ => return Err(Error::WrongType),
                }
                Ok(Exit::Done)
            }
            Syscall::ThreadConfigure => {
                let (_, th) = self.ccap(t, a[0], ObjectType::Thread, Rights::WRITE)?;
                let want = [ObjectType::CapSpace, ObjectType::AddressSpace, ObjectType::Endpoint, ObjectType::Frame];
                // What the thread does with each is a use of it the
                // capability must allow: running in a CapSpace or an
                // address space is full use of its slots or mappings (read
                // and write), a fault is sent through the endpoint, the IPC
                // buffer is read and written.
                let rw = Rights(Rights::READ.0 | Rights::WRITE.0);
                let need = [rw, rw, Rights::WRITE, rw];
                // Validate everything before changing anything.
                let mut src = [0; 4];
                for i in 0..4 {
                    let arg = a[i + 1];
                    if arg == KEEP || arg == 0 {
                        continue;
                    }
                    let (s, cap) = self.ccap(t, arg, want[i], need[i])?;
                    if i == 3 && (cap.bits != PAGE_BITS || cap.flags & FLAG_DEVICE != 0) {
                        return Err(Error::BadArg);
                    }
                    // No thread runs in an address space made for devices.
                    if i == 1 && cap.flags & FLAG_IO != 0 {
                        return Err(Error::BadArg);
                    }
                    src[i] = s;
                }
                // The thread may be running on another processor with the
                // CapSpace and address space it is about to lose.
                let home = self.thread(th.obj).cpu.get() as usize;
                self.cpus.get(home).dirty.set(true);
                if home != self.hal.cpu() && self.cpus.get(home).cur.get() == th.obj {
                    self.hal.kick(home, true);
                }
                let slots = Thread::embedded_slots(th.obj);
                for i in 0..4 {
                    if a[i + 1] == KEEP {
                        continue;
                    }
                    self.delete(slots[i]);
                    // The thread's capability is a sibling of the source: it
                    // does not go away when the caller deletes its own.
                    if src[i] != 0
                        && let Some(mut cap) = self.slot(src[i]).cap()
                    {
                        if cap.ty == ObjectType::Frame {
                            cap.badge = 0;
                        }
                        self.insert_sibling(src[i], slots[i], &cap);
                    }
                }
                Ok(Exit::Done)
            }
            Syscall::ThreadStart => {
                let (_, th) = self.ccap(t, a[0], ObjectType::Thread, Rights::WRITE)?;
                if !H::user_ip_ok(a[1]) {
                    return Err(Error::BadArg);
                }
                self.cancel(th.obj, false);
                self.fresh_extended_state(th.obj);
                let r = &self.thread(th.obj).regs;
                for c in r.iter() {
                    c.set(0);
                }
                r[H::IP].set(a[1]);
                r[H::SP].set(a[2]);
                r[H::ARG[0]].set(a[3]);
                r[H::FLAGS].set(H::FLAGS_INIT);
                if a[4] != 0 {
                    self.make_runnable(th.obj);
                }
                Ok(if th.obj == t { Exit::Blocked } else { Exit::Done })
            }
            Syscall::ThreadControl => {
                let (_, th) = self.ccap(t, a[0], ObjectType::Thread, Rights::WRITE)?;
                match a[1] {
                    0 => {
                        // The caller's own status must be written before it stops.
                        if th.obj == t {
                            self.thread(t).regs[H::SYS].set(0);
                        }
                        self.cancel(th.obj, true);
                        Ok(if th.obj == t { Exit::Blocked } else { Exit::Done })
                    }
                    1 => {
                        if self.thread(th.obj).state.get() == INACTIVE {
                            self.make_runnable(th.obj);
                        }
                        Ok(Exit::Done)
                    }
                    2 => {
                        // Move the thread to processor `a2`, doing what it
                        // was doing. A deadline reservation is on one
                        // processor: unbind it first. A group is on one
                        // processor too, and is left only for another: a
                        // thread moved out of it would take a share of its
                        // own (or of another processor's own group).
                        if a[2] >= self.ncpu() as u64 {
                            return Err(Error::BadArg);
                        }
                        if self.dl_of(th.obj).is_some()
                            || (self.thread(th.obj).group.get() != crate::sched::UNBOUND && a[2] != self.thread(th.obj).cpu.get())
                        {
                            return Err(Error::InUse);
                        }
                        if th.obj == t {
                            self.thread(t).regs[H::SYS].set(0);
                        }
                        self.move_thread(th.obj, a[2] as usize);
                        Ok(if th.obj == t { Exit::Blocked } else { Exit::Done })
                    }
                    3 => {
                        let s = if a[2] == 0 { 0 } else { self.ccap(t, a[2], ObjectType::SchedContext, Rights::READ)?.1.obj };
                        let was = self.thread(th.obj).cpu.get();
                        // A group on another processor takes the thread there:
                        // the caller's own status is written before it goes.
                        if th.obj == t {
                            self.thread(t).regs[H::SYS].set(0);
                        }
                        self.sched_bind(th.obj, s)?;
                        let home = self.thread(th.obj).cpu.get() as usize;
                        if th.obj == t && home as u64 != was {
                            return Ok(Exit::Blocked);
                        }
                        if home != self.hal.cpu() && self.cpus.get(home).cur.get() == th.obj {
                            self.hal.kick(home, false);
                        }
                        Ok(Exit::Done)
                    }
                    _ => Err(Error::BadArg),
                }
            }
            Syscall::IrqBind | Syscall::IrqAck => {
                let (_, hw) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if hw.flags & crate::cap::FLAG_IRQS == 0 {
                    return Err(Error::WrongType);
                }
                if a[1] < hw.obj || a[1] >= hw.badge || a[1] >= silica_abi::IRQ_LINES {
                    return Err(Error::BadArg);
                }
                let line = a[1] as usize;
                if s == Syscall::IrqAck {
                    if self.irqs[line].0.get() != 0 {
                        self.hal.irq_mask(line, false);
                    }
                    return Ok(Exit::Done);
                }
                let (to, badge) = if a[2] == 0 {
                    (0, 0)
                } else {
                    let (_, n) = self.ccap(t, a[2], ObjectType::Notification, Rights::WRITE)?;
                    (n.obj, n.badge.max(1))
                };
                // Everything is checked, and the line routed anew (which
                // changes nothing when it fails), before the old binding is
                // touched: a bind that fails leaves the line as it was.
                let msi = line >= self.hal.msi_first();
                // To the processor of whoever waits on the notification (a
                // driver binds its lines before its thread is placed, or
                // from another thread), else the binder's.
                let cpu = if to == 0 { 0 } else { self.waiter_cpu(to).unwrap_or(self.hal.cpu() as u64) as usize };
                if to != 0 {
                    if msi {
                        // A message, raised by one device.
                        if a[3] != silica_abi::IRQ_MSI {
                            return Err(Error::BadArg);
                        }
                        let source = self.device(t, a[4], a[5])?;
                        let (address, data) = self.hal.msi_route(line, cpu, source).ok_or(Error::Unsupported)?;
                        let r = &self.thread(t).regs;
                        r[H::ARG[0]].set(address);
                        r[H::ARG[1]].set(data);
                    } else if a[3] & silica_abi::IRQ_MSI != 0 {
                        return Err(Error::BadArg);
                    } else if !self.hal.irq_route(line, cpu, a[3] & silica_abi::IRQ_LEVEL != 0, a[3] & silica_abi::IRQ_ACTIVE_LOW != 0) {
                        return Err(Error::Unsupported);
                    }
                }
                // Whatever the line signalled before no longer hears from it,
                // and its news is no longer known.
                self.unwatch_line(line);
                let old = self.irqs[line].0.replace(0);
                if old != 0 {
                    let q = self.queue(old);
                    q.bound.set(q.bound.get() - 1);
                }
                if to == 0 {
                    self.irq_stat[line].cpu.set(0);
                    self.hal.irq_mask(line, true);
                    // A message line unbound: no device may raise it.
                    if msi && old != 0 {
                        self.hal.msi_release(line);
                    }
                } else {
                    self.irqs[line].0.set(to);
                    let stat = &self.irq_stat[line];
                    stat.cpu.set(cpu as u64 + 1);
                    stat.fired.store(0, core::sync::atomic::Ordering::Relaxed);
                    stat.far.store(0, core::sync::atomic::Ordering::Relaxed);
                    self.irqs[line].1.set(badge);
                    let q = self.queue(to);
                    q.bound.set(q.bound.get() + 1);
                }
                Ok(Exit::Done)
            }
            Syscall::IrqQuiet => {
                let (_, hw) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if hw.flags & crate::cap::FLAG_IRQS == 0 {
                    return Err(Error::WrongType);
                }
                if a[1] < hw.obj || a[1] >= hw.badge || a[1] >= silica_abi::IRQ_LINES {
                    return Err(Error::BadArg);
                }
                let (slot, frame) = self.ccap(t, a[2], ObjectType::Frame, Rights(Rights::READ.0 | Rights::WRITE.0))?;
                let size = 1u64 << frame.bits;
                let bits = a[4] as u32;
                if frame.flags & crate::cap::FLAG_DEVICE != 0 || !a[3].is_multiple_of(4) || a[3] >= size || bits == 0 || a[4] >> 32 != 0 {
                    return Err(Error::BadArg);
                }
                if !self.quiet_line(a[1] as usize, slot, frame.obj, a[3], bits) {
                    return Err(Error::BadArg);
                }
                Ok(Exit::Done)
            }
            Syscall::CapNotice => {
                let watched = self.cslot(t, a[0])?;
                let (n, _) = self.ccap(t, a[1], ObjectType::Notification, Rights::WRITE)?;
                let notice = self.cslot(t, a[2])?;
                self.notice(watched, n, notice, a[3])?;
                Ok(Exit::Done)
            }
            Syscall::IrqWatch => {
                let (_, hw) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if hw.flags & crate::cap::FLAG_IRQS == 0 {
                    return Err(Error::WrongType);
                }
                if a[1] < hw.obj || a[1] >= hw.badge || a[1] >= silica_abi::IRQ_LINES {
                    return Err(Error::BadArg);
                }
                let line = a[1] as usize;
                if a[2] == 0 {
                    self.unwatch_line(line);
                    return Ok(Exit::Done);
                }
                let (slot, frame) = self.ccap(t, a[2], ObjectType::Frame, Rights::READ)?;
                let size = 1u64 << frame.bits;
                // A message line, bound: a wire may be shared, and its
                // interrupt held off until acknowledged.
                let mask = a[4] as u32;
                if line < self.hal.msi_first()
                    || self.irqs[line].0.get() == 0
                    || frame.flags & crate::cap::FLAG_DEVICE != 0
                    || !a[3].is_multiple_of(4)
                    || a[3] >= size
                    || mask == 0
                    || a[4] >> 32 != 0
                {
                    return Err(Error::BadArg);
                }
                if !self.watch_line(line, self.hal.cpu(), slot, frame.obj, a[3], mask) {
                    return Err(Error::NoMemory);
                }
                Ok(Exit::Done)
            }
            Syscall::Inspect => {
                let slot = self.cslot(t, a[0])?;
                let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
                // Looking is a use: it needs the read right.
                if !cap.rights.has(Rights::READ) {
                    return Err(Error::NoRights);
                }
                let r = &self.thread(t).regs;
                let thread = |obj: Paddr| {
                    let th = self.thread(obj);
                    [th.state.get(), th.cpu.get(), th.regs[H::IP].get(), th.regs[H::SP].get(), th.blocked_on.get(), 0]
                };
                let isolated = |c: &crate::cap::Cap| {
                    let yes = c.ty == ObjectType::AddressSpace && c.flags & crate::cap::FLAG_ISOLATED != 0;
                    if yes { silica_abi::INSPECT_ISOLATED } else { 0 }
                };
                let out = match cap.ty {
                    ObjectType::CapSpace if a[2] == 1 => {
                        // The thread in that slot of the CapSpace.
                        let s = self.slot_in(&cap, a[1])?;
                        match self.slot(s).cap() {
                            Some(c) if c.ty == ObjectType::Thread => thread(c.obj),
                            Some(_) => return Err(Error::WrongType),
                            None => return Err(Error::BadCap),
                        }
                    }
                    ObjectType::CapSpace if a[2] == 0 => {
                        let s = self.slot_in(&cap, a[1])?;
                        match self.view(s) {
                            Some(v) => [v.cap.ty as u64, v.cap.rights.0 as u64, v.cap.bits as u64, v.cap.obj, v.cap.badge, v.depth as u64 | isolated(&v.cap)],
                            None => [0; 6],
                        }
                    }
                    ObjectType::Thread if a[1] == 0 && a[2] == 0 => thread(cap.obj),
                    ObjectType::SchedContext => {
                        let sc = self.sched(cap.obj);
                        let deadline = sc.class.get() == silica_abi::SCHED_DEADLINE;
                        let size = if deadline { sc.budget.get() } else { sc.weight.get() };
                        let cpu = sc.cpu.get().saturating_sub(1);
                        [sc.class.get(), size, sc.period.get(), sc.remaining.get(), sc.consumed.get(), cpu]
                    }
                    ObjectType::CapSpace | ObjectType::Thread => return Err(Error::BadArg),
                    // Whether a call is parked on the notification, its
                    // caller still waiting for the answer: how a server
                    // tells a parked caller from one that has gone.
                    ObjectType::Notification => [(self.queue(cap.obj).kind.get() == crate::ipc::PARKED) as u64, 0, 0, 0, 0, 0],
                    // An interrupt line: the processor it is routed to plus one
                    // (0: not bound), how many interrupts it delivered, and how
                    // many of them took every lock to reach a waiter elsewhere.
                    ObjectType::Hw if cap.flags & crate::cap::FLAG_IRQS != 0 => {
                        if a[1] < cap.obj || a[1] >= cap.badge || a[1] >= silica_abi::IRQ_LINES {
                            return Err(Error::BadArg);
                        }
                        let s = &self.irq_stat[a[1] as usize];
                        let n = |c: &core::sync::atomic::AtomicU64| c.load(core::sync::atomic::Ordering::Relaxed);
                        [s.cpu.get(), n(&s.fired), n(&s.far), 0, 0, 0]
                    }
                    // Where a device reaches a frame on a machine that
                    // cannot confine DMA: its physical address.
                    ObjectType::Hw if cap.flags & FLAG_DMA != 0 => {
                        let (_, frame) = self.ccap(t, a[1], ObjectType::Frame, Rights::NONE)?;
                        if self.hal.io_top() != 0 {
                            return Err(Error::Unsupported);
                        }
                        [frame.obj, 0, 0, 0, 0, 0]
                    }
                    _ => return Err(Error::WrongType),
                };
                for (i, v) in out.iter().enumerate() {
                    r[H::ARG[i]].set(*v);
                }
                Ok(Exit::Done)
            }
            Syscall::SchedConfigure => {
                let (_, sc) = self.ccap(t, a[0], ObjectType::SchedContext, Rights::WRITE)?;
                // The processors whose time the caller may give out (a5: a
                // processor-time capability; 0: none, its own share only).
                let time = if a[5] == 0 {
                    0..0
                } else {
                    let (_, hw) = self.ccap(t, a[5], ObjectType::Hw, Rights::WRITE)?;
                    if hw.flags & crate::cap::FLAG_SCHED == 0 {
                        return Err(Error::WrongType);
                    }
                    hw.obj..hw.badge
                };
                self.sched_configure(sc.obj, a[1], a[2], a[3], a[4], time)?;
                // Its threads' processors look at their queues again.
                let mut b = self.sched(sc.obj).threads.get();
                while b != 0 {
                    let home = self.thread(b).cpu.get() as usize;
                    self.cpus.get(home).dirty.set(true);
                    self.cpus.get(home).resched.set(true);
                    b = self.thread(b).sc_next.get();
                }
                Ok(Exit::Done)
            }
            Syscall::DeviceAttach => {
                let id = self.device(t, a[0], a[1])?;
                let space = if a[2] == 0 {
                    0
                } else {
                    let (_, asp) = self.ccap(t, a[2], ObjectType::AddressSpace, Rights::WRITE)?;
                    if asp.flags & FLAG_IO == 0 {
                        return Err(Error::WrongType);
                    }
                    asp.obj
                };
                self.attach(id, space)?;
                Ok(Exit::Done)
            }
            Syscall::Console => {
                let (_, cap) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if cap.flags & FLAG_CONSOLE == 0 {
                    return Err(Error::WrongType);
                }
                let (n, more) = ((a[1] & !silica_abi::CONSOLE_MORE) as usize, a[1] & silica_abi::CONSOLE_MORE != 0);
                if n > 32 {
                    return Err(Error::BadArg);
                }
                let mut bytes = [0u8; 32];
                for (i, w) in a[2..6].iter().enumerate() {
                    bytes[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
                }
                let th = self.thread(t);
                let mut line = [0u8; crate::thread::LINE_WORDS * 8];
                let held = th.line_len.get() as usize;
                for (i, w) in th.line.iter().enumerate() {
                    line[i * 8..i * 8 + 8].copy_from_slice(&w.get().to_le_bytes());
                }
                // A thread object is aligned: its address with the low bit
                // set is never 0 nor the kernel.
                let writer = t | 1;
                if more && held + n <= line.len() {
                    // Held until the last piece, so the whole reaches the
                    // console together.
                    line[held..held + n].copy_from_slice(&bytes[..n]);
                    for (i, w) in th.line.iter().enumerate() {
                        w.set(u64::from_le_bytes(line[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8])));
                    }
                    th.line_len.set((held + n) as u64);
                } else {
                    crate::log::write_parts(&self.hal, writer, &[&line[..held], &bytes[..n]]);
                    th.line_len.set(0);
                }
                Ok(Exit::Done)
            }
            Syscall::PowerOff => {
                let (_, cap) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if cap.flags & FLAG_POWER == 0 {
                    return Err(Error::WrongType);
                }
                match a[2] {
                    0 => self.hal.power_off(a[1]),
                    1 => self.hal.reset(a[1]),
                    _ => return Err(Error::BadArg),
                }
                Err(Error::Unsupported)
            }
            Syscall::PortIo => {
                let (width, write) = (a[2] & 0xFF, a[2] & 0x100 != 0);
                let need = if write { Rights::WRITE } else { Rights::READ };
                let (_, cap) = self.ccap(t, a[0], ObjectType::Hw, need)?;
                if cap.flags & FLAG_PORTS == 0 {
                    return Err(Error::WrongType);
                }
                if !matches!(width, 1 | 2 | 4) || a[1] < cap.obj || a[1].checked_add(width).is_none_or(|e| e > cap.badge) {
                    return Err(Error::BadArg);
                }
                let v = self.hal.port_io(a[1] as u16, width as u8, write, a[3] as u32);
                self.thread(t).regs[H::ARG[0]].set(v as u64);
                Ok(Exit::Done)
            }
        }
    }

    /// Finish emptying what an earlier entry left (`drain_reap`); false
    /// when an interrupt stops it again. Holding every lock.
    #[cold]
    #[inline(never)]
    fn finish_reap(&self) -> bool {
        self.drain_reap()
    }

    /// `ThreadStart`: thread `t` starts a new program, and none of the
    /// previous one's vector and x87 state goes with the thread object. That
    /// state may still be live in a processor's registers (the thread was
    /// running, or that processor went idle with it loaded); that processor
    /// forgets it is loaded, so it is never saved over the initial image put
    /// in the save area now. Out of line: in line, the dispatch around it
    /// cost the page fault 32 instructions (`bench.sh --only insns`). The
    /// caller holds every lock.
    #[cold]
    #[inline(never)]
    fn fresh_extended_state(&self, t: Paddr) {
        for cpu in self.cpus.iter(self.ncpu()) {
            if cpu.loaded.get() == t {
                cpu.loaded.set(0);
                cpu.dirty.set(true);
            }
        }
        let fpu = t + crate::thread::FPU_OFFSET as u64;
        self.phys.zero(fpu, (1 << silica_abi::THREAD_BITS) - crate::thread::FPU_OFFSET as u64);
        self.hal.init_thread(&self.phys, t);
    }

    /// The device identity `id`, which the device-identity capability at
    /// `index` must cover.
    fn device(&self, t: Paddr, index: u64, id: u64) -> Result<u64, Error> {
        let (_, hw) = self.ccap(t, index, ObjectType::Hw, Rights::WRITE)?;
        if hw.flags & FLAG_DMA == 0 {
            return Err(Error::WrongType);
        }
        if id < hw.obj || id >= hw.badge {
            return Err(Error::BadArg);
        }
        Ok(id)
    }

    fn recv_any(&self, t: Paddr, a0: u64, deadline: u64, park: u64) -> Result<Exit, Error> {
        let slot = self.cslot(t, crate::ipc::recv_slot(a0))?;
        let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
        if !cap.rights.has(Rights::READ) {
            return Err(Error::NoRights);
        }
        let listen = crate::ipc::listen_slot(a0);
        match cap.ty {
            ObjectType::Endpoint => {
                let listen = if listen == 0 {
                    (0, 0)
                } else {
                    let (s, n) = self.ccap(t, listen, ObjectType::Notification, Rights::READ)?;
                    (s, n.obj)
                };
                self.thread(t).via_slot.set(slot);
                self.recv(t, cap.obj, deadline, park, listen)
            }
            // A notification is waited on by itself.
            ObjectType::Notification if listen != 0 => Err(Error::BadArg),
            ObjectType::Notification => {
                self.thread(t).via_slot.set(slot);
                self.wait(t, cap.obj, deadline)
            }
            _ => Err(Error::WrongType),
        }
    }

    /// Copy the capability in slot `s` into the empty slot `d`: as a child of
    /// `s`, or with `sibling` as an independent capability beside it.
    pub(crate) fn copy(&self, s: Paddr, d: Paddr, mask: Rights, badge: u64, sibling: bool) -> Result<(), Error> {
        let mut cap = self.slot(s).cap().ok_or(Error::BadCap)?;
        if !self.slot(d).is_empty() {
            return Err(Error::SlotFull);
        }
        // A notice is never copied: it stays a leaf (`CapNotice`).
        if self.slot(s).meta.get() & crate::cap::META_NOTICE != 0 {
            return Err(Error::InUse);
        }
        cap.rights = Rights(cap.rights.0 & mask.0);
        match cap.ty {
            // An Untyped has one owner; split it by retyping.
            ObjectType::Untyped => return Err(Error::InUse),
            ObjectType::Endpoint | ObjectType::Notification if badge != 0 => {
                if cap.badge != 0 {
                    return Err(Error::BadArg);
                }
                cap.badge = badge;
            }
            ObjectType::Hw if badge != 0 => {
                // Narrow the range: low 32 bits the new start, high 32 bits the new end.
                let (start, end) = (badge & 0xFFFF_FFFF, badge >> 32);
                if start < cap.obj || end > cap.badge || start >= end {
                    return Err(Error::BadArg);
                }
                cap.obj = start;
                cap.badge = end;
            }
            ObjectType::Frame => cap.badge = 0, // a copy starts unmapped
            _ if badge != 0 => return Err(Error::BadArg),
            _ => {}
        }
        if sibling {
            self.insert_sibling(s, d, &cap);
            return Ok(());
        }
        self.insert_child(s, d, &cap)
    }
}

/// A slot as an inspector sees it (the debug console, the tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotView {
    /// The capability.
    pub cap: Cap,
    /// Generations between this capability and its root ancestor.
    pub depth: u16,
    /// Neighbours in the derivation list (0: none).
    pub prev: Paddr,
    /// See `prev`.
    pub next: Paddr,
    /// For a mapped Frame: the address space and virtual address.
    pub mapped: Option<(Paddr, u64)>,
    /// For an Untyped: bytes used, and bytes known to be zero from the start.
    pub watermark: (u64, u64),
    /// `CapNotice`: the slot of this capability's notice (0: not watched).
    pub watched_by: Paddr,
    /// `CapNotice`: for a notice, the slot it watches (0: not a notice).
    pub notice_of: Paddr,
}

/// A thread as an inspector sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadView {
    /// One of the state constants of [`crate::thread`].
    pub state: u64,
    /// The thread's embedded slots: CapSpace, address space, fault endpoint, IPC buffer.
    pub slots: [Paddr; 4],
    /// What it is blocked on (0: nothing).
    pub blocked_on: Paddr,
    /// The thread it owes a reply (0: none).
    pub caller: Paddr,
    /// The processor it runs on.
    pub cpu: usize,
}

impl<H: Hal> Kernel<H> {
    /// Address of slot `index` in the CapSpace `cs` names.
    pub fn slot_addr(&self, cs: &Cap, index: u64) -> Result<Paddr, Error> {
        self.slot_in(cs, index)
    }

    /// What slot `s` holds. `s` must be a slot address obtained from
    /// [`Kernel::slot_addr`], a [`ThreadView`] or a [`SlotView`] link.
    pub fn view(&self, s: Paddr) -> Option<SlotView> {
        let slot = self.slot(s);
        let cap = slot.cap()?;
        let mapped = (cap.ty == ObjectType::Frame && slot.badge.get() & 1 != 0).then(|| (slot.aux.get(), slot.badge.get() & !0xFFF));
        let watermark = if cap.ty == ObjectType::Untyped { (slot.badge.get(), slot.aux.get()) } else { (0, 0) };
        let m = slot.meta.get();
        let watched_by = if m & crate::cap::META_WATCHED != 0 { slot.mnext.get() } else { 0 };
        let notice_of = if m & crate::cap::META_NOTICE != 0 { slot.mprev.get() } else { 0 };
        Some(SlotView { cap, depth: slot.depth(), prev: slot.prev.get(), next: slot.next.get(), mapped, watermark, watched_by, notice_of })
    }

    /// The state of the thread object at `t` (from a Thread capability).
    pub fn thread_view(&self, t: Paddr) -> ThreadView {
        let th = self.thread(t);
        ThreadView {
            state: th.state.get(),
            slots: Thread::embedded_slots(t),
            blocked_on: th.blocked_on.get(),
            caller: th.caller.get(),
            cpu: th.cpu.get() as usize,
        }
    }

    /// Whether every processor's run queues are well formed: the fair
    /// queue a balanced tree of the processor's groups with threads
    /// waiting, in order of the group's virtual runtime, each a balanced
    /// tree of its own waiting threads in order of theirs; the deadline
    /// queue a list linked both ways; each ending and holding only runnable
    /// threads of that processor that are not running, of the queue's class
    /// and group. For tests and fuzzing.
    pub fn check_runqueues(&self) -> Result<(), &'static str> {
        use crate::order;
        use crate::sched::{Groups, Members, UNBOUND};
        for (i, cpu) in self.cpus.iter(self.ncpu()).enumerate() {
            let groups = Groups(self, cpu);
            let ng = order::check(&groups, 1 << 20)?;
            let waiting = |t: Paddr, dl: bool| {
                let th = self.thread(t);
                if th.cpu.get() as usize != i || th.state.get() != RUNNABLE || t == cpu.cur.get() {
                    return Err("a run queue holds a thread that is not a waiting runnable thread of its processor");
                }
                if self.dl_of(t).is_some() != dl {
                    return Err("a thread is on the run queue of the wrong class");
                }
                Ok(())
            };
            let (mut g, mut seen, mut own) = (cpu.gq_first.get(), 0usize, false);
            while g != 0 {
                if g == UNBOUND {
                    own = true;
                } else if self.sched(g).class.get() != silica_abi::SCHED_GROUP || self.sched(g).cpu.get() != i as u64 + 1 {
                    return Err("a processor's run queue holds a group that is not one of its own");
                }
                let gr = self.grp(cpu, g);
                if gr.first.get() == 0 {
                    return Err("a group with no thread waiting is on its processor's run queue");
                }
                let members = Members(self, gr);
                let n = order::check(&members, 1 << 20)?;
                let (mut t, mut m) = (gr.first.get(), 0usize);
                while t != 0 {
                    waiting(t, false)?;
                    if self.group_of(t) != g {
                        return Err("a group's run queue holds a thread of another group");
                    }
                    m += 1;
                    t = order::next(&members, t);
                }
                if m != n {
                    return Err("a group's run queue's order does not reach every thread in it");
                }
                seen += 1;
                g = order::next(&groups, g);
            }
            if seen != ng {
                return Err("the fair run queue's order does not reach every group in it");
            }
            if cpu.unbound.first.get() != 0 && !own {
                return Err("a processor's own group has threads waiting and is not on its run queue");
            }
            let (mut t, mut prev, mut n) = (cpu.dl_head.get(), 0, 0usize);
            while t != 0 {
                let th = self.thread(t);
                if th.qprev.get() != prev {
                    return Err("a run queue's back link is wrong");
                }
                waiting(t, true)?;
                n += 1;
                if n > 1 << 20 {
                    return Err("a run queue does not end");
                }
                (prev, t) = (t, th.qnext.get());
            }
            if cpu.dl_tail.get() != prev {
                return Err("a run queue's tail is wrong");
            }
        }
        Ok(())
    }

    /// Whether every processor's timers are well formed: a balanced tree
    /// in deadline order holding only threads of that processor that have
    /// a deadline. For tests and fuzzing.
    pub fn check_timers(&self) -> Result<(), &'static str> {
        use crate::order;
        for (i, cpu) in self.cpus.iter(self.ncpu()).enumerate() {
            let timers = crate::thread::Timers(self, cpu);
            order::check(&timers, 1 << 20)?;
            let mut t = cpu.timers.get();
            while t != 0 {
                let th = self.thread(t);
                if th.cpu.get() as usize != i {
                    return Err("a thread waits on the timers of a processor that is not its home");
                }
                if th.deadline.get() == 0 {
                    return Err("a processor's timers hold a thread with no deadline");
                }
                t = order::next(&timers, t);
            }
        }
        Ok(())
    }

    /// Check the waiting lists of the endpoint or notification at `q`
    /// (from a capability to it) for consistency: every thread queued
    /// there waits on it, every delivered call is linked to it, and each
    /// of them came through a slot that still holds a capability to it.
    /// For tests and fuzzing; says what is wrong.
    pub fn check_queue(&self, q: Paddr, notification: bool) -> Result<(), &'static str> {
        use crate::thread::{RECEIVING, REPLY, SENDING};
        let qu = self.queue(q);
        let through = |t: Paddr| {
            let s = self.thread(t).via_slot.get();
            s == 0 || self.slot(s).cap().is_some_and(|c| c.obj == q)
        };
        let (mut t, mut prev) = (qu.head.get(), 0);
        if qu.kind.get() == crate::ipc::PARKED {
            let th = self.thread(t);
            if t == 0 || qu.tail.get() != t || th.state.get() != REPLY || th.blocked_on.get() != q || th.via.get() == 0 {
                return Err("a notification marked as holding a parked call does not hold one");
            }
        } else {
            while t != 0 {
                let th = self.thread(t);
                if !matches!(th.state.get(), SENDING | RECEIVING) || th.blocked_on.get() != q || th.qprev.get() != prev {
                    return Err("a queued thread does not wait on the queue it is linked in");
                }
                if !through(t) {
                    return Err("a thread waits through a slot that no longer holds the capability");
                }
                (prev, t) = (t, th.qnext.get());
            }
            if qu.tail.get() != prev || (prev == 0) != (qu.kind.get() == 0) {
                return Err("the queue's tail or kind does not match its contents");
            }
        }
        let (mut c, mut prev) = (qu.pend_head.get(), 0);
        while c != 0 {
            let ct = self.thread(c);
            if ct.state.get() != REPLY || ct.via.get() != q || ct.qprev.get() != prev {
                return Err("a call in progress is not linked to its endpoint");
            }
            if !through(c) {
                return Err("a call in progress came through a slot that no longer holds the capability");
            }
            (prev, c) = (c, ct.qnext.get());
        }
        if notification {
            // Where an endpoint keeps that list's tail, a notification
            // names its last listener, which names it back.
            let l = qu.listener().get();
            if l != 0 && self.thread(l).listen.get() != q {
                return Err("a notification names a listener that does not listen on it");
            }
        } else if qu.pend_tail.get() != prev {
            return Err("the list of calls in progress has the wrong tail");
        }
        Ok(())
    }

    /// Whether any thread of this processor can run now or will be woken by a deadline.
    pub fn can_run(&self) -> bool {
        let me = self.me();
        let c = me.cur.get();
        (c != 0 && self.thread(c).state.get() == RUNNABLE)
            || me.gq_first.get() != 0
            || me.dl_head.get() != 0
            || me.direct.get() != 0
            || me.timers.get() != 0
    }
}
