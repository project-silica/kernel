//! The kernel state for one processor and the syscall dispatcher.

use crate::cap::{Cap, FLAG_CONSOLE, FLAG_DEVICE, FLAG_PORTS, FLAG_POWER};
use crate::hal::Hal;
use crate::mem::{Paddr, Phys};
use crate::smp::{Cpu, Held, MAX_CPUS};
use crate::thread::{INACTIVE, IPC_CALL, IPC_GRANT, RUNNABLE, Thread};
use core::cell::Cell;
use silica_abi::{Error, KEEP, MsgInfo, NONBLOCK, ObjectType, PAGE_BITS, Rights, Syscall};

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
    pub(crate) cpus: [Cpu; MAX_CPUS],
    pub(crate) ncpu: Cell<usize>,
    /// This processor holds every lock (see `smp.rs`).
    pub(crate) global: Cell<bool>,
    /// Held, with the holder's own processor lock, while mapping.
    pub(crate) structure: crate::lock::Lock,
    pub(crate) globals: Cell<u64>,
    /// Objects waiting to be emptied (only used holding every lock).
    pub(crate) reap: Cell<u64>,
    /// The notification and badge each interrupt line signals (changed
    /// only by a holder of every lock).
    pub(crate) irqs: [(Cell<Paddr>, Cell<u64>); silica_abi::IRQ_LINES as usize],
}

impl<H: Hal> Kernel<H> {
    /// A kernel with no threads, for a machine with one processor until
    /// [`Kernel::set_cpu_count`] says otherwise.
    pub const fn new(hal: H, phys: Phys) -> Kernel<H> {
        Kernel {
            hal,
            phys,
            cpus: [const { Cpu::new() }; MAX_CPUS],
            ncpu: Cell::new(1),
            global: Cell::new(false),
            structure: crate::lock::Lock::new(),
            globals: Cell::new(0),
            reap: Cell::new(0),
            irqs: [const { (Cell::new(0), Cell::new(0)) }; silica_abi::IRQ_LINES as usize],
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
        if self.ncpu.get() == 1 {
            return true;
        }
        let h = self.held();
        let queue = |index: u64| match self.cslot(t, index).ok().and_then(|slot| self.slot(slot).cap()) {
            Some(cap) => self.queue_is_local(h, t, &cap),
            None => true, // BadCap, reported without touching anything
        };
        match s {
            // `Map` takes the structure lock itself.
            Syscall::Yield | Syscall::PortIo | Syscall::Map | Syscall::Console | Syscall::PowerOff => true,
            // Passing capabilities changes capability slots.
            Syscall::Call | Syscall::Send => MsgInfo(a[1]).caps() == 0 && queue(a[0] & !NONBLOCK),
            Syscall::Recv => queue(a[0]) && self.park_is_local(h, t, a[2]),
            Syscall::ReplyRecv => {
                // The reply unlinks the caller from the endpoint its call came through.
                let c = self.thread(t).caller.get();
                let via = c == 0 || self.owned(h, self.thread(c).via.get());
                MsgInfo(a[1]).caps() == 0 && via && self.here(h, t) && (a[0] == 0 || queue(a[0]))
            }
            _ => false,
        }
    }

    /// For a syscall that is not local: the one other processor whose lock
    /// would make it so, if what can be read under this processor's lock
    /// names one. A guess, checked by `is_local` once both locks are held.
    fn partner(&self, s: Syscall, t: Paddr, a: &[u64; 6]) -> Option<usize> {
        let me = self.hal.cpu() as u64;
        let other = |cpu: u64| (cpu != me).then_some(cpu as usize);
        // Of a queue: its owner, or (the queue being this processor's) the
        // home of the thread at its head or of that thread's caller.
        let of_queue = |index: u64| -> Option<usize> {
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
        };
        match s {
            Syscall::Call | Syscall::Send if MsgInfo(a[1]).caps() == 0 => of_queue(a[0] & !NONBLOCK),
            Syscall::Recv => of_queue(a[0]),
            Syscall::ReplyRecv if MsgInfo(a[1]).caps() == 0 => {
                let c = self.thread(t).caller.get();
                let of_caller = if c == 0 {
                    None
                } else {
                    other(self.thread(c).cpu.get())
                        .or_else(|| self.queue(self.thread(c).via.get()).owner.get().checked_sub(1).and_then(other))
                };
                of_caller.or_else(|| if a[0] != 0 { of_queue(a[0]) } else { None })
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

    /// Handle the syscall the current thread just made.
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
            self.go_local();
        }
        self.leave_pair();
    }

    fn dispatch(&self, s: Syscall, t: Paddr, a: [u64; 6]) -> Result<Exit, Error> {
        let grant = |cap: &Cap| if cap.rights.has(Rights::GRANT) { IPC_GRANT } else { 0 };
        match s {
            Syscall::Call => {
                let (slot, ep) = self.ccap(t, a[0], ObjectType::Endpoint, Rights::WRITE)?;
                self.thread(t).via_slot.set(slot);
                self.send(t, ep.obj, ep.badge, IPC_CALL | grant(&ep), true)
            }
            Syscall::Send => {
                let slot = self.cslot(t, a[0] & !NONBLOCK)?;
                let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
                if !cap.rights.has(Rights::WRITE) {
                    return Err(Error::NoRights);
                }
                self.thread(t).via_slot.set(slot);
                match cap.ty {
                    ObjectType::Endpoint => self.send(t, cap.obj, cap.badge, grant(&cap), a[0] & NONBLOCK == 0),
                    // A capability without a badge signals bit 0.
                    ObjectType::Notification => self.notify(t, cap.obj, cap.badge.max(1)),
                    _ => Err(Error::WrongType),
                }
            }
            Syscall::Recv => self.recv_any(t, a[0], a[1], a[2]),
            Syscall::ReplyRecv => {
                let caller = self.thread(t).caller.get();
                let replied = self.reply(t)?;
                if a[0] == 0 {
                    return if replied { Ok(Exit::Done) } else { Err(Error::NoCaller) };
                }
                let r = self.recv_any(t, a[0], 0, 0);
                // The server now waits: the caller it answered runs next.
                let me = self.me();
                if replied
                    && r == Ok(Exit::Blocked)
                    && caller != t
                    && self.thread(caller).state.get() == RUNNABLE
                    && self.thread(caller).cpu.get() as usize == self.hal.cpu()
                {
                    self.q_remove(&me.rq_head, &me.rq_tail, caller);
                    self.hand_off(caller);
                }
                r
            }
            Syscall::Yield => {
                self.me().resched.set(true);
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
                let s = self.cslot(t, a[2])?;
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
                Ok(Exit::Done)
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
                let (fs, _) = self.ccap(t, a[0], ObjectType::Frame, Rights::NONE)?;
                self.unmap_slot(fs);
                Ok(Exit::Done)
            }
            Syscall::ThreadConfigure => {
                let (_, th) = self.ccap(t, a[0], ObjectType::Thread, Rights::WRITE)?;
                let want = [ObjectType::CapSpace, ObjectType::AddressSpace, ObjectType::Endpoint, ObjectType::Frame];
                // Validate everything before changing anything.
                let mut src = [0; 4];
                for i in 0..4 {
                    let arg = a[i + 1];
                    if arg == KEEP || arg == 0 {
                        continue;
                    }
                    let (s, cap) = self.ccap(t, arg, want[i], Rights::NONE)?;
                    if i == 3 && (cap.bits != PAGE_BITS || cap.flags & FLAG_DEVICE != 0 || !cap.rights.has(Rights(3))) {
                        return Err(Error::BadArg);
                    }
                    src[i] = s;
                }
                // The thread may be running on another processor with the
                // CapSpace and address space it is about to lose.
                let home = self.thread(th.obj).cpu.get() as usize;
                self.cpus[home].dirty.set(true);
                if home != self.hal.cpu() && self.cpus[home].cur.get() == th.obj {
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
                        // Move the thread to processor `a2`. It is stopped
                        // first and stays stopped.
                        if a[2] >= self.ncpu.get() as u64 {
                            return Err(Error::BadArg);
                        }
                        if th.obj == t {
                            self.thread(t).regs[H::SYS].set(0);
                        }
                        self.cancel(th.obj, true);
                        let old = self.home(th.obj);
                        if old.loaded.get() == th.obj {
                            old.loaded.set(0);
                        }
                        if old.cur.get() == th.obj {
                            old.cur.set(0);
                        }
                        self.thread(th.obj).cpu.set(a[2]);
                        Ok(if th.obj == t { Exit::Blocked } else { Exit::Done })
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
                // Whatever the line signalled before no longer hears from it.
                let old = self.irqs[line].0.replace(0);
                self.hal.irq_mask(line, true);
                if old != 0 {
                    let q = self.queue(old);
                    q.bound.set(q.bound.get() - 1);
                }
                if to != 0 {
                    if !self.hal.irq_route(line, self.hal.cpu(), a[3] & silica_abi::IRQ_LEVEL != 0, a[3] & silica_abi::IRQ_ACTIVE_LOW != 0)
                    {
                        return Err(Error::Unsupported);
                    }
                    self.irqs[line].0.set(to);
                    self.irqs[line].1.set(badge);
                    let q = self.queue(to);
                    q.bound.set(q.bound.get() + 1);
                }
                Ok(Exit::Done)
            }
            Syscall::Inspect => {
                let slot = self.cslot(t, a[0])?;
                let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
                let r = &self.thread(t).regs;
                let thread = |obj: Paddr| {
                    let th = self.thread(obj);
                    [th.state.get(), th.cpu.get(), th.regs[H::IP].get(), th.regs[H::SP].get(), th.blocked_on.get(), 0]
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
                            Some(v) => [v.cap.ty as u64, v.cap.rights.0 as u64, v.cap.bits as u64, v.cap.obj, v.cap.badge, v.depth as u64],
                            None => [0; 6],
                        }
                    }
                    ObjectType::Thread if a[1] == 0 && a[2] == 0 => thread(cap.obj),
                    ObjectType::CapSpace | ObjectType::Thread => return Err(Error::BadArg),
                    _ => return Err(Error::WrongType),
                };
                for (i, v) in out.iter().enumerate() {
                    r[H::ARG[i]].set(*v);
                }
                Ok(Exit::Done)
            }
            Syscall::Console => {
                let (_, cap) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if cap.flags & FLAG_CONSOLE == 0 {
                    return Err(Error::WrongType);
                }
                let n = a[1] as usize;
                if n > 32 {
                    return Err(Error::BadArg);
                }
                let mut bytes = [0u8; 32];
                for (i, w) in a[2..6].iter().enumerate() {
                    bytes[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
                }
                self.hal.console(&bytes[..n]);
                Ok(Exit::Done)
            }
            Syscall::PowerOff => {
                let (_, cap) = self.ccap(t, a[0], ObjectType::Hw, Rights::WRITE)?;
                if cap.flags & FLAG_POWER == 0 {
                    return Err(Error::WrongType);
                }
                self.hal.power_off(a[1]);
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

    fn recv_any(&self, t: Paddr, index: u64, deadline: u64, park: u64) -> Result<Exit, Error> {
        let slot = self.cslot(t, index)?;
        let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
        if !cap.rights.has(Rights::READ) {
            return Err(Error::NoRights);
        }
        self.thread(t).via_slot.set(slot);
        match cap.ty {
            ObjectType::Endpoint => self.recv(t, cap.obj, deadline, park),
            ObjectType::Notification => self.wait(t, cap.obj, deadline),
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
        Some(SlotView { cap, depth: slot.depth(), prev: slot.prev.get(), next: slot.next.get(), mapped, watermark })
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

    /// Whether every processor's timer list is well formed: linked both
    /// ways, in deadline order, ending, and holding only threads of that
    /// processor that have a deadline. For tests and fuzzing.
    pub fn check_timers(&self) -> Result<(), &'static str> {
        for (i, cpu) in self.cpus.iter().enumerate().take(self.ncpu.get()) {
            let (mut t, mut prev, mut last, mut n) = (cpu.timers.get(), 0, 0, 0usize);
            while t != 0 {
                let th = self.thread(t);
                if th.tprev.get() != prev {
                    return Err("a timer list's back link is wrong");
                }
                if th.cpu.get() as usize != i {
                    return Err("a thread waits on the timer list of a processor that is not its home");
                }
                let d = th.deadline.get();
                if d == 0 || d < last {
                    return Err("a timer list holds a thread with no deadline, or is out of order");
                }
                n += 1;
                if n > 1 << 20 {
                    return Err("a timer list does not end");
                }
                (prev, last, t) = (t, d, th.tnext.get());
            }
        }
        Ok(())
    }

    /// Check the waiting lists of the endpoint or notification at `q`
    /// (from a capability to it) for consistency: every thread queued
    /// there waits on it, every delivered call is linked to it, and each
    /// of them came through a slot that still holds a capability to it.
    /// For tests and fuzzing; says what is wrong.
    pub fn check_queue(&self, q: Paddr) -> Result<(), &'static str> {
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
        if qu.pend_tail.get() != prev {
            return Err("the list of calls in progress has the wrong tail");
        }
        Ok(())
    }

    /// Whether any thread of this processor can run now or will be woken by a deadline.
    pub fn can_run(&self) -> bool {
        let me = self.me();
        let c = me.cur.get();
        (c != 0 && self.thread(c).state.get() == RUNNABLE) || me.rq_head.get() != 0 || me.direct.get() != 0 || me.timers.get() != 0
    }
}
