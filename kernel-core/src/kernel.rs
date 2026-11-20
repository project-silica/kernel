//! The kernel state for one processor and the syscall dispatcher.

use crate::cap::{Cap, FLAG_DEVICE, FLAG_PORTS};
use crate::hal::Hal;
use crate::mem::{Paddr, Phys};
use crate::thread::{INACTIVE, IPC_CALL, Thread};
use core::cell::Cell;
use silica_abi::{Error, KEEP, NONBLOCK, ObjectType, PAGE_BITS, Rights, Syscall};

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

/// The kernel. One per machine at M1 (one processor runs kernel code).
pub struct Kernel<H: Hal> {
    /// The architecture.
    pub hal: H,
    /// Physical memory.
    pub phys: Phys,
    pub(crate) cur: Cell<Paddr>,
    pub(crate) loaded: Cell<Paddr>,
    pub(crate) loaded_root: Cell<Paddr>,
    pub(crate) rq_head: Cell<Paddr>,
    pub(crate) rq_tail: Cell<Paddr>,
    pub(crate) timers: Cell<Paddr>,
    pub(crate) timer_at: Cell<u64>,
    pub(crate) slice_end: Cell<u64>,
    pub(crate) resched: Cell<bool>,
    pub(crate) reap: Cell<u64>,
}

impl<H: Hal> Kernel<H> {
    /// A kernel with no threads.
    pub const fn new(hal: H, phys: Phys) -> Kernel<H> {
        Kernel {
            hal,
            phys,
            cur: Cell::new(0),
            loaded: Cell::new(0),
            loaded_root: Cell::new(0),
            rq_head: Cell::new(0),
            rq_tail: Cell::new(0),
            timers: Cell::new(0),
            timer_at: Cell::new(u64::MAX),
            slice_end: Cell::new(0),
            resched: Cell::new(false),
            reap: Cell::new(0),
        }
    }

    /// The thread this processor is running (0 while idle).
    pub fn current(&self) -> Paddr {
        self.cur.get()
    }

    /// The saved registers of thread `t`.
    pub fn regs(&self, t: Paddr) -> &[Cell<u64>; crate::hal::NREGS] {
        &self.thread(t).regs
    }

    /// The slot `index` of the current thread's CapSpace.
    fn cslot(&self, t: Paddr, index: u64) -> Result<Paddr, Error> {
        let cs = self.thread(t).cspace.cap().ok_or(Error::BadCap)?;
        self.slot_in(&cs, index)
    }

    /// The capability at `index`, which must be of type `ty` and carry `rights`.
    fn ccap(
        &self,
        t: Paddr,
        index: u64,
        ty: ObjectType,
        rights: Rights,
    ) -> Result<(Paddr, Cap), Error> {
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

    /// Handle the syscall the current thread just made.
    pub fn syscall(&self) {
        let t = self.cur.get();
        let r = &self.thread(t).regs;
        let a = [0, 1, 2, 3, 4, 5].map(|i| r[H::ARG[i]].get());
        let res = match Syscall::from_u64(r[H::SYS].get()) {
            Some(s) => self.dispatch(s, t, a),
            None => Err(Error::BadSyscall),
        };
        match res {
            Ok(Exit::Done) => r[H::SYS].set(0),
            Ok(Exit::Blocked) => {}
            Ok(Exit::Preempted) => r[H::IP].set(r[H::IP].get().wrapping_sub(H::SYSCALL_LEN)),
            Err(e) => r[H::SYS].set(e as u64),
        }
    }

    fn dispatch(&self, s: Syscall, t: Paddr, a: [u64; 6]) -> Result<Exit, Error> {
        match s {
            Syscall::Call => {
                let (_, ep) = self.ccap(t, a[0], ObjectType::Endpoint, Rights::WRITE)?;
                self.send(t, ep.obj, ep.badge, IPC_CALL, true)
            }
            Syscall::Send => {
                let slot = self.cslot(t, a[0] & !NONBLOCK)?;
                let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
                if !cap.rights.has(Rights::WRITE) {
                    return Err(Error::NoRights);
                }
                match cap.ty {
                    ObjectType::Endpoint => {
                        self.send(t, cap.obj, cap.badge, 0, a[0] & NONBLOCK == 0)
                    }
                    ObjectType::Notification => {
                        // A capability without a badge signals bit 0.
                        self.signal(cap.obj, cap.badge.max(1));
                        Ok(Exit::Done)
                    }
                    _ => Err(Error::WrongType),
                }
            }
            Syscall::Recv => self.recv_any(t, a[0], a[1]),
            Syscall::ReplyRecv => {
                let replied = self.reply(t);
                if a[0] != 0 {
                    self.recv_any(t, a[0], 0)
                } else if replied {
                    Ok(Exit::Done)
                } else {
                    Err(Error::NoCaller)
                }
            }
            Syscall::Yield => {
                self.resched.set(true);
                Ok(Exit::Done)
            }
            Syscall::Retype => {
                let us = self.cslot(t, a[0])?;
                let ty = ObjectType::from_u64(a[1]).ok_or(Error::BadArg)?;
                let (_, dst) = self.ccap(t, a[3], ObjectType::CapSpace, Rights::WRITE)?;
                self.retype(us, ty, a[2], &dst, a[4], a[5])
            }
            Syscall::CapCopy => {
                let (_, dst) = self.ccap(t, a[0], ObjectType::CapSpace, Rights::WRITE)?;
                let d = self.slot_in(&dst, a[1])?;
                let s = self.cslot(t, a[2])?;
                self.copy(s, d, Rights(a[3] as u8), a[4])?;
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
                if !self.revoke_if_untyped(s) {
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
                Ok(if self.revoke(s) {
                    Exit::Done
                } else {
                    Exit::Preempted
                })
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
                let want = [
                    ObjectType::CapSpace,
                    ObjectType::AddressSpace,
                    ObjectType::Endpoint,
                    ObjectType::Frame,
                ];
                // Validate everything before changing anything.
                let mut src = [0; 4];
                for i in 0..4 {
                    let arg = a[i + 1];
                    if arg == KEEP || arg == 0 {
                        continue;
                    }
                    let (s, cap) = self.ccap(t, arg, want[i], Rights::NONE)?;
                    if i == 3
                        && (cap.bits != PAGE_BITS
                            || cap.flags & FLAG_DEVICE != 0
                            || !cap.rights.has(Rights(3)))
                    {
                        return Err(Error::BadArg);
                    }
                    if self.slot(s).depth() == u16::MAX {
                        return Err(Error::TooDeep);
                    }
                    src[i] = s;
                }
                let slots = Thread::embedded_slots(th.obj);
                for i in 0..4 {
                    if a[i + 1] == KEEP {
                        continue;
                    }
                    self.delete(slots[i]);
                    if src[i] != 0 {
                        // The source may have been deleted by the line above
                        // when it was the thread's own embedded capability.
                        if let Some(mut cap) = self.slot(src[i]).cap() {
                            if cap.ty == ObjectType::Frame {
                                cap.badge = 0;
                            }
                            self.insert_child(src[i], slots[i], &cap)?;
                        }
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
                Ok(if th.obj == t {
                    Exit::Blocked
                } else {
                    Exit::Done
                })
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
                        Ok(if th.obj == t {
                            Exit::Blocked
                        } else {
                            Exit::Done
                        })
                    }
                    1 => {
                        if self.thread(th.obj).state.get() == INACTIVE {
                            self.make_runnable(th.obj);
                        }
                        Ok(Exit::Done)
                    }
                    _ => Err(Error::BadArg),
                }
            }
            Syscall::PortIo => {
                let (width, write) = (a[2] & 0xFF, a[2] & 0x100 != 0);
                let need = if write { Rights::WRITE } else { Rights::READ };
                let (_, cap) = self.ccap(t, a[0], ObjectType::Hw, need)?;
                if cap.flags & FLAG_PORTS == 0 {
                    return Err(Error::WrongType);
                }
                if !matches!(width, 1 | 2 | 4)
                    || a[1] < cap.obj
                    || a[1].checked_add(width).is_none_or(|e| e > cap.badge)
                {
                    return Err(Error::BadArg);
                }
                let v = self
                    .hal
                    .port_io(a[1] as u16, width as u8, write, a[3] as u32);
                self.thread(t).regs[H::ARG[0]].set(v as u64);
                Ok(Exit::Done)
            }
        }
    }

    fn recv_any(&self, t: Paddr, index: u64, deadline: u64) -> Result<Exit, Error> {
        let slot = self.cslot(t, index)?;
        let cap = self.slot(slot).cap().ok_or(Error::BadCap)?;
        if !cap.rights.has(Rights::READ) {
            return Err(Error::NoRights);
        }
        match cap.ty {
            ObjectType::Endpoint => self.recv(t, cap.obj, deadline),
            ObjectType::Notification => self.wait(t, cap.obj, deadline),
            _ => Err(Error::WrongType),
        }
    }

    fn revoke_if_untyped(&self, s: Paddr) -> bool {
        match self.slot(s).cap() {
            Some(c) if c.ty == ObjectType::Untyped => self.revoke(s),
            _ => true,
        }
    }

    /// Derive a capability from slot `s` into the empty slot `d`.
    pub(crate) fn copy(&self, s: Paddr, d: Paddr, mask: Rights, badge: u64) -> Result<(), Error> {
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
        let mapped = (cap.ty == ObjectType::Frame && slot.badge.get() & 1 != 0)
            .then(|| (slot.aux.get(), slot.badge.get() & !0xFFF));
        let watermark = if cap.ty == ObjectType::Untyped {
            (slot.badge.get(), slot.aux.get())
        } else {
            (0, 0)
        };
        Some(SlotView {
            cap,
            depth: slot.depth(),
            prev: slot.prev.get(),
            next: slot.next.get(),
            mapped,
            watermark,
        })
    }

    /// The state of the thread object at `t` (from a Thread capability).
    pub fn thread_view(&self, t: Paddr) -> ThreadView {
        let th = self.thread(t);
        ThreadView {
            state: th.state.get(),
            slots: Thread::embedded_slots(t),
            blocked_on: th.blocked_on.get(),
            caller: th.caller.get(),
        }
    }

    /// Whether any thread can run now or will be woken by a deadline.
    pub fn can_run(&self) -> bool {
        let c = self.cur.get();
        (c != 0 && self.thread(c).state.get() == crate::thread::RUNNABLE)
            || self.rq_head.get() != 0
            || self.timers.get() != 0
    }
}
