//! Synchronous IPC on Endpoints, signals on Notifications, and fault delivery.

use crate::hal::Hal;
use crate::kernel::{Exit, Kernel};
use crate::mem::Paddr;
use crate::thread::{INACTIVE, IPC_CALL, IPC_FAULT, IPC_GRANT, RECEIVING, REPLY, SENDING};
use silica_abi::{Error, FAULT_LABEL, FAULT_MAP, FAULT_RESUME, Fault, MsgInfo, ObjectType, Rights};

const SENDERS: u64 = 1;
const RECEIVERS: u64 = 2;

/// The queue holds one call waiting for its answer (a parked caller).
const PARKED: u64 = 3;

impl<H: Hal> Kernel<H> {
    fn ipc_buffer(&self, t: Paddr) -> Option<Paddr> {
        self.thread(t).ipcbuf.cap().map(|c| c.obj)
    }

    /// Copy the pending message of `from` into `to`'s registers and buffer,
    /// and with `grant` pass the capabilities it names.
    fn transfer(&self, from: Paddr, to: Paddr, badge: u64, grant: bool) {
        let (f, t) = (self.thread(from), self.thread(to));
        t.regs[H::ARG[0]].set(badge);
        if f.ipc_flags.get() & IPC_FAULT != 0 {
            t.regs[H::ARG[1]].set(MsgInfo::new(FAULT_LABEL + f.fault[0].get(), 4).0);
            t.regs[H::ARG[2]].set(f.regs[H::IP].get());
            t.regs[H::ARG[3]].set(f.fault[1].get());
            t.regs[H::ARG[4]].set(f.fault[2].get());
            t.regs[H::ARG[5]].set(f.regs[H::SP].get());
            return;
        }
        let info = MsgInfo(f.regs[H::ARG[1]].get());
        let mut words = info.words();
        for i in 2..6 {
            t.regs[H::ARG[i]].set(f.regs[H::ARG[i]].get());
        }
        let buffers = match (self.ipc_buffer(from), self.ipc_buffer(to)) {
            (Some(src), Some(dst)) => Some((src, dst)),
            _ => None,
        };
        if words > MsgInfo::REG_WORDS {
            match buffers {
                Some((src, dst)) => {
                    for i in MsgInfo::REG_WORDS..words {
                        let o = (i * 8) as u64;
                        self.phys.word(dst + o).set(self.phys.word(src + o).get());
                    }
                }
                None => words = MsgInfo::REG_WORDS,
            }
        }
        // Capabilities: each one the sender lists goes into the slot the
        // receiver listed at the same position. The first that cannot be
        // passed ends the transfer; the receiver is told how many arrived.
        let mut passed = 0;
        if grant
            && info.caps() > 0
            && let Some((src, dst)) = buffers
        {
            for i in 0..info.caps() {
                let o = (i * 8) as u64;
                let named = self.phys.word(src + silica_abi::IPC_CAPS as u64 + o).get();
                let into = self.phys.word(dst + silica_abi::IPC_RECV as u64 + o).get();
                let (Ok(s), Ok(d)) = (self.cslot(from, named & silica_abi::IPC_CAP_SLOT), self.cslot(to, into)) else { break };
                let ok = match named >> 60 {
                    silica_abi::IPC_CAP_DERIVE => self.copy(s, d, Rights::ALL, 0, false).is_ok(),
                    silica_abi::IPC_CAP_DUPLICATE => self.copy(s, d, Rights::ALL, 0, true).is_ok(),
                    silica_abi::IPC_CAP_MOVE if !self.slot(s).is_empty() && self.slot(d).is_empty() => {
                        self.move_slot(s, d);
                        true
                    }
                    _ => false,
                };
                if !ok {
                    break;
                }
                passed += 1;
            }
        }
        t.regs[H::ARG[1]].set(MsgInfo::with_caps(info.label(), words, passed).0);
    }

    /// `server` has received `caller`'s call through endpoint `ep` and owes
    /// it an answer. If the server asked for the call to be parked on a
    /// notification (the `park` argument of `Recv`), the caller waits there
    /// and any thread holding that notification can answer by sending to it;
    /// otherwise the caller waits for the server's `ReplyRecv`.
    fn await_reply(&self, caller: Paddr, server: Paddr, ep: Paddr) {
        let park = self.thread(server).park.replace(0);
        if park != 0
            && let Some(n) = self.cslot(server, park).ok().and_then(|s| self.slot(s).cap())
            && n.ty == ObjectType::Notification
        {
            let q = self.queue(n.obj);
            let cpu = self.thread(server).cpu.get() + 1;
            // A notification nobody waits on and nobody owns yet, or one
            // this processor already owns.
            if q.head.get() == 0 && q.word.get() == 0 && (q.owner.claim(cpu) || self.global.get()) {
                q.kind.set(PARKED);
                self.q_push(&q.head, &q.tail, caller);
                self.thread(caller).via.set(0);
                self.block(caller, REPLY, n.obj, 0);
                return;
            }
        }
        self.drop_caller(server);
        self.thread(server).caller.set(caller);
        // Linked on the endpoint, so destroying the endpoint finds the call.
        let q = self.queue(ep);
        self.q_push(&q.pend_head, &q.pend_tail, caller);
        self.thread(caller).via.set(ep);
        // Keeps the IPC_FAULT flag: the reply path needs it.
        self.block(caller, REPLY, server, 0);
    }

    /// Take `caller` (waiting for a server's reply) off its endpoint's list
    /// of calls in progress.
    pub(crate) fn unpend(&self, caller: Paddr) {
        let ep = self.thread(caller).via.replace(0);
        if ep != 0 {
            let q = self.queue(ep);
            self.q_remove(&q.pend_head, &q.pend_tail, caller);
        }
    }

    /// Send the current thread's message to endpoint `ep`.
    pub(crate) fn send(&self, t: Paddr, ep: Paddr, badge: u64, flags: u64, blocking: bool) -> Result<Exit, Error> {
        let q = self.queue(ep);
        let th = self.thread(t);
        th.ipc_flags.set(flags);
        if q.kind.get() == RECEIVERS {
            let r = self.q_pop(&q.head, &q.tail);
            if q.head.get() == 0 {
                q.kind.set(0);
            }
            self.transfer(t, r, badge, flags & IPC_GRANT != 0);
            if flags & IPC_CALL != 0 {
                // The caller waits for the receiver, so the receiver runs now.
                self.thread(r).regs[H::SYS].set(0);
                self.hand_off(r);
                self.await_reply(t, r, ep);
                return Ok(Exit::Blocked);
            }
            self.thread(r).park.set(0);
            self.wake(r, Ok(()));
            th.ipc_flags.set(0);
            return Ok(Exit::Done);
        }
        if !blocking {
            th.ipc_flags.set(0);
            return Err(Error::WouldBlock);
        }
        th.ipc_badge.set(badge);
        q.kind.set(SENDERS);
        self.q_push(&q.head, &q.tail, t);
        self.block(t, SENDING, ep, 0);
        Ok(Exit::Blocked)
    }

    /// Receive on endpoint `ep`. `deadline`: 0 for ever, 1 do not block.
    /// `park`: a notification of the receiver's on which the next call is
    /// parked (0: none).
    pub(crate) fn recv(&self, t: Paddr, ep: Paddr, deadline: u64, park: u64) -> Result<Exit, Error> {
        let q = self.queue(ep);
        self.thread(t).park.set(park);
        if q.kind.get() == SENDERS {
            let s = self.q_pop(&q.head, &q.tail);
            if q.head.get() == 0 {
                q.kind.set(0);
            }
            let st = self.thread(s);
            self.transfer(s, t, st.ipc_badge.get(), st.ipc_flags.get() & IPC_GRANT != 0);
            if st.ipc_flags.get() & IPC_CALL != 0 {
                self.await_reply(s, t, ep);
            } else {
                st.ipc_flags.set(0);
                self.wake(s, Ok(()));
            }
            self.thread(t).park.set(0);
            return Ok(Exit::Done);
        }
        let r = self.wait_on(t, ep, deadline);
        if r.is_err() {
            self.thread(t).park.set(0);
        }
        r
    }

    fn wait_on(&self, t: Paddr, q: Paddr, deadline: u64) -> Result<Exit, Error> {
        if deadline == 1 {
            return Err(Error::WouldBlock);
        }
        if deadline != 0 && deadline <= self.hal.now() {
            return Err(Error::Timeout);
        }
        let qu = self.queue(q);
        // The queue belongs to the processor of whoever waits to receive
        // on it; only a holder of every lock may move it.
        if self.global.get() {
            qu.owner.set(self.thread(t).cpu.get() + 1);
        }
        qu.kind.set(RECEIVERS);
        self.q_push(&qu.head, &qu.tail, t);
        self.block(t, RECEIVING, q, deadline);
        Ok(Exit::Blocked)
    }

    /// A message sent to notification `n`: if a call is parked there, this
    /// is its answer; otherwise OR `badge` into the pending word, waking one waiter.
    pub(crate) fn notify(&self, t: Paddr, n: Paddr, badge: u64) -> Result<Exit, Error> {
        let q = self.queue(n);
        if q.kind.get() != PARKED {
            self.signal(n, badge);
            return Ok(Exit::Done);
        }
        let c = q.head.get();
        self.answer(t, c)?;
        Ok(Exit::Done)
    }

    /// Interrupt line `line` fired on this processor (which holds its
    /// lock): signal what the line is bound to. The architecture has
    /// already held a level-triggered line off until it is acknowledged.
    pub fn irq(&self, line: usize) {
        let Some(irq) = self.irqs.get(line) else { return };
        let n = irq.0.get();
        if n == 0 {
            // Nothing listens (it was unbound as the interrupt arrived).
            self.hal.irq_mask(line, true);
            return;
        }
        let q = self.queue(n);
        let me = self.hal.cpu() as u64;
        let local = self.ncpu.get() == 1 || (q.owner.get() == me + 1 && (q.head.get() == 0 || self.thread(q.head.get()).cpu.get() == me));
        if local {
            self.signal(n, irq.1.get());
            return;
        }
        // The waiter is another processor's: take every lock, and look
        // again at what the line is bound to now.
        self.go_global();
        let n = irq.0.get();
        if n != 0 {
            self.signal(n, irq.1.get());
        }
        self.go_local();
    }

    /// OR `badge` into notification `n`, waking one waiter.
    pub fn signal(&self, n: Paddr, badge: u64) {
        let q = self.queue(n);
        if q.kind.get() == PARKED {
            // A parked call is answered with a message, not woken by a signal.
            q.word.set(q.word.get() | badge);
            return;
        }
        let w = self.q_pop(&q.head, &q.tail);
        if w == 0 {
            q.word.set(q.word.get() | badge);
            return;
        }
        if q.head.get() == 0 {
            q.kind.set(0);
        }
        let wt = self.thread(w);
        wt.regs[H::ARG[0]].set(badge);
        wt.regs[H::ARG[1]].set(0);
        self.wake(w, Ok(()));
    }

    /// Take the pending word of notification `n`, or wait for a signal.
    pub(crate) fn wait(&self, t: Paddr, n: Paddr, deadline: u64) -> Result<Exit, Error> {
        let q = self.queue(n);
        if q.kind.get() == PARKED {
            // A call is parked here; it cannot also be waited on.
            return Err(Error::InUse);
        }
        let word = q.word.replace(0);
        if word != 0 {
            let th = self.thread(t);
            th.regs[H::ARG[0]].set(word);
            th.regs[H::ARG[1]].set(0);
            return Ok(Exit::Done);
        }
        self.wait_on(t, n, deadline)
    }

    /// Deliver `t`'s message as the answer to the call of `c`, which waits
    /// on `t` or is parked on a notification.
    ///
    /// The answer to a fault can carry the cure: with word 0 `FAULT_MAP` the
    /// kernel maps a frame (words 1 to 3: address space, frame, address with
    /// rights in its low bits; capabilities of the answering thread) and
    /// then resumes the thread. If the mapping fails the error goes to the
    /// answering thread and the call stays unanswered.
    fn answer(&self, t: Paddr, c: Paddr) -> Result<(), Error> {
        let th = self.thread(t);
        let ct = self.thread(c);
        let fault = ct.ipc_flags.get() & IPC_FAULT != 0;
        let word = |i: usize| th.regs[H::ARG[2 + i]].get();
        if fault && word(0) == FAULT_MAP {
            let (_, asp) = self.ccap(t, word(1), ObjectType::AddressSpace, Rights::WRITE)?;
            let (fs, _) = self.ccap(t, word(2), ObjectType::Frame, Rights::NONE)?;
            self.map(asp.obj, fs, word(3) & !0xFFF, Rights(word(3) as u8 & 0xF))?;
        }
        // Off whatever it waits on: the endpoint's list of calls in
        // progress, or the notification it is parked on.
        if ct.via.get() != 0 {
            self.unpend(c);
            self.thread(ct.blocked_on.get()).caller.set(0);
        } else {
            let q = self.queue(ct.blocked_on.get());
            self.q_remove(&q.head, &q.tail, c);
            q.kind.set(0);
        }
        ct.ipc_flags.set(0);
        if fault {
            if word(0) == FAULT_RESUME || word(0) == FAULT_MAP {
                self.make_runnable(c);
            } else {
                ct.blocked_on.set(0);
                ct.state.set(INACTIVE);
                self.home(c).dirty.set(true);
            }
            return Ok(());
        }
        // An answer may always carry capabilities: the caller chose to call.
        self.transfer(t, c, 0, true);
        self.wake(c, Ok(()));
        Ok(())
    }

    /// Answer the call `t` received last. Returns whether there was one.
    pub(crate) fn reply(&self, t: Paddr) -> Result<bool, Error> {
        let c = self.thread(t).caller.get();
        if c == 0 {
            return Ok(false);
        }
        self.answer(t, c)?;
        Ok(true)
    }

    /// The current thread took a fault it cannot continue from. The fault is
    /// sent to its fault endpoint as a call; with no endpoint the thread stops.
    pub fn fault(&self, kind: Fault, addr: u64, code: u64) {
        let me = self.me();
        let t = me.cur.get();
        // The thread may have been stopped or destroyed by another
        // processor while this one waited for its lock.
        if t == 0 || self.thread(t).state.get() != crate::thread::RUNNABLE {
            me.dirty.set(true);
            return;
        }
        let th = self.thread(t);
        let global = self.ncpu.get() > 1 && !th.fault_ep.cap().is_some_and(|c| self.queue_is_local(t, &c));
        if global {
            self.go_global();
            if me.cur.get() != t || th.state.get() != crate::thread::RUNNABLE {
                me.dirty.set(true);
                self.go_local();
                return;
            }
        }
        th.fault[0].set(kind as u64);
        th.fault[1].set(addr);
        th.fault[2].set(code);
        match th.fault_ep.cap() {
            Some(c) if c.ty == ObjectType::Endpoint => {
                let _ = self.send(t, c.obj, c.badge, IPC_CALL | IPC_FAULT, true);
            }
            _ => {
                self.cancel(t, false);
                crate::klog!(
                    self,
                    "thread {:#x}: {:?} fault at ip {:#x}, address {:#x}, code {:#x}; no fault endpoint, thread stopped",
                    t,
                    kind,
                    th.regs[H::IP].get(),
                    addr,
                    code
                );
            }
        }
        if global {
            self.go_local();
        }
    }
}
