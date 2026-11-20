//! Synchronous IPC on Endpoints, signals on Notifications, and fault delivery.

use crate::hal::Hal;
use crate::kernel::{Exit, Kernel};
use crate::mem::Paddr;
use crate::thread::{INACTIVE, IPC_CALL, IPC_FAULT, RECEIVING, REPLY, SENDING};
use silica_abi::{Error, FAULT_LABEL, FAULT_RESUME, Fault, MsgInfo, ObjectType};

const SENDERS: u64 = 1;
const RECEIVERS: u64 = 2;

impl<H: Hal> Kernel<H> {
    fn ipc_buffer(&self, t: Paddr) -> Option<Paddr> {
        self.thread(t).ipcbuf.cap().map(|c| c.obj)
    }

    /// Copy the pending message of `from` into `to`'s registers and buffer.
    fn transfer(&self, from: Paddr, to: Paddr, badge: u64) {
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
        if words > MsgInfo::REG_WORDS {
            match (self.ipc_buffer(from), self.ipc_buffer(to)) {
                (Some(src), Some(dst)) => {
                    for i in MsgInfo::REG_WORDS..words {
                        let o = (i * 8) as u64;
                        self.phys.word(dst + o).set(self.phys.word(src + o).get());
                    }
                }
                _ => words = MsgInfo::REG_WORDS,
            }
        }
        t.regs[H::ARG[1]].set(MsgInfo::new(info.label(), words).0);
    }

    /// `server` has received `caller`'s call and now owes it a reply.
    fn await_reply(&self, caller: Paddr, server: Paddr) {
        self.drop_caller(server);
        self.thread(server).caller.set(caller);
        // Keeps the IPC_FAULT flag: the reply path needs it.
        self.block(caller, REPLY, server, 0);
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
            self.transfer(t, r, badge);
            if flags & IPC_CALL != 0 {
                // The caller waits for the receiver, so the receiver runs now.
                self.thread(r).regs[H::SYS].set(0);
                self.hand_off(r);
                self.await_reply(t, r);
                return Ok(Exit::Blocked);
            }
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
    pub(crate) fn recv(&self, t: Paddr, ep: Paddr, deadline: u64) -> Result<Exit, Error> {
        let q = self.queue(ep);
        if q.kind.get() == SENDERS {
            let s = self.q_pop(&q.head, &q.tail);
            if q.head.get() == 0 {
                q.kind.set(0);
            }
            let st = self.thread(s);
            self.transfer(s, t, st.ipc_badge.get());
            if st.ipc_flags.get() & IPC_CALL != 0 {
                self.await_reply(s, t);
            } else {
                st.ipc_flags.set(0);
                self.wake(s, Ok(()));
            }
            return Ok(Exit::Done);
        }
        self.wait_on(t, ep, deadline)
    }

    fn wait_on(&self, t: Paddr, q: Paddr, deadline: u64) -> Result<Exit, Error> {
        if deadline == 1 {
            return Err(Error::WouldBlock);
        }
        if deadline != 0 && deadline <= self.hal.now() {
            return Err(Error::Timeout);
        }
        let qu = self.queue(q);
        qu.kind.set(RECEIVERS);
        self.q_push(&qu.head, &qu.tail, t);
        self.block(t, RECEIVING, q, deadline);
        Ok(Exit::Blocked)
    }

    /// OR `badge` into notification `n`, waking one waiter.
    pub fn signal(&self, n: Paddr, badge: u64) {
        let q = self.queue(n);
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
        let word = q.word.replace(0);
        if word != 0 {
            let th = self.thread(t);
            th.regs[H::ARG[0]].set(word);
            th.regs[H::ARG[1]].set(0);
            return Ok(Exit::Done);
        }
        self.wait_on(t, n, deadline)
    }

    /// Answer the call `t` received last. Returns whether there was a caller.
    pub(crate) fn reply(&self, t: Paddr) -> bool {
        let th = self.thread(t);
        let c = th.caller.replace(0);
        if c == 0 {
            return false;
        }
        let ct = self.thread(c);
        if ct.ipc_flags.replace(0) & IPC_FAULT != 0 {
            if th.regs[H::ARG[2]].get() == FAULT_RESUME {
                self.make_runnable(c);
            } else {
                ct.blocked_on.set(0);
                ct.state.set(INACTIVE);
            }
            return true;
        }
        self.transfer(t, c, 0);
        self.wake(c, Ok(()));
        true
    }

    /// The current thread took a fault it cannot continue from. The fault is
    /// sent to its fault endpoint as a call; with no endpoint the thread stops.
    pub fn fault(&self, kind: Fault, addr: u64, code: u64) {
        let t = self.cur.get();
        let th = self.thread(t);
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
    }
}
