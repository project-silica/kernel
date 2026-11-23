//! Synchronous IPC on Endpoints, signals on Notifications, and fault delivery.

use crate::hal::Hal;
use crate::kernel::{Exit, Kernel};
use crate::mem::Paddr;
use crate::thread::{INACTIVE, IPC_CALL, IPC_FAULT, IPC_GRANT, IPC_PARKED, RECEIVING, REPLY, SENDING};
use silica_abi::{Error, FAULT_LABEL, FAULT_MAP, FAULT_RESUME, Fault, LISTEN_SHIFT, MsgInfo, ObjectType, Rights};

const SENDERS: u64 = 1;
pub(crate) const RECEIVERS: u64 = 2;

/// The queue holds one call waiting for its answer (a parked caller).
pub(crate) const PARKED: u64 = 3;

/// The notification slot a receive listens on, from its first argument.
pub(crate) fn listen_slot(a0: u64) -> u64 {
    a0 >> LISTEN_SHIFT & 0x7FFF_FFFF
}

/// The endpoint or notification slot of a receive, from its first argument.
pub(crate) fn recv_slot(a0: u64) -> u64 {
    a0 & 0xFFFF_FFFF
}

impl<H: Hal> Kernel<H> {
    /// The thread that is listening on notification `n` right now: waiting
    /// to receive on an endpoint, having named `n` to listen on as well
    /// (0: none). A notification keeps naming its last listener after that
    /// receive has ended, so that a server's loop registers once; what
    /// makes the name count is the thread's own state, read here under its
    /// home processor's lock (which the caller holds).
    pub(crate) fn listener(&self, n: Paddr) -> Paddr {
        let l = self.queue(n).listener().get();
        if l != 0 {
            let lt = self.thread(l);
            if lt.listen.get() == n && lt.state.get() == RECEIVING && lt.blocked_on.get() != n && listen_slot(lt.regs[H::ARG[0]].get()) != 0
            {
                return l;
            }
        }
        0
    }

    /// End the receive of `l`, which listens on a notification: off its
    /// endpoint's queue, with the notification's `word` as what it received.
    fn wake_listener(&self, l: Paddr, word: u64, status: Result<(), Error>) {
        let lt = self.thread(l);
        let eq = self.queue(lt.blocked_on.get());
        self.q_remove(&eq.head, &eq.tail, l);
        if eq.head.get() == 0 {
            eq.kind.set(0);
        }
        lt.park.set(0);
        lt.regs[H::ARG[0]].set(word);
        lt.regs[H::ARG[1]].set(MsgInfo::NOTIFIED.0);
        self.wake(l, status);
    }

    /// The capability in `slot` to notification `n` is leaving its slot
    /// (`slot` 0: the notification itself is going): a thread listening
    /// through it stops waiting, with `Gone`, and the notification names no
    /// listener. The caller holds every lock.
    pub(crate) fn unlisten(&self, n: Paddr, slot: Paddr) {
        let q = self.queue(n);
        let l = q.listener().get();
        if l == 0 {
            return;
        }
        let lt = self.thread(l);
        if slot != 0 && lt.listen_slot.get() != slot {
            return;
        }
        if self.listener(n) != 0 {
            self.wake_listener(l, 0, Err(Error::Gone));
        }
        lt.listen.set(0);
        q.listener().set(0);
    }

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
        // Linked on the endpoint, so destroying it, or revoking the
        // capability the call came through, finds the call.
        let q = self.queue(ep);
        self.q_push(&q.pend_head, &q.pend_tail, caller);
        let ct = self.thread(caller);
        ct.via.set(ep);
        // A caller that gave away the very capability it called through
        // (in this message) is tied to no slot any more.
        let vs = ct.via_slot.get();
        if vs != 0 && !self.slot(vs).cap().is_some_and(|c| c.obj == ep) {
            ct.via_slot.set(0);
        }
        let park = self.thread(server).park.replace(0);
        if park != 0
            && let Some(n) = self.cslot(server, park).ok().and_then(|s| self.slot(s).cap())
            && n.ty == ObjectType::Notification
        {
            let q = self.queue(n.obj);
            let cpu = self.thread(server).cpu.get() + 1;
            // A notification nobody waits on, which nobody owns yet or
            // whose owner's lock is held (every entry that gets here has
            // made sure of that): it becomes the server's processor's.
            if q.head.get() == 0 && q.word.get() == 0 && (q.owner.claim(cpu) || self.holds(q.owner.get() - 1)) {
                q.owner.set(cpu);
                // One call at most is parked on a notification, so it is
                // the whole queue; the caller's links stay the endpoint's.
                q.kind.set(PARKED);
                q.head.set(caller);
                q.tail.set(caller);
                let ct = self.thread(caller);
                ct.ipc_flags.set(ct.ipc_flags.get() | IPC_PARKED);
                self.block(caller, REPLY, n.obj, 0);
                return;
            }
        }
        self.drop_caller(server);
        self.thread(server).caller.set(caller);
        // Keeps the IPC_FAULT flag: the reply path needs it.
        self.block(caller, REPLY, server, 0);
    }

    /// Take `caller` (its call delivered and unanswered) off its endpoint's
    /// list of calls in progress.
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

    /// Receive on endpoint `ep`. `deadline`: 0 or `u64::MAX` for ever, 1 do not block.
    /// `park`: a notification of the receiver's on which the next call is
    /// parked (0: none). `listen`: a notification to listen on as well, as
    /// (capability slot, object), or (0, 0).
    pub(crate) fn recv(&self, t: Paddr, ep: Paddr, deadline: u64, park: u64, listen: (Paddr, Paddr)) -> Result<Exit, Error> {
        let th = self.thread(t);
        if listen.1 != 0 {
            // The notification becomes this processor's, like a queue a
            // thread waits on. Whoever decided which locks this entry
            // needs made sure of its owner's and its last listener's; if
            // it was claimed by another processor since, look again.
            let n = self.queue(listen.1);
            let cpu = th.cpu.get() + 1;
            if !(n.owner.claim(cpu) || self.holds(n.owner.get() - 1)) {
                return Ok(Exit::Preempted);
            }
            if n.kind.get() == PARKED || n.head.get() != 0 {
                // A call is parked there, or a thread waits on it directly.
                return Err(Error::InUse);
            }
            let word = n.word.replace(0);
            if word != 0 {
                th.regs[H::ARG[0]].set(word);
                th.regs[H::ARG[1]].set(MsgInfo::NOTIFIED.0);
                return Ok(Exit::Done);
            }
            let last = n.listener().get();
            if last != t {
                if last != 0 {
                    if !self.holds(self.thread(last).cpu.get()) {
                        return Ok(Exit::Preempted);
                    }
                    if self.listener(listen.1) != 0 {
                        return Err(Error::InUse);
                    }
                    self.thread(last).listen.set(0);
                }
                // This thread listens on one notification at a time.
                let old = th.listen.get();
                if old != 0 && old != listen.1 && self.queue(old).listener().get() == t {
                    self.queue(old).listener().set(0);
                }
                n.listener().set(t);
                th.listen.set(listen.1);
            }
            n.owner.set(cpu);
            th.listen_slot.set(listen.0);
        }
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
        // on it. It moves only while its old owner's lock is held, which
        // every entry that gets here has made sure of.
        qu.owner.set(self.thread(t).cpu.get() + 1);
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
        // A watched line whose word has not changed since it was last
        // delivered: the idle watch delivered this news already.
        if !self.line_news(line) {
            return;
        }
        let q = self.queue(n);
        let me = self.hal.cpu() as u64;
        // Its waiter, or its listener and the endpoint that one waits on,
        // must be this processor's.
        let mine = |t: Paddr| t == 0 || self.thread(t).cpu.get() == me;
        let local = self.ncpu.get() == 1
            || (q.owner.get() == me + 1 && mine(q.head.get()) && mine(q.listener().get()) && {
                let l = self.listener(n);
                l == 0 || self.queue(self.thread(l).blocked_on.get()).owner.get() == me + 1
            });
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
            // A thread listening while it receives on an endpoint hears it
            // now; otherwise it is kept for whoever looks next.
            let l = self.listener(n);
            if l != 0 {
                self.wake_listener(l, q.word.replace(0) | badge, Ok(()));
                return;
            }
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
        // Nor while a thread listens on it; a listener that has moved on
        // is forgotten.
        let last = q.listener().get();
        if last != 0 {
            if self.listener(n) != 0 {
                return Err(Error::InUse);
            }
            self.thread(last).listen.set(0);
            q.listener().set(0);
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
        // Off the endpoint's list of calls in progress and off the thread
        // or notification the answer was owed by.
        self.unanswer(c);
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
        if self.fast_fault(t, kind, addr, code) {
            return;
        }
        let th = self.thread(t);
        let global = self.ncpu.get() > 1 && !th.fault_ep.cap().is_some_and(|c| self.queue_is_local(self.held(), t, &c));
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
                self.thread(t).via_slot.set(0);
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

// The fast paths. A call and a reply-and-receive between two threads of one
// processor, through endpoints that processor owns, with no capabilities in
// the message, is most of all IPC. These do exactly what the general path
// (`send`, `hand_off`, `await_reply`; `reply`, `recv_any`, `hand_off`) does
// in that case, without the checks for everything that cannot happen in
// it. Each first checks every condition, changing nothing, and returns
// false if one fails: the general path then runs as if the fast one had
// not been tried. Under this processor's lock alone, everything checked
// stays as it was checked: it belongs to this processor.
impl<H: Hal> Kernel<H> {
    /// The endpoint or notification capability at `index` of `t`'s
    /// CapSpace with `rights`, and its slot.
    fn fast_cap(&self, t: Paddr, index: u64, ty: ObjectType, rights: Rights) -> Option<(Paddr, crate::cap::Cap)> {
        let slot = self.cslot(t, index).ok()?;
        let cap = self.slot(slot).cap()?;
        (cap.ty == ty && cap.rights.has(rights)).then_some((slot, cap))
    }

    /// Whether a message of `words` words from `from` to `to` can be copied
    /// whole: it fits in registers, or both have an IPC buffer.
    fn fast_words_fit(&self, from: Paddr, to: Paddr, words: usize) -> bool {
        words <= MsgInfo::REG_WORDS || (self.ipc_buffer(from).is_some() && self.ipc_buffer(to).is_some())
    }

    /// Copy `t`'s message (no capabilities) to `to`, as `transfer` does.
    fn fast_transfer(&self, a: &[u64; 6], from: Paddr, to: Paddr, badge: u64) {
        let info = MsgInfo(a[1]);
        let words = info.words();
        let tt = self.thread(to);
        tt.regs[H::ARG[0]].set(badge);
        tt.regs[H::ARG[1]].set(MsgInfo::with_caps(info.label(), words, 0).0);
        for i in 2..6 {
            tt.regs[H::ARG[i]].set(a[i]);
        }
        if words > MsgInfo::REG_WORDS
            && let (Some(src), Some(dst)) = (self.ipc_buffer(from), self.ipc_buffer(to))
        {
            for i in MsgInfo::REG_WORDS..words {
                let o = (i * 8) as u64;
                self.phys.word(dst + o).set(self.phys.word(src + o).get());
            }
        }
        tt.regs[H::SYS].set(0);
    }

    /// Hand this processor to `to`, which waits without a deadline, at the
    /// end of this kernel entry.
    fn fast_hand_off(&self, to: Paddr) {
        let th = self.thread(to);
        th.blocked_on.set(0);
        th.state.set(crate::thread::RUNNABLE);
        let me = self.me();
        me.direct.set(to);
        me.dirty.set(true);
    }

    /// `Call` of thread `t` (current on this processor) with arguments `a`.
    pub(crate) fn fast_call(&self, t: Paddr, a: &[u64; 6]) -> bool {
        let info = MsgInfo(a[1]);
        if info.caps() != 0 {
            return false;
        }
        let Some((slot, ep)) = self.fast_cap(t, a[0], ObjectType::Endpoint, Rights::WRITE) else { return false };
        let me = self.hal.cpu() as u64;
        let q = self.queue(ep.obj);
        if q.owner.get() != me + 1 || q.kind.get() != RECEIVERS {
            return false;
        }
        let r = q.head.get();
        let (th, rt) = (self.thread(t), self.thread(r));
        let local = |x: Paddr| x == 0 || self.thread(x).cpu.get() == me;
        if rt.cpu.get() != me
            || rt.caller.get() != 0
            || rt.park.get() != 0
            || rt.deadline.get() != 0
            || !local(th.caller.get())
            || self.me().direct.get() != 0
            || !self.fast_words_fit(t, r, info.words())
        {
            return false;
        }
        // `send` to a waiting receiver.
        th.via_slot.set(slot);
        th.ipc_flags.set(IPC_CALL | if ep.rights.has(Rights::GRANT) { IPC_GRANT } else { 0 });
        self.q_remove(&q.head, &q.tail, r);
        if q.head.get() == 0 {
            q.kind.set(0);
        }
        self.fast_transfer(a, t, r, ep.badge);
        self.fast_hand_off(r);
        // `await_reply`, without a notification to park on.
        self.q_push(&q.pend_head, &q.pend_tail, t);
        th.via.set(ep.obj);
        rt.caller.set(t);
        th.state.set(REPLY);
        th.blocked_on.set(r);
        true
    }

    /// A fault of thread `t` (current on this processor), reported to its
    /// fault endpoint as a call, as `fault` does through `send`.
    pub(crate) fn fast_fault(&self, t: Paddr, kind: Fault, addr: u64, code: u64) -> bool {
        let th = self.thread(t);
        let Some(ep) = th.fault_ep.cap() else { return false };
        if ep.ty != ObjectType::Endpoint {
            return false;
        }
        let me = self.hal.cpu() as u64;
        let q = self.queue(ep.obj);
        if q.owner.get() != me + 1 || q.kind.get() != RECEIVERS {
            return false;
        }
        let r = q.head.get();
        let rt = self.thread(r);
        let local = |x: Paddr| x == 0 || self.thread(x).cpu.get() == me;
        if rt.cpu.get() != me
            || rt.caller.get() != 0
            || rt.park.get() != 0
            || rt.deadline.get() != 0
            || !local(th.caller.get())
            || self.me().direct.get() != 0
        {
            return false;
        }
        th.fault[0].set(kind as u64);
        th.fault[1].set(addr);
        th.fault[2].set(code);
        th.via_slot.set(0);
        th.ipc_flags.set(IPC_CALL | IPC_FAULT);
        self.q_remove(&q.head, &q.tail, r);
        if q.head.get() == 0 {
            q.kind.set(0);
        }
        // The fault message, as `transfer` builds it.
        rt.regs[H::ARG[0]].set(ep.badge);
        rt.regs[H::ARG[1]].set(MsgInfo::new(FAULT_LABEL + kind as u64, 4).0);
        rt.regs[H::ARG[2]].set(th.regs[H::IP].get());
        rt.regs[H::ARG[3]].set(addr);
        rt.regs[H::ARG[4]].set(code);
        rt.regs[H::ARG[5]].set(th.regs[H::SP].get());
        rt.regs[H::SYS].set(0);
        self.fast_hand_off(r);
        self.q_push(&q.pend_head, &q.pend_tail, t);
        th.via.set(ep.obj);
        rt.caller.set(t);
        th.state.set(REPLY);
        th.blocked_on.set(r);
        true
    }

    /// Whether a fast `ReplyRecv` of thread `t` may go back to listening on
    /// the notification in slot `listen` without doing anything about it:
    /// only in a server loop's steady state, where the thread is already
    /// the notification's listener through the same capability and nothing
    /// is pending there. Out of line: a receive that does not listen pays
    /// one test for it.
    #[cold]
    #[inline(never)]
    fn fast_listen(&self, t: Paddr, listen: u64) -> bool {
        let th = self.thread(t);
        let n = th.listen.get();
        let Ok(ls) = self.cslot(t, listen) else { return false };
        if n == 0
            || th.listen_slot.get() != ls
            || !self.slot(ls).cap().is_some_and(|c| c.ty == ObjectType::Notification && c.obj == n && c.rights.has(Rights::READ))
        {
            return false;
        }
        let nq = self.queue(n);
        nq.owner.get() == self.hal.cpu() as u64 + 1 && nq.word.get() == 0 && nq.kind.get() == 0 && nq.listener().get() == t
    }

    /// `ReplyRecv` of thread `t` (current on this processor) with arguments `a`.
    pub(crate) fn fast_reply_recv(&self, t: Paddr, a: &[u64; 6]) -> bool {
        let info = MsgInfo(a[1]);
        // The endpoint's slot is the low half of `a0` (a notification to
        // listen on is in the upper half, looked at last: keeping both
        // halves apart from the start cost the round trip spills).
        if info.caps() != 0 || a[0] as u32 == 0 {
            return false;
        }
        let th = self.thread(t);
        let c = th.caller.get();
        if c == 0 {
            return false;
        }
        let me = self.hal.cpu() as u64;
        let ct = self.thread(c);
        let via = ct.via.get();
        // An answer to a fault resumes the thread (after mapping a page, for
        // FAULT_MAP); any other answer to one stops it, which is left to the
        // general path.
        let fault = ct.ipc_flags.get() & IPC_FAULT != 0;
        if fault && a[2] != FAULT_MAP && a[2] != FAULT_RESUME {
            return false;
        }
        if ct.cpu.get() != me
            || ct.ipc_flags.get() & IPC_PARKED != 0
            || ct.deadline.get() != 0
            || (via != 0 && self.queue(via).owner.get() != me + 1)
            || self.me().direct.get() != 0
            || !self.fast_words_fit(t, c, info.words())
        {
            return false;
        }
        let Some((slot, ep)) = self.fast_cap(t, recv_slot(a[0]), ObjectType::Endpoint, Rights::READ) else { return false };
        let q = self.queue(ep.obj);
        if q.owner.get() != me + 1 || q.kind.get() == SENDERS {
            return false;
        }
        if a[0] >> LISTEN_SHIFT != 0 && !self.fast_listen(t, listen_slot(a[0])) {
            return false;
        }
        // The mapping is the last thing that can fail: if it does, nothing
        // has changed, and the general path tries it again and reports why.
        if fault && a[2] == FAULT_MAP {
            let Ok((_, asp)) = self.ccap(t, a[3], ObjectType::AddressSpace, Rights::WRITE) else { return false };
            let Ok((fs, _)) = self.ccap(t, a[4], ObjectType::Frame, Rights::NONE) else { return false };
            if self.map(asp.obj, fs, a[5] & !0xFFF, Rights(a[5] as u8 & 0xF)).is_err() {
                return false;
            }
        }
        // `answer` the caller and hand this processor to it: a faulting
        // thread resumes where it faulted, a caller gets the message.
        self.unpend(c);
        th.caller.set(0);
        ct.ipc_flags.set(0);
        if !fault {
            self.fast_transfer(a, t, c, 0);
        }
        self.fast_hand_off(c);
        // `recv` with nothing to receive yet.
        th.park.set(0);
        th.via_slot.set(slot);
        q.kind.set(RECEIVERS);
        self.q_push(&q.head, &q.tail, t);
        th.state.set(RECEIVING);
        th.blocked_on.set(ep.obj);
        true
    }
}
