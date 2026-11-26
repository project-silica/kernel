//! Watched interrupt lines (`IrqWatch`): a message-signalled line whose
//! news is one word of memory the device writes (virtio's used index).
//!
//! A processor with nothing to run spins for a while before it halts
//! (`sched::Watch`). The interrupt a device sends meanwhile reaches it only
//! after the emulator or the bus has carried the message, and the processor
//! has spent that time spinning anyway. A watched line's word is read in
//! that spin instead, so the completion is seen when the device writes it:
//! a change is delivered exactly as the interrupt would be ([`Kernel::irq`]).
//! The interrupt that follows then finds the word unchanged since the last
//! delivery and is dropped, so the driver is woken once for one news. A
//! processor that halts is woken by the interrupt as before.
//!
//! Telling the device not to interrupt (`IrqQuiet`): while the processor
//! watches, the interrupt it would send is work for nobody (dropped, after
//! the cost of taking it). A line may name a second word, in memory the
//! device reads before it interrupts (virtio's "no interrupt" flag), whose
//! masked bits the kernel alone writes from then on: set, under the lock,
//! just before the processor starts watching, and cleared, under the lock,
//! as soon as it stops for any reason, before it looks at the watched word
//! once more. The device writes its news and then reads the flag; the
//! kernel clears the flag and then reads the news: one of the two sees the
//! other's write, so a completion is either delivered by that last look or
//! interrupts. The device is quiet only while a processor is watching for
//! it; a processor that is busy, or halted, is interrupted as before.
//!
//! Only a processor already idle reads the words: no thread is made to wait
//! and no power is spent that the idle watch would not spend. The lines are
//! watched by the processor they are delivered to.
//!
//! Lost wake-ups: an interrupt is dropped only when `word & mask` equals the
//! value read at a delivery made after the word took that value, and that
//! delivery's signal stays pending until the driver takes it; the driver
//! takes everything new each time it is signalled (the `IrqWatch` contract).
//!
//! A watch names its frames by the capability slots the driver named them
//! with (as a thread's IPC buffer is kept), and looks at a slot each time it
//! uses a word: while the slot holds that frame, the frame is alive, and
//! when it no longer does the watch is inert (every interrupt is news, and
//! no word is written). Frame deletion therefore costs nothing; a slot's
//! own memory is reused only when its CapSpace is destroyed, which ends the
//! watches naming slots in it (`unwatch_space`).
//!
//! A watch lives in its line's own entry, so a driver can watch every line
//! it holds and takes nothing from anyone; the watched lines are a list
//! through those entries, so a processor going idle looks at them alone.
//! The list and the entries are changed only by a holder of every lock; a
//! line's `seen` also by its processor under its own lock, where the line
//! is delivered.

use crate::hal::{Hal, Watched};
use crate::kernel::Kernel;
use crate::mem::Paddr;
use core::cell::Cell;
use silica_abi::{ObjectType, Rights};

/// One line's watch.
pub(crate) struct LineWatch {
    /// The slot that names the frame of the word; 0: the line is not watched.
    slot: Cell<Paddr>,
    /// The frame, and the word's offset in it.
    frame: Cell<Paddr>,
    offset: Cell<u64>,
    mask: Cell<u32>,
    /// `word & mask` at the last delivery (or when the watch began).
    seen: Cell<u32>,
    /// The quiet word (`IrqQuiet`): the slot naming its frame (0: none),
    /// the frame, the offset, and the bits that say "no interrupt".
    quiet_slot: Cell<Paddr>,
    quiet_frame: Cell<Paddr>,
    quiet_offset: Cell<u64>,
    quiet_bits: Cell<u32>,
    /// The processor the line is delivered to: the one that watches it.
    cpu: Cell<u16>,
    /// The next watched line, plus one (0: the last).
    next: Cell<u8>,
}

impl LineWatch {
    pub(crate) const fn new() -> LineWatch {
        LineWatch {
            slot: Cell::new(0),
            frame: Cell::new(0),
            offset: Cell::new(0),
            mask: Cell::new(0),
            seen: Cell::new(0),
            quiet_slot: Cell::new(0),
            quiet_frame: Cell::new(0),
            quiet_offset: Cell::new(0),
            quiet_bits: Cell::new(0),
            cpu: Cell::new(0),
            next: Cell::new(0),
        }
    }
}

/// Most words one processor watches in one idle wait.
pub const WATCHED_PER_CPU: usize = 4;

impl<H: Hal> Kernel<H> {
    fn line_watch(&self, line: usize) -> Option<&LineWatch> {
        let w = &self.watches[line];
        (w.slot.get() != 0).then_some(w)
    }

    /// The watched lines and their watches. A watch may be released while
    /// this goes on: its successor was read before it was given out.
    fn watched(&self) -> impl Iterator<Item = (usize, &LineWatch)> {
        let mut next = self.watched.get() as usize;
        core::iter::from_fn(move || {
            let line = next.checked_sub(1)?;
            let w = &self.watches[line];
            next = w.next.get() as usize;
            Some((line, w))
        })
    }

    /// Whether slot `slot` still holds a capability to frame `frame` with
    /// `rights`, of memory (not a device's), whose size covers the word at
    /// `offset`: while it does, the frame is alive and the word is in it. The
    /// size is checked at every use, not once: the slot may since hold a
    /// smaller frame at the same address (the first was deleted and its
    /// memory retyped into the same slot), and a word past that one's end is
    /// memory its Untyped may have made a kernel object of.
    fn names(&self, slot: Paddr, frame: Paddr, offset: u64, rights: Rights) -> bool {
        self.slot(slot).cap().is_some_and(|c| {
            c.ty == ObjectType::Frame
                && c.obj == frame
                && c.rights.has(rights)
                && c.flags & crate::cap::FLAG_DEVICE == 0
                && offset < 1u64 << c.bits
        })
    }

    /// The watched word of `w`, while its frame is still named.
    fn word(&self, w: &LineWatch) -> Option<Paddr> {
        self.names(w.slot.get(), w.frame.get(), w.offset.get(), Rights::READ).then(|| w.frame.get() + w.offset.get())
    }

    /// The quiet word of `w`, if it has one and its frame is still named.
    fn quiet(&self, w: &LineWatch) -> Option<Paddr> {
        let s = w.quiet_slot.get();
        (s != 0 && self.names(s, w.quiet_frame.get(), w.quiet_offset.get(), Rights(Rights::READ.0 | Rights::WRITE.0)))
            .then(|| w.quiet_frame.get() + w.quiet_offset.get())
    }

    /// The device of `w` may interrupt.
    fn loud(&self, w: &LineWatch) {
        if let Some(q) = self.quiet(w) {
            self.phys.device_word_update(q, 0, w.quiet_bits.get());
        }
    }

    /// `IrqWatch`: the news of `line`, delivered to processor `cpu`, is the
    /// word at `offset` in the frame `frame`, named by slot `slot`, under
    /// `mask`. Holding every lock.
    pub(crate) fn watch_line(&self, line: usize, cpu: usize, slot: Paddr, frame: Paddr, offset: u64, mask: u32) {
        self.unwatch_line(line);
        let w = &self.watches[line];
        w.frame.set(frame);
        w.offset.set(offset);
        w.mask.set(mask);
        // What is there now is not news: a driver arms the watch before it
        // starts requests, or looks once more after arming.
        w.seen.set(self.phys.device_word(frame + offset) & mask);
        w.cpu.set(cpu as u16);
        w.quiet_slot.set(0);
        w.slot.set(slot);
        w.next.set(self.watched.replace(line as u8 + 1));
    }

    /// `line` is no longer watched (rebound, unbound, or asked). Holding
    /// every lock.
    pub(crate) fn unwatch_line(&self, line: usize) {
        if let Some(w) = self.line_watch(line) {
            self.release(line, w);
        }
    }

    /// End the watch `w` of `line`: a device told to be quiet interrupts
    /// again (its processor may be watching now, and would not clear it any
    /// more), and the line is off the list.
    fn release(&self, line: usize, w: &LineWatch) {
        self.loud(w);
        w.quiet_slot.set(0);
        w.slot.set(0);
        let me = line as u8 + 1;
        match self.watched().find(|(_, before)| before.next.get() == me) {
            Some((_, before)) => before.next.set(w.next.get()),
            None => self.watched.set(w.next.get()),
        }
        // Its processor may be reading the word in its idle wait: it looks
        // again (a read until then is only a reason to look under the lock).
        let cpu = w.cpu.get() as usize;
        if cpu != self.hal.cpu() {
            self.hal.kick(cpu, false);
        }
    }

    /// `IrqQuiet`: while watched `line`'s processor watches it, the bits
    /// `bits` of the word at `offset` in frame `frame`, named by slot
    /// `slot`, are set, and clear at all other times. False when the line
    /// is not watched. Holding every lock.
    pub(crate) fn quiet_line(&self, line: usize, slot: Paddr, frame: Paddr, offset: u64, bits: u32) -> bool {
        let Some(w) = self.line_watch(line) else { return false };
        self.loud(w);
        w.quiet_frame.set(frame);
        w.quiet_offset.set(offset);
        w.quiet_bits.set(bits);
        w.quiet_slot.set(slot);
        // Loud until its processor next watches.
        self.loud(w);
        true
    }

    /// The CapSpace whose slots are `[start, end)` is being destroyed: no
    /// watch names a frame by one of them any more. Holding every lock.
    #[cold]
    #[inline(never)]
    pub(crate) fn unwatch_space(&self, start: Paddr, end: Paddr) {
        let within = |s: Paddr| start <= s && s < end;
        for (line, w) in self.watched() {
            if within(w.quiet_slot.get()) {
                self.loud(w);
                w.quiet_slot.set(0);
            }
            if within(w.slot.get()) {
                self.release(line, w);
            }
        }
    }

    /// Whether the interrupt (or the watch) of watched `line` brings news:
    /// its word changed since the last delivery, which this one now is.
    /// True for a line that is not watched, or whose frame is no longer
    /// named. Holding the line's processor's lock (or every lock).
    #[inline(always)]
    pub(crate) fn line_news(&self, line: usize) -> bool {
        self.watched.get() == 0 || self.line_news_slow(line)
    }

    #[inline(never)]
    fn line_news_slow(&self, line: usize) -> bool {
        let Some(w) = self.line_watch(line) else { return true };
        let Some(word) = self.word(w) else { return true };
        let now = self.phys.device_word(word) & w.mask.get();
        now != w.seen.replace(now)
    }

    /// The words this processor watches while idle, into `out`; how many.
    /// Their devices are told to be quiet meanwhile. Holding this
    /// processor's lock.
    pub(crate) fn begin_watches(&self, out: &mut [Watched; WATCHED_PER_CPU]) -> usize {
        let me = self.hal.cpu() as u16;
        let mut n = 0;
        for (_, w) in self.watched() {
            if n == WATCHED_PER_CPU || w.cpu.get() != me {
                continue;
            }
            let Some(word) = self.word(w) else { continue };
            out[n] = Watched { at: self.phys.ptr(word).cast::<u32>().cast_const(), mask: w.mask.get(), seen: w.seen.get() };
            n += 1;
            if let Some(q) = self.quiet(w) {
                self.phys.device_word_update(q, w.quiet_bits.get(), 0);
            }
        }
        n
    }

    /// Back from watching: the devices told to be quiet interrupt again,
    /// then each line whose word changed is delivered, as its interrupt
    /// would be. Holding this processor's lock.
    pub(crate) fn end_watches(&self) {
        if self.watched.get() == 0 {
            return;
        }
        let me = self.hal.cpu() as u16;
        for (_, w) in self.watched().filter(|(_, w)| w.cpu.get() == me) {
            self.loud(w);
        }
        // The flag is cleared before the news is read (the atomic
        // read-modify-write orders them on x86; a fence on the rest).
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        for (line, w) in self.watched().filter(|(_, w)| w.cpu.get() == me) {
            let Some(word) = self.word(w) else { continue };
            if self.phys.device_word(word) & w.mask.get() != w.seen.get() {
                // `irq` takes the news (`line_news`) and signals.
                self.irq(line);
            }
        }
    }
}
