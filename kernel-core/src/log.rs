//! The kernel's console output.

use crate::hal::Hal;
use crate::lock::Lock;
use core::fmt;
use core::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

/// The writer whose line the console is in the middle of (0: none; a line
/// ends with a newline). Kept with [`LINE`] held, as is everything below.
static OPEN: AtomicU64 = AtomicU64::new(0);
static LINE: Lock = Lock::new();

/// The text of the open line so far, and the line another writer cut
/// short, with its writer (0: none), shown again when that other line ends.
static OPEN_TEXT: Text = Text::new();
static CUT_TEXT: Text = Text::new();
static CUT_BY: AtomicU64 = AtomicU64::new(0);

/// Up to [`TEXT`] bytes of one line; a longer line is not kept (its length
/// past the buffer) and so not shown again.
struct Text {
    bytes: [AtomicU8; TEXT],
    len: AtomicUsize,
}

const TEXT: usize = 256;

impl Text {
    const fn new() -> Text {
        Text { bytes: [const { AtomicU8::new(0) }; TEXT], len: AtomicUsize::new(0) }
    }

    fn clear(&self) {
        self.len.store(0, Ordering::Relaxed);
    }

    fn add(&self, b: &[u8]) {
        let len = self.len.load(Ordering::Relaxed);
        for (to, from) in self.bytes.iter().skip(len).zip(b) {
            to.store(*from, Ordering::Relaxed);
        }
        self.len.store(len.saturating_add(b.len()), Ordering::Relaxed);
    }

    fn set(&self, from: &Text) {
        self.clear();
        let len = from.len.load(Ordering::Relaxed);
        for (to, b) in self.bytes.iter().zip(&from.bytes).take(len) {
            to.store(b.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        self.len.store(len, Ordering::Relaxed);
    }

    /// Write it out, if it was kept whole.
    fn show<H: Hal>(&self, hal: &H) -> bool {
        let len = self.len.load(Ordering::Relaxed);
        if len > TEXT {
            return false;
        }
        let mut buf = [0u8; TEXT];
        for (to, b) in buf.iter_mut().zip(&self.bytes).take(len) {
            *to = b.load(Ordering::Relaxed);
        }
        hal.console(&buf[..len]);
        true
    }
}

/// The kernel's own lines, as a writer.
pub const KERNEL: u64 = u64::MAX;

/// Write `bytes` for `writer` (the kernel, or a thread through its console
/// capability), whole. A writer that finds the line left unfinished by
/// another starts a fresh one, so a program's prompt and a kernel line never
/// share a line, and every line a reader anchors on starts where it should;
/// once the other's line is done, the line it cut short is shown again
/// (a prompt and what was typed after it), so that its writer's next text
/// continues it on the screen as on the line.
pub fn write<H: Hal>(hal: &H, writer: u64, bytes: &[u8]) {
    write_parts(hal, writer, &[bytes]);
}

/// [`write()`] of `parts`, one after another, with nobody else's text among them.
pub fn write_parts<H: Hal>(hal: &H, writer: u64, parts: &[&[u8]]) {
    let Some(last) = parts.iter().rev().find_map(|p| p.last()) else { return };
    LINE.lock();
    let open = OPEN.load(Ordering::Relaxed);
    if open != 0 && open != writer {
        hal.console(b"\r\n");
        // One cut line is kept: the first, until it is shown again.
        if CUT_BY.load(Ordering::Relaxed) == 0 {
            CUT_TEXT.set(&OPEN_TEXT);
            CUT_BY.store(open, Ordering::Relaxed);
        }
        OPEN_TEXT.clear();
    }
    if CUT_BY.load(Ordering::Relaxed) == writer {
        // This writer's own line was cut: it goes on where it was.
        CUT_TEXT.show(hal);
        OPEN_TEXT.set(&CUT_TEXT);
        CUT_BY.store(0, Ordering::Relaxed);
    } else if open != writer {
        OPEN_TEXT.clear();
    }
    for p in parts.iter().filter(|p| !p.is_empty()) {
        hal.console(p);
        match p.iter().rposition(|b| *b == b'\n') {
            Some(n) => {
                OPEN_TEXT.clear();
                OPEN_TEXT.add(&p[n + 1..]);
            }
            None => OPEN_TEXT.add(p),
        }
    }
    let mut now = if *last == b'\n' { 0 } else { writer };
    let cut_by = CUT_BY.load(Ordering::Relaxed);
    if now == 0 && cut_by != 0 {
        // The line done, the one it cut short again below it.
        if CUT_TEXT.show(hal) {
            OPEN_TEXT.set(&CUT_TEXT);
            now = cut_by;
        }
        CUT_BY.store(0, Ordering::Relaxed);
    }
    OPEN.store(now, Ordering::Relaxed);
    LINE.unlock();
}

/// Lines a device can cause (an access the IOMMU refused it) allowed in one
/// [`RATE_WINDOW`]: the console is shared by every processor and a real
/// serial port takes only so many a second, so how often a device and its
/// driver can make the kernel write is the port's rate, whatever they do.
const RATE_LINES: u64 = 10;
const RATE_WINDOW: u64 = 1_000_000_000;

/// The machine's budget of such lines (one per kernel, kept with the line lock
/// held).
pub struct Rate {
    start: AtomicU64,
    count: AtomicU64,
    dropped: AtomicU64,
}

impl Rate {
    /// A full budget.
    pub const fn new() -> Rate {
        Rate { start: AtomicU64::new(0), count: AtomicU64::new(0), dropped: AtomicU64::new(0) }
    }

    /// Whether a line a program caused may be written at `now`: at most
    /// ten in a second. `Some(n)` when it may, `n` being how many
    /// were not written since the last one that was (the line says so).
    pub fn allow(&self, now: u64) -> Option<u64> {
        LINE.lock();
        let start = self.start.load(Ordering::Relaxed);
        if start == 0 || now.saturating_sub(start) >= RATE_WINDOW {
            self.start.store(now.max(1), Ordering::Relaxed);
            self.count.store(0, Ordering::Relaxed);
        }
        let n = self.count.load(Ordering::Relaxed);
        let r = if n < RATE_LINES {
            self.count.store(n + 1, Ordering::Relaxed);
            Some(self.dropped.swap(0, Ordering::Relaxed))
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            None
        };
        LINE.unlock();
        r
    }
}

impl Default for Rate {
    fn default() -> Rate {
        Rate::new()
    }
}

/// One line for the architecture's console, gathered and written with one
/// call, so that lines from different processors (and the pieces programs
/// write through their console capability) do not tear one another. A line
/// longer than the buffer is written in several calls.
pub struct Console<'a, H: Hal> {
    hal: &'a H,
    buf: [u8; 200],
    len: usize,
}

impl<'a, H: Hal> Console<'a, H> {
    /// An empty line for `hal`'s console.
    pub fn new(hal: &'a H) -> Console<'a, H> {
        Console { hal, buf: [0; 200], len: 0 }
    }

    /// Write out what has been gathered.
    pub fn flush(&mut self) {
        if self.len != 0 {
            write(self.hal, KERNEL, &self.buf[..self.len]);
            self.len = 0;
        }
    }
}

impl<H: Hal> fmt::Write for Console<'_, H> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for chunk in s.as_bytes().chunks(self.buf.len()) {
            if self.len + chunk.len() > self.buf.len() {
                self.flush();
            }
            self.buf[self.len..self.len + chunk.len()].copy_from_slice(chunk);
            self.len += chunk.len();
        }
        Ok(())
    }
}

/// Write one line to the kernel console.
#[macro_export]
macro_rules! klog {
    ($k:expr, $($arg:tt)*) => {{
        use core::fmt::Write as _;
        let mut c = $crate::log::Console::new(&$k.hal);
        let _ = write!(c, $($arg)*);
        let _ = c.write_str("\r\n");
        c.flush();
    }};
}

/// The bytes a console has been given and not yet sent: the architecture's
/// console writes into it and feeds the device only as fast as the device
/// takes them, never waiting on it (a 16550 at 115,200 baud takes a byte
/// every 87 us: a whole line, waited for with interrupts off, held a
/// processor for 19 ms). Every use is under the console's own lock; the
/// atomics only make it shareable.
///
/// A writer that finds no room for a piece drops the whole piece and it is
/// counted; the next piece that fits is preceded by a line saying how many
/// bytes were not written, so a reader knows the console lost something.
pub struct TxRing<const N: usize> {
    buf: [AtomicU8; N],
    /// Bytes ever put and ever taken: `head - tail` are waiting.
    head: AtomicUsize,
    tail: AtomicUsize,
    dropped: AtomicU64,
}

impl<const N: usize> TxRing<N> {
    /// An empty ring.
    pub const fn new() -> Self {
        TxRing { buf: [const { AtomicU8::new(0) }; N], head: AtomicUsize::new(0), tail: AtomicUsize::new(0), dropped: AtomicU64::new(0) }
    }

    fn room(&self) -> usize {
        N - (self.head.load(Ordering::Relaxed) - self.tail.load(Ordering::Relaxed))
    }

    fn push(&self, bytes: &[u8]) {
        let mut h = self.head.load(Ordering::Relaxed);
        for &b in bytes {
            self.buf[h % N].store(b, Ordering::Relaxed);
            h += 1;
        }
        self.head.store(h, Ordering::Relaxed);
    }

    /// Queue `bytes` whole, or drop them whole and count them.
    pub fn put(&self, bytes: &[u8]) {
        let dropped = self.dropped.load(Ordering::Relaxed);
        if dropped != 0 {
            let mut note = [0u8; 64];
            let n = note_dropped(&mut note, dropped);
            if self.room() < n + bytes.len() {
                self.dropped.store(dropped + bytes.len() as u64, Ordering::Relaxed);
                return;
            }
            self.push(&note[..n]);
            self.dropped.store(0, Ordering::Relaxed);
        }
        if self.room() < bytes.len() {
            self.dropped.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            return;
        }
        self.push(bytes);
    }

    /// The last bytes put, sent or not, as many as `out` holds and the
    /// ring still has (a crash record's console tail). Returns how many.
    pub fn recent(&self, out: &mut [u8]) -> usize {
        let h = self.head.load(Ordering::Relaxed);
        let n = out.len().min(h).min(N);
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o = self.buf[(h - n + i) % N].load(Ordering::Relaxed);
        }
        n
    }

    /// Whether anything waits to be sent.
    pub fn pending(&self) -> bool {
        self.head.load(Ordering::Relaxed) != self.tail.load(Ordering::Relaxed)
    }

    /// The next byte to send, taken.
    pub fn take(&self) -> Option<u8> {
        let t = self.tail.load(Ordering::Relaxed);
        if t == self.head.load(Ordering::Relaxed) {
            return None;
        }
        let b = self.buf[t % N].load(Ordering::Relaxed);
        self.tail.store(t + 1, Ordering::Relaxed);
        Some(b)
    }
}

impl<const N: usize> Default for TxRing<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// "\r\n[console: N bytes not written]\r\n" into `out`; its length.
fn note_dropped(out: &mut [u8; 64], n: u64) -> usize {
    let mut len = 0;
    let mut put = |b: &[u8]| {
        out[len..len + b.len()].copy_from_slice(b);
        len += b.len();
    };
    put(b"\r\n[console: ");
    let mut digits = [0u8; 20];
    let (mut v, mut i) = (n, digits.len());
    loop {
        i -= 1;
        digits[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    put(&digits[i..]);
    put(b" bytes not written]\r\n");
    len
}
