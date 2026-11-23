//! The kernel's own lines on the firmware's framebuffer (component C30),
//! for a machine with no serial port, and its panic on any machine.
//!
//! From boot until the root task's first instruction every line the kernel
//! writes is drawn here too, light grey on black, eight by sixteen pixels a
//! character (`silica_formats::font`), on the framebuffer the loader found
//! (GOP; 32 bits a pixel). Then the screen is the display driver's: the
//! kernel draws nothing more, until a panic, which is drawn over whatever is
//! there, in white, from the top, since the machine is stopping anyway. A
//! line past the bottom wraps to the top, clearing the line below it: no
//! scrolling, which on uncached device memory would cost more than the
//! lines are worth.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use kernel_core::boot::BootInfo;
use kernel_core::mem::Phys;
use silica_formats::font::{self, HEIGHT, WIDTH};

const GREY: u32 = 0x00AA_AAAA;
const WHITE: u32 = 0x00FF_FFFF;

/// The framebuffer: its address in the kernel's map (0: none), its size in
/// pixels and its stride.
static BASE: AtomicU64 = AtomicU64::new(0);
static WIDTH_PX: AtomicU32 = AtomicU32::new(0);
static HEIGHT_PX: AtomicU32 = AtomicU32::new(0);
static STRIDE: AtomicU32 = AtomicU32::new(0);
/// Where the next character goes, in cells.
static COL: AtomicU32 = AtomicU32::new(0);
static ROW: AtomicU32 = AtomicU32::new(0);
/// Lines are drawn (until the root task runs, and after a panic).
static DRAWING: AtomicBool = AtomicBool::new(false);
static PANICKING: AtomicBool = AtomicBool::new(false);

/// Find the framebuffer and start drawing. Boot processor, once, with the
/// physical map up.
pub fn init(bi: &BootInfo, phys: &Phys) {
    let (w, h, stride) = (bi.fb_width as u64, bi.fb_height as u64, bi.fb_stride as u64);
    let bytes = stride * h * 4;
    if bi.fb_base == 0 || !matches!(bi.fb_format, 1 | 2) || w < WIDTH as u64 || h < HEIGHT as u64 || stride < w {
        return;
    }
    if bi.fb_base.checked_add(bytes).is_none_or(|end| end > bi.phys_map_len) {
        return;
    }
    BASE.store(phys.ptr(bi.fb_base) as u64, Ordering::Relaxed);
    WIDTH_PX.store(w as u32, Ordering::Relaxed);
    HEIGHT_PX.store(h as u32, Ordering::Relaxed);
    STRIDE.store(stride as u32, Ordering::Relaxed);
    // The firmware's text goes; the kernel's starts at the top.
    for y in 0..h {
        fill_row(y, 0);
    }
    DRAWING.store(true, Ordering::Release);
}

/// Whether the kernel has a framebuffer to draw on.
pub fn present() -> bool {
    BASE.load(Ordering::Relaxed) != 0
}

/// The root task is about to run: the screen is the display driver's.
pub fn release() {
    DRAWING.store(false, Ordering::Release);
}

/// A panic: draw what follows from the top, in white, over the screen.
pub fn panic() {
    if present() {
        PANICKING.store(true, Ordering::Relaxed);
        COL.store(0, Ordering::Relaxed);
        ROW.store(0, Ordering::Relaxed);
        DRAWING.store(true, Ordering::Release);
    }
}

fn pixel(x: u64, y: u64, colour: u32) {
    let at = BASE.load(Ordering::Relaxed) + (y * STRIDE.load(Ordering::Relaxed) as u64 + x) * 4;
    // SAFETY: inside the framebuffer, which `init` checked lies in the
    // physical map; nothing of the kernel's is there.
    #[allow(unsafe_code)]
    unsafe {
        (at as *mut u32).write_volatile(colour)
    };
}

fn fill_row(y: u64, colour: u32) {
    for x in 0..WIDTH_PX.load(Ordering::Relaxed) as u64 {
        pixel(x, y, colour);
    }
}

/// Clear the cell row `row`.
fn clear_line(row: u32) {
    for y in 0..HEIGHT as u64 {
        fill_row(row as u64 * HEIGHT as u64 + y, 0);
    }
}

/// Draw `bytes` (the kernel's text: ASCII, `\n` ending a line, `\r` back
/// to its start), if lines are being drawn. Called with the console's line
/// lock held, or on the panic path alone.
pub fn write(bytes: &[u8]) {
    if !DRAWING.load(Ordering::Acquire) {
        return;
    }
    let (cols, rows) = (WIDTH_PX.load(Ordering::Relaxed) / WIDTH as u32, HEIGHT_PX.load(Ordering::Relaxed) / HEIGHT as u32);
    let colour = if PANICKING.load(Ordering::Relaxed) { WHITE } else { GREY };
    let (mut col, mut row) = (COL.load(Ordering::Relaxed), ROW.load(Ordering::Relaxed));
    let newline = |row: &mut u32, col: &mut u32| {
        *col = 0;
        *row = (*row + 1) % rows;
        clear_line(*row);
    };
    for &b in bytes {
        match b {
            b'\n' => newline(&mut row, &mut col),
            b'\r' => col = 0,
            _ => {
                if col == cols {
                    newline(&mut row, &mut col);
                }
                let glyph = font::glyph(b as char);
                for (dy, bits) in glyph.iter().enumerate() {
                    let y = row as u64 * HEIGHT as u64 + dy as u64;
                    for dx in 0..WIDTH {
                        let on = bits & (0x80 >> dx) != 0;
                        pixel(col as u64 * WIDTH as u64 + dx as u64, y, if on { colour } else { 0 });
                    }
                }
                col += 1;
            }
        }
    }
    COL.store(col, Ordering::Relaxed);
    ROW.store(row, Ordering::Relaxed);
}
