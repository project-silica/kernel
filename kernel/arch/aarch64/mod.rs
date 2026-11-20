//! aarch64 architecture glue (component B05).
//!
//! Booting so far: the loader enters `_start` at EL1 with the MMU on (the
//! kernel half in TTBR1, all RAM at `PHYS_MAP_BASE`, see
//! `boot-uefi/src/arch/aarch64.rs`). The kernel takes its boot stack,
//! installs exception vectors that report any exception as a kernel fault,
//! maps the console UART the ACPI SPCR table names into its device window,
//! checks the boot information and reports the machine. Not yet built: the
//! GIC, the timer, user mode, system calls and the root task; the kernel
//! says so and stops (`spec/MILESTONES.md` M2: aarch64 boots to the same
//! tests as x86_64).

use core::arch::{asm, global_asm};
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use kernel_core::ExitStatus;
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, Cmdline, PHYS_MAP_BASE, Region};

const BLOCK: u64 = 2 << 20;
const STACK_SIZE: usize = 64 << 10;

/// The boot processor's stack.
#[repr(C, align(4096))]
struct Stack(UnsafeCell<[u8; STACK_SIZE]>);
// SAFETY: used only as the boot processor's stack, through the stack pointer.
unsafe impl Sync for Stack {}
static BOOT_STACK: Stack = Stack(UnsafeCell::new([0; STACK_SIZE]));

global_asm!(
    r#"
.section .text.boot, "ax"
.global _start
_start:
    msr daifset, #0xf
    adrp x9, {stack}
    add x9, x9, :lo12:{stack}
    add x9, x9, #{size}
    mov sp, x9
    adrp x9, vectors
    add x9, x9, :lo12:vectors
    msr vbar_el1, x9
    isb
    mov x29, #0
    mov x30, #0
    bl {kmain}
    b .

// Exception vectors: 16 entries of 128 bytes. Until user mode exists every
// exception is a kernel fault: report it with the entry it came through.
.section .text, "ax"
.balign 2048
vectors:
.irp n, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
.balign 128
    mov x3, #\n
    mrs x0, esr_el1
    mrs x1, elr_el1
    mrs x2, far_el1
    mov x4, x29
    b {fault}
.endr
"#,
    stack = sym BOOT_STACK,
    size = const STACK_SIZE,
    kmain = sym kmain,
    fault = sym kernel_fault,
);

// ---- console ---------------------------------------------------------------

/// The console UART's virtual address (0: none known).
static UART: AtomicU64 = AtomicU64::new(0);
/// The kernel was told a test harness listens for the exit status.
static HARNESS: AtomicBool = AtomicBool::new(false);

fn write_bytes(s: &[u8]) {
    let uart = UART.load(Ordering::Relaxed);
    if uart == 0 {
        return;
    }
    for &b in s {
        let mut spins = 0;
        // SAFETY: `uart` is a PL011-compatible UART mapped as device memory
        // in the device window: flag register at 0x18 (bit 5: transmit FIFO
        // full), data register at 0.
        unsafe {
            while core::ptr::read_volatile((uart + 0x18) as *const u32) & 0x20 != 0 && spins < 100_000 {
                spins += 1;
            }
            core::ptr::write_volatile(uart as *mut u32, b as u32);
        }
    }
}

struct Out;
impl Write for Out {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        write_bytes(s.as_bytes());
        Ok(())
    }
}

/// Stop this processor: tell a test harness the kernel failed, then wait.
fn die() -> ! {
    stop(ExitStatus::Failure)
}

fn stop(status: ExitStatus) -> ! {
    if HARNESS.load(Ordering::Relaxed) {
        // Semihosting SYS_EXIT_EXTENDED, ADP_Stopped_ApplicationExit, with the
        // status encoded as x86's debug-exit device reports it.
        let block = [0x2_0026u64, (status.code() as u64) << 1 | 1];
        // SAFETY: the call reads the two-word block and ends the emulation.
        unsafe { asm!("hlt #0xf000", in("w0") 0x20u32, in("x1") block.as_ptr(), options(nostack)) }
    }
    loop {
        // SAFETY: masking interrupts and waiting change nothing else.
        unsafe { asm!("msr daifset, #0xf", "wfe", options(nomem, nostack)) }
    }
}

// ---- the device window -------------------------------------------------------
//
// Device registers are mapped Device-nGnRE in 2 MiB blocks at DEVICE_BASE,
// through two tables in the kernel image, so the window exists before the
// kernel owns any memory. Normal memory is never mapped over devices.

const DEVICE_BASE: u64 = 0xFFFF_FF00_0000_0000;
const ATTR_DEVICE: u64 = 1 << 2;
const AF: u64 = 1 << 10;
const PXN: u64 = 1 << 53;
const UXN: u64 = 1 << 54;
const ADDR: u64 = 0x0000_FFFF_FFFF_F000;

#[repr(C, align(4096))]
struct Table(UnsafeCell<[u64; 512]>);
// SAFETY: changed only by the boot processor during boot.
unsafe impl Sync for Table {}
static DEVICE_L1: Table = Table(UnsafeCell::new([0; 512]));
static DEVICE_L2: Table = Table(UnsafeCell::new([0; 512]));
static DEVICE_USED: AtomicU64 = AtomicU64::new(0);

/// Where the kernel image is in physical memory: (virtual start, physical start).
static IMAGE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn image_phys(v: u64) -> u64 {
    v - IMAGE[0].load(Ordering::Relaxed) + IMAGE[1].load(Ordering::Relaxed)
}

/// Map the device registers at physical `phys` (with the rest of their
/// 2 MiB block) and return their virtual address. Boot processor, boot only.
fn map_device(phys: u64) -> Option<u64> {
    let block = phys & !(BLOCK - 1);
    // SAFETY: the tables are the boot processor's alone during boot, and
    // TTBR1's root is the loader's table, mapped through the physical map.
    unsafe {
        let l2 = &mut *DEVICE_L2.0.get();
        for (i, e) in l2.iter().enumerate().take(DEVICE_USED.load(Ordering::Relaxed) as usize) {
            if *e & ADDR == block {
                return Some(DEVICE_BASE + i as u64 * BLOCK + (phys - block));
            }
        }
        let i = DEVICE_USED.fetch_add(1, Ordering::Relaxed) as usize;
        if i >= 512 {
            return None;
        }
        if i == 0 {
            let root: u64;
            asm!("mrs {}, ttbr1_el1", out(reg) root, options(nomem, nostack, preserves_flags));
            let l1 = &mut *DEVICE_L1.0.get();
            l1[0] = image_phys(DEVICE_L2.0.get() as u64) | 0b11;
            let root = (PHYS_MAP_BASE + (root & ADDR)) as *mut u64;
            *root.add((DEVICE_BASE >> 39 & 511) as usize) = image_phys(DEVICE_L1.0.get() as u64) | 0b11;
        }
        l2[i] = block | 0b01 | ATTR_DEVICE | AF | PXN | UXN;
        asm!("dsb ishst", "isb", options(nostack, preserves_flags));
        Some(DEVICE_BASE + i as u64 * BLOCK + (phys - block))
    }
}

/// The base of the UART the ACPI SPCR table names, if it is a PL011 or an
/// SBSA generic UART in memory space.
fn spcr_uart(rsdp: u64) -> Option<u64> {
    let read8 = |p: u64| -> [u8; 8] {
        // SAFETY: ACPI tables are in RAM the physical map covers; reads are
        // bounded by the lengths the tables give.
        unsafe { core::ptr::read_unaligned((PHYS_MAP_BASE + p) as *const [u8; 8]) }
    };
    let u64_at = |p: u64| u64::from_le_bytes(read8(p));
    let u32_at = |p: u64| u64_at(p) as u32;
    if rsdp == 0 || read8(rsdp) != *b"RSD PTR " {
        return None;
    }
    let xsdt = u64_at(rsdp + 24);
    if xsdt == 0 || read8(xsdt)[..4] != *b"XSDT" {
        return None;
    }
    let len = u32_at(xsdt + 4) as u64;
    let mut at = xsdt + 36;
    while at + 8 <= xsdt + len {
        let t = u64_at(at);
        at += 8;
        if t == 0 || read8(t)[..4] != *b"SPCR" || u32_at(t + 4) < 80 {
            continue;
        }
        let (kind, space, base) = (read8(t + 36)[0], read8(t + 40)[0], u64_at(t + 44));
        if matches!(kind, 0x03 | 0x0E) && space == 0 && base != 0 {
            return Some(base);
        }
    }
    None
}

// ---- faults and panics -------------------------------------------------------

/// Print return addresses by following frame records (x29) on the boot stack.
fn backtrace(first: u64, mut fp: u64) {
    let slide = SLIDE.load(Ordering::Relaxed);
    let _ = write!(Out, "#0 {:#018x}\r\n", first.wrapping_sub(slide));
    let lo = BOOT_STACK.0.get() as u64;
    let hi = lo + STACK_SIZE as u64;
    for n in 1..32 {
        if !(fp.is_multiple_of(16) && fp >= lo && fp + 16 <= hi) {
            break;
        }
        // SAFETY: `fp` was just checked to lie inside the boot stack.
        let (next, ret) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        if ret == 0 {
            break;
        }
        let _ = write!(Out, "#{} {:#018x}\r\n", n, ret.wrapping_sub(slide));
        fp = next;
    }
}

static SLIDE: AtomicU64 = AtomicU64::new(0);

/// The kernel panicked: say why, show where, stop.
pub fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    let _ = write!(Out, "\r\nPANIC: {}\r\n", info.message());
    if let Some(l) = info.location() {
        let _ = write!(Out, "  at {}:{}\r\n", l.file(), l.line());
    }
    let _ = write!(Out, "  kernel slide {:#x}; backtrace (link addresses):\r\n", SLIDE.load(Ordering::Relaxed));
    let (pc, fp): (u64, u64);
    // SAFETY: reads the program counter and the frame pointer.
    unsafe { asm!("adr {}, .", "mov {}, x29", out(reg) pc, out(reg) fp, options(nomem, nostack)) }
    backtrace(pc, fp);
    die()
}

/// An exception taken in the kernel (the only kind there is so far).
extern "C" fn kernel_fault(esr: u64, elr: u64, far: u64, entry: u64, fp: u64) -> ! {
    let _ = write!(
        Out,
        "\r\nPANIC: kernel exception through vector {} (syndrome {:#x}: class {:#x}) at {:#x}, address {:#x}\r\n  kernel slide {:#x}; backtrace (link addresses):\r\n",
        entry,
        esr,
        esr >> 26 & 0x3F,
        elr.wrapping_sub(SLIDE.load(Ordering::Relaxed)),
        far,
        SLIDE.load(Ordering::Relaxed)
    );
    backtrace(elr, fp);
    die()
}

// ---- boot ------------------------------------------------------------------

extern "C" fn kmain(info: u64) -> ! {
    // SAFETY: the loader passes the physical address of a BootInfo in loader
    // memory, which the physical map covers and nothing overwrites.
    let bi: BootInfo = unsafe { core::ptr::read((PHYS_MAP_BASE + info) as *const BootInfo) };
    if bi.magic != BOOT_MAGIC || bi.version != BOOT_VERSION {
        die();
    }
    IMAGE[0].store(bi.kernel_virt, Ordering::Relaxed);
    IMAGE[1].store(bi.kernel_phys, Ordering::Relaxed);
    SLIDE.store(bi.kernel_slide, Ordering::Relaxed);
    // SAFETY: loader memory the physical map covers, read-only from here on.
    let cmdline = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.cmdline) as *const u8, bi.cmdline_len as usize) };
    let cmdline = Cmdline(cmdline);
    HARNESS.store(cmdline.get("exit") == Some(b"semihosting"), Ordering::Relaxed);
    if let Some(uart) = spcr_uart(bi.rsdp).and_then(map_device) {
        UART.store(uart, Ordering::Relaxed);
    }

    // SAFETY: as for the command line.
    let regions = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.regions) as *const Region, bi.region_count as usize) };
    let usable: u64 = regions.iter().filter(|r| r.kind == 1).map(|r| r.len).sum();
    let (freq, el): (u64, u64);
    // SAFETY: reading the counter frequency and the exception level.
    unsafe { asm!("mrs {}, cntfrq_el0", "mrs {}, CurrentEL", out(reg) freq, out(reg) el, options(nomem, nostack)) }
    let _ = write!(
        Out,
        "silica: aarch64, {} MiB free, counter {} Hz, EL{}, kernel slide {:#x}\r\n",
        usable >> 20,
        freq,
        el >> 2 & 3,
        bi.kernel_slide
    );
    if cmdline.get("panic") == Some(b"test") {
        panic!("deliberate panic (panic=test on the command line)");
    }
    let _ = write!(Out, "ERROR: kernel: aarch64 has no interrupt controller, timer, user mode or root task yet (B05)\r\n");
    die()
}
