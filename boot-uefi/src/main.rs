//! Silica UEFI loader for x86_64 and aarch64 (component B01).
//!
//! Reads `SILICA\KERNEL.ELF`, and when present `SILICA\RAMDISK.IMG` and
//! `SILICA\CMDLINE.TXT`, from the volume it was started from; places the
//! kernel at a random 2 MiB-aligned address in the top 2 GiB; builds the
//! page tables the kernel starts on (all physical memory at
//! `PHYS_MAP_BASE`, the kernel image with per-segment permissions); collects
//! the memory map, framebuffer, ACPI root, firmware entropy and wall-clock
//! time into a `BootInfo`; leaves boot services and jumps to the kernel.
//!
//! Every failure prints one line on the serial console saying what went wrong.
//! What differs between architectures is in `arch/`.
#![no_std]
#![no_main]

mod efi;

#[cfg_attr(target_arch = "x86_64", path = "arch/x86_64.rs")]
#[cfg_attr(target_arch = "aarch64", path = "arch/aarch64.rs")]
mod arch;

use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, KERNEL_BASE, PHYS_MAP_BASE, Region, RegionKind};
use kernel_core::elf::{Elf, PF_W, PF_X};
use silica_abi::root::{ENTROPY_CPU, ENTROPY_FIRMWARE};
use silica_formats::blake2s::Blake2s;

const PAGE: u64 = 4096;
const HUGE: u64 = 2 << 20;

/// A test harness is listening for the exit status: the command line says
/// how (`exit=semihosting` on aarch64, see `arch/aarch64.rs`; `exit=port`,
/// QEMU's debug-exit device, on x86_64). On a machine with neither, the
/// loader writes nothing it was not told is there.
static HARNESS: AtomicBool = AtomicBool::new(false);

fn print(s: &str) {
    arch::print(s);
}

/// Print `v` in decimal (no allocation).
fn print_u64(mut v: u64) {
    let mut b = [0u8; 20];
    let mut i = b.len();
    loop {
        i -= 1;
        b[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    print(core::str::from_utf8(&b[i..]).unwrap_or("?"));
}

/// Report a fatal error and stop: status through the emulator's exit device
/// when there is one, then halt.
fn fail(what: &str) -> ! {
    print("ERROR: loader: ");
    print(what);
    print("\r\n");
    arch::stop(kernel_core::ExitStatus::Failure.code(), HARNESS.load(Ordering::Relaxed))
}

struct Boot {
    bs: *const efi::BootServices,
}

impl Boot {
    /// Allocate zeroed pages of the given memory type.
    fn pages(&self, ty: u32, bytes: u64, what: &str) -> u64 {
        let mut addr = 0u64;
        let n = bytes.div_ceil(PAGE) as usize;
        // SAFETY: boot services are valid until ExitBootServices; `addr`
        // outlives the call.
        let st = unsafe { ((*self.bs).allocate_pages)(0, ty, n.max(1), &mut addr) };
        if st != efi::SUCCESS {
            fail(what);
        }
        // SAFETY: the firmware just gave us these pages.
        unsafe { ptr::write_bytes(addr as *mut u8, 0, n.max(1) * PAGE as usize) };
        addr
    }

    /// The crash record's region (`kernel_core::crash`): the first of a few
    /// fixed addresses the firmware lets this loader have, neither zeroed
    /// nor ever given to anyone else, so a panic's record outlives a warm
    /// reset at an address the next boot finds again. 0 when none is free.
    fn crash_region(&self, ram: u64) -> u64 {
        let pages = (kernel_core::crash::REGION_BYTES / PAGE) as usize;
        for at in CRASH_AT {
            let mut addr = ram + at;
            // SAFETY: valid boot services; `addr` outlives the call. Type 2
            // is AllocateAddress: exactly these pages or nothing.
            if unsafe { ((*self.bs).allocate_pages)(2, efi::MEM_SILICA_CRASH, pages, &mut addr) } == efi::SUCCESS {
                return addr;
            }
        }
        0
    }

    fn protocol<T>(&self, handle: efi::Handle, guid: &efi::Guid) -> Option<*mut T> {
        let mut p: *mut c_void = ptr::null_mut();
        // SAFETY: valid boot services; `p` outlives the call.
        let st = unsafe { ((*self.bs).handle_protocol)(handle, guid, &mut p) };
        (st == efi::SUCCESS && !p.is_null()).then_some(p.cast())
    }

    fn locate<T>(&self, guid: &efi::Guid) -> Option<*mut T> {
        let mut p: *mut c_void = ptr::null_mut();
        // SAFETY: valid boot services; `p` outlives the call.
        let st = unsafe { ((*self.bs).locate_protocol)(guid, ptr::null_mut(), &mut p) };
        (st == efi::SUCCESS && !p.is_null()).then_some(p.cast())
    }

    /// Read a whole file into pages of type `ty`. `None` when it does not exist.
    fn read(&self, root: *mut efi::File, name: &str, ty: u32) -> Option<(u64, u64)> {
        let mut wide = [0u16; 64];
        for (i, b) in name.bytes().enumerate() {
            wide[i] = b as u16;
        }
        let mut f: *mut efi::File = ptr::null_mut();
        // SAFETY: `root` is the open volume; the name is NUL-terminated UCS-2.
        if unsafe { ((*root).open)(root, &mut f, wide.as_ptr(), 1, 0) } != efi::SUCCESS {
            return None;
        }
        let mut size = 0u64;
        // SAFETY: `f` is an open file. Seeking to the all-ones position goes to the end.
        unsafe {
            ((*f).set_position)(f, u64::MAX);
            ((*f).get_position)(f, &mut size);
            ((*f).set_position)(f, 0);
        }
        let buf = self.pages(ty, size, "out of memory reading a file");
        let mut got = size as usize;
        // SAFETY: `buf` has at least `size` bytes.
        let st = unsafe { ((*f).read)(f, &mut got, buf as *mut u8) };
        // SAFETY: `f` is open and not used again.
        unsafe { ((*f).close)(f) };
        if st != efi::SUCCESS || got as u64 != size {
            fail("short read from the boot volume");
        }
        Some((buf, size))
    }
}

/// Page-table builder over pages allocated from the firmware: `root` for
/// the kernel half, `low` for the loader's identity alias (the same table
/// where the architecture has one root).
struct Tables<'a> {
    boot: &'a Boot,
    root: u64,
    low: u64,
}

impl Tables<'_> {
    fn entry(&self, table: u64, index: u64) -> *mut u64 {
        (table + index * 8) as *mut u64
    }
    fn root_for(&self, virt: u64) -> u64 {
        if virt >> 63 == 1 { self.root } else { self.low }
    }
    /// The table the entry points to, created when missing.
    fn next(&self, e: *mut u64) -> u64 {
        // SAFETY: `e` points into a page-table page this loader allocated;
        // nothing else uses these pages before the kernel starts.
        unsafe {
            if !arch::present(*e) {
                *e = arch::table(self.boot.pages(efi::MEM_SILICA_KERNEL, PAGE, "out of memory for page tables"));
            }
            *e & arch::ADDR
        }
    }
    /// Map `[phys, phys + len)` at `virt` with 2 MiB pages.
    fn map_huge(&self, virt: u64, phys: u64, len: u64, write: bool, exec: bool) {
        let mut off = 0;
        while off < len {
            let v = virt + off;
            let pdpt = self.next(self.entry(self.root_for(v), v >> 39 & 511));
            let pd = self.next(self.entry(pdpt, v >> 30 & 511));
            // SAFETY: as in `next`.
            unsafe { *self.entry(pd, v >> 21 & 511) = arch::block(phys + off, write, exec) };
            off += HUGE;
        }
    }
    /// Map `[0, top)` at `virt`: what the firmware's memory map (`ram`: of a
    /// range, whether it is memory throughout, `Some(true)`, none of it,
    /// `Some(false)`, or a mix) says is memory as ordinary write-back
    /// memory, and the rest (device registers, reserved ranges, holes)
    /// uncached, in 2 MiB pages where a block is all one or the other and 4
    /// KiB pages where it is mixed. Device registers the kernel reaches
    /// through this map (the I/O APIC, the IOMMUs) are then uncached
    /// whatever the firmware's memory-type ranges say. The framebuffer
    /// (`fb`: base and length) is write-combining: the kernel's console
    /// writes it a pixel at a time, which uncached would take seconds a
    /// screen.
    #[cfg(target_arch = "x86_64")]
    fn map_physical(&self, virt: u64, top: u64, fb: (u64, u64), ram: impl Fn(u64, u64) -> Option<bool>) {
        let in_fb = |a: u64| fb.1 != 0 && a >= fb.0 && a - fb.0 < fb.1;
        let touches_fb = |a: u64, len: u64| fb.1 != 0 && a < fb.0.saturating_add(fb.1) && fb.0 < a + len;
        let mut at = 0;
        while at < top {
            let v = virt + at;
            let pdpt = self.next(self.entry(self.root_for(v), v >> 39 & 511));
            let pd = self.next(self.entry(pdpt, v >> 30 & 511));
            // The first 2 MiB in pages, page 0 left out: physical page 0 is
            // never an object, so a null object address faults (the kernel
            // relies on it).
            match if at == 0 || touches_fb(at, HUGE) { None } else { ram(at, HUGE) } {
                // SAFETY: as in `next`.
                Some(true) => unsafe { *self.entry(pd, v >> 21 & 511) = arch::block(at, true, true) },
                // SAFETY: as in `next`.
                Some(false) => unsafe { *self.entry(pd, v >> 21 & 511) = arch::block_device(at) },
                None => {
                    let pt = self.next(self.entry(pd, v >> 21 & 511));
                    for p in (0..HUGE).step_by(PAGE as usize) {
                        if at + p == 0 {
                            continue;
                        }
                        let e = if in_fb(at + p) {
                            arch::page_framebuffer(at + p)
                        } else if ram(at + p, PAGE) == Some(true) {
                            arch::page(at + p, true, true)
                        } else {
                            arch::page_device(at + p)
                        };
                        // SAFETY: as in `next`.
                        unsafe { *self.entry(pt, (v + p) >> 12 & 511) = e };
                    }
                }
            }
            at += HUGE;
        }
    }
    /// Map one 4 KiB page.
    fn map_page(&self, virt: u64, phys: u64, write: bool, exec: bool) {
        let pdpt = self.next(self.entry(self.root_for(virt), virt >> 39 & 511));
        let pd = self.next(self.entry(pdpt, virt >> 30 & 511));
        let pt = self.next(self.entry(pd, virt >> 21 & 511));
        // SAFETY: as in `next`.
        unsafe { *self.entry(pt, virt >> 12 & 511) = arch::page(phys, write, exec) };
    }
}

/// The boot seed: every source of randomness there is, hashed together,
/// with the sources the loader trusts (`ENTROPY_` bits). No single source
/// decides it: the firmware's protocol and the processor's generator are
/// each taken as they come (either may be absent, broken or hostile), and
/// timing jitter is always added but never counted, because on a virtual
/// machine or a simple core it may be predictable.
fn gather_entropy(boot: &Boot) -> ([u8; 32], u64) {
    let mut h = Blake2s::new();
    h.update(b"silica boot seed 1");
    let mut sources = 0;
    let mut fw = [0u8; 32];
    if let Some(rng) = boot.locate::<efi::Rng>(&efi::RNG) {
        // SAFETY: valid protocol; the buffer is 32 bytes.
        if unsafe { ((*rng).get_rng)(rng, ptr::null(), 32, fw.as_mut_ptr()) } == efi::SUCCESS && fw != [0; 32] {
            sources |= ENTROPY_FIRMWARE;
        }
    }
    h.update(&fw);
    // Four words of 64: as much as the seed holds.
    let mut cpu = [0u64; 4];
    let mut got = 0;
    for w in cpu.iter_mut() {
        if let Some(v) = arch::cpu_random() {
            *w = v;
            got += 1;
        }
    }
    // A generator that answers with the same word every time is broken.
    if got == cpu.len() && cpu.iter().any(|&w| w != cpu[0]) {
        sources |= ENTROPY_CPU;
    }
    for w in cpu {
        h.update(&w.to_le_bytes());
    }
    // Timing: how long each of many small, cache-dependent steps takes.
    let mut x = arch::counter();
    for i in 0..4096u64 {
        let t = arch::counter();
        x = x.rotate_left(7) ^ t ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        if i % 64 == 63 {
            h.update(&x.to_le_bytes());
        }
    }
    h.update(&arch::counter().to_le_bytes());
    (h.finish(), sources)
}

fn unix_seconds(t: &efi::Time) -> u64 {
    if t.year < 1970 || t.month == 0 || t.month > 12 || t.day == 0 {
        return 0;
    }
    // Days since 1970-01-01 (proleptic Gregorian, Howard Hinnant's algorithm).
    let y = t.year as i64 - (t.month <= 2) as i64;
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (t.month as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + t.day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let mut s = days * 86_400 + t.hour as i64 * 3600 + t.minute as i64 * 60 + t.second as i64;
    if t.timezone != 0x7FF {
        s -= t.timezone as i64 * 60;
    }
    s.max(0) as u64
}

/// The region kind for a firmware memory type.
fn kind_of(ty: u32) -> RegionKind {
    match ty {
        efi::MEM_CONVENTIONAL | efi::MEM_BS_CODE | efi::MEM_BS_DATA | efi::MEM_LOADER_CODE => RegionKind::Usable,
        efi::MEM_LOADER_DATA => RegionKind::Loader,
        efi::MEM_SILICA_KERNEL => RegionKind::Kernel,
        efi::MEM_SILICA_RAMDISK => RegionKind::Ramdisk,
        // The crash record: the kernel's, never handed out.
        efi::MEM_SILICA_CRASH => RegionKind::Kernel,
        efi::MEM_MMIO | efi::MEM_MMIO_PORT | efi::MEM_RESERVED => RegionKind::Firmware,
        // ACPI tables and NVS, runtime services, persistent memory and the
        // like: memory, the firmware's.
        _ => RegionKind::FirmwareMemory,
    }
}

const MAX_REGIONS: usize = 1024;
const MAP_BYTES: u64 = 256 * 1024;

/// Where the crash record may live, in order, from the start of memory: low
/// enough for the smallest machine (256 MB), above what the firmware uses
/// early in its boot.
const CRASH_AT: [u64; 3] = [0x0300_0000, 0x0500_0000, 0x0700_0000];

/// UEFI image entry point.
// SAFETY: `efi_main` is the entry symbol the UEFI target's linker expects;
// only the firmware calls it.
#[unsafe(no_mangle)]
pub extern "efiapi" fn efi_main(image: efi::Handle, st: *const efi::SystemTable) -> efi::Status {
    let loader_tsc = arch::counter();
    arch::console(st);
    // SAFETY: the firmware passes a valid system table; boot services are
    // usable until ExitBootServices below.
    let (bs, rt, cfg) =
        unsafe { ((*st).boot_services, (*st).runtime_services, core::slice::from_raw_parts((*st).config_table, (*st).config_count)) };
    let boot = Boot { bs };

    let li: *mut efi::LoadedImage = boot.protocol(image, &efi::LOADED_IMAGE).unwrap_or_else(|| fail("no loaded-image protocol"));
    // SAFETY: the protocol pointer is valid while boot services run.
    let fs: *mut efi::SimpleFs =
        boot.protocol(unsafe { (*li).device }, &efi::SIMPLE_FS).unwrap_or_else(|| fail("boot volume has no file system"));
    let mut root: *mut efi::File = ptr::null_mut();
    // SAFETY: `fs` is a valid protocol instance.
    if unsafe { ((*fs).open_volume)(fs, &mut root) } != efi::SUCCESS {
        fail("cannot open the boot volume");
    }
    let t_open = arch::counter();
    let (kfile, klen) = boot.read(root, "SILICA\\KERNEL.ELF", efi::MEM_LOADER_DATA).unwrap_or_else(|| fail("SILICA\\KERNEL.ELF not found"));
    let ramdisk = boot.read(root, "SILICA\\RAMDISK.IMG", efi::MEM_SILICA_RAMDISK).unwrap_or((0, 0));
    let cmdline = boot.read(root, "SILICA\\CMDLINE.TXT", efi::MEM_LOADER_DATA).unwrap_or((0, 0));
    let t_files = arch::counter();
    // SAFETY: the command line was read into `cmdline.1` bytes at `cmdline.0`.
    let words = unsafe { core::slice::from_raw_parts(cmdline.0 as *const u8, cmdline.1 as usize) };
    HARNESS.store(words.split(|b| b.is_ascii_whitespace()).any(|w| w == b"exit=semihosting" || w == b"exit=port"), Ordering::Relaxed);
    // Refused here, once the command line says whether a harness hears the
    // status (the loader is built for plain x86-64 and Armv8.0, so reading
    // two files first is safe on any processor of either).
    if let Some(what) = arch::unsupported() {
        print("ERROR: loader: this processor lacks ");
        print(what);
        print(", which Silica requires\r\n");
        arch::stop(kernel_core::ExitStatus::Failure.code(), HARNESS.load(Ordering::Relaxed))
    }

    // Entropy and time, while the firmware can still be asked.
    let (seed, entropy_sources) = gather_entropy(&boot);
    let t_entropy = arch::counter();
    let mut time = efi::Time::default();
    // SAFETY: runtime services are valid; `time` outlives the call.
    let wall_secs = if unsafe { ((*rt).get_time)(&mut time, ptr::null_mut()) } == efi::SUCCESS { unix_seconds(&time) } else { 0 };
    let wall_tsc = arch::counter();

    // The kernel image.
    // SAFETY: `kfile` holds `klen` bytes read above and is not modified again.
    let file = unsafe { core::slice::from_raw_parts(kfile as *const u8, klen as usize) };
    let elf = Elf::parse(file, arch::MACHINE).unwrap_or_else(|_| fail(arch::NOT_OURS));
    let (lo, hi) = elf.span().unwrap_or_else(|| fail("kernel image has no loadable segments"));
    if !elf.pie || lo != KERNEL_BASE || hi - lo > 64 << 20 {
        fail("kernel image is not position-independent at the kernel base, or is over 64 MiB");
    }
    // KASLR: a 2 MiB-aligned slide inside the top 2 GiB, leaving 128 MiB at
    // the top, from a value derived from the seed (the root task's secret
    // is derived apart from it, so it does not reveal the slide).
    let slots = ((2u64 << 30) - (128 << 20) - (hi - lo)) / HUGE;
    let rnd = Blake2s::keyed(&seed).update(b"kernel address slide").finish();
    let rnd = u64::from_le_bytes([rnd[0], rnd[1], rnd[2], rnd[3], rnd[4], rnd[5], rnd[6], rnd[7]]);
    let slide = (rnd % slots) * HUGE;
    let entropy = Blake2s::keyed(&seed).update(b"root task").finish();
    let kphys = boot.pages(efi::MEM_SILICA_KERNEL, hi - lo, "out of memory for the kernel image");
    for s in elf.segments() {
        // SAFETY: the destination lies inside the pages just allocated
        // (`span` covers every segment) and the source inside the file.
        unsafe { ptr::copy_nonoverlapping(s.data.as_ptr(), (kphys + (s.vaddr - lo)) as *mut u8, s.data.len()) };
    }
    for r in elf.relative_relocs() {
        let (at, addend) = r.unwrap_or_else(|_| fail("kernel image has a relocation the loader cannot apply"));
        if at < lo || at.checked_add(8).is_none_or(|e| e > hi) {
            fail("kernel relocation outside the image");
        }
        // SAFETY: checked to lie inside the loaded image.
        unsafe { ((kphys + (at - lo)) as *mut u64).write_unaligned(addend.wrapping_add(slide)) };
    }
    arch::sync_code(kphys, hi - lo);

    // Framebuffer and ACPI.
    let mut fb = (0u64, 0u32, 0u32, 0u32, 0u32);
    if let Some(gop) = boot.locate::<efi::Gop>(&efi::GOP) {
        // SAFETY: valid protocol with a current mode.
        unsafe {
            let m = (*gop).mode;
            if !m.is_null() && !(*m).info.is_null() && (*(*m).info).format <= 1 {
                let i = &*(*m).info;
                fb = ((*m).fb_base, i.width, i.height, i.stride, i.format + 1);
            }
        }
    }
    let rsdp = cfg.iter().find(|c| c.guid == efi::ACPI2).map_or(0, |c| c.table as u64);
    arch::console_from_acpi(rsdp);

    // Memory for the things that must exist before the final memory map.
    let info_page = boot.pages(efi::MEM_LOADER_DATA, PAGE, "out of memory for boot information");
    let regions = boot.pages(efi::MEM_LOADER_DATA, (MAX_REGIONS * size_of::<Region>()) as u64, "out of memory for the region table");
    let map = boot.pages(efi::MEM_LOADER_DATA, MAP_BYTES, "out of memory for the firmware memory map");

    // Page tables. Sized from a first look at the memory map.
    let (mut size, mut key, mut dsize, mut dver) = (MAP_BYTES as usize, 0usize, 0usize, 0u32);
    // SAFETY: `map` has MAP_BYTES bytes; the out-parameters outlive the call.
    if unsafe { ((*bs).get_memory_map)(&mut size, map as *mut u8, &mut key, &mut dsize, &mut dver) } != efi::SUCCESS {
        fail("cannot read the firmware memory map");
    }
    let mut top = 4u64 << 30;
    let mut ram = u64::MAX;
    for i in 0..size / dsize {
        // SAFETY: the firmware wrote `size` bytes of descriptors of `dsize` bytes each.
        let d = unsafe { &*((map as usize + i * dsize) as *const efi::MemoryDescriptor) };
        if !matches!(kind_of(d.ty), RegionKind::Firmware | RegionKind::FirmwareMemory) {
            top = top.max(d.phys + d.pages * PAGE);
            ram = ram.min(d.phys);
        }
    }
    let top = (top + (1 << 30) - 1) & !((1 << 30) - 1);
    // The crash record's region, at the same place every boot: an offset
    // from the start of the machine's memory (0 on a PC, 1 GiB on an Arm
    // virtual machine).
    let crash = boot.crash_region(if ram == u64::MAX { 0 } else { ram & !0xF_FFFF });
    let root = boot.pages(efi::MEM_SILICA_KERNEL, PAGE, "out of memory for page tables");
    let low = if arch::SPLIT_ROOTS { boot.pages(efi::MEM_SILICA_KERNEL, PAGE, "out of memory for page tables") } else { root };
    let tables = Tables { boot: &boot, root, low };
    // Physical memory at PHYS_MAP_BASE: everything below the top, or only
    // the RAM where the architecture must not map device registers as
    // memory. The same tables serve as an alias at address 0 so this loader
    // keeps running after the switch; the kernel removes the alias and makes
    // the rest non-executable.
    if arch::MAP_ALL {
        #[cfg(target_arch = "x86_64")]
        tables.map_physical(PHYS_MAP_BASE, top, (fb.0, fb.3 as u64 * fb.2 as u64 * 4), |at, len| {
            // How much of `[at, at + len)` the firmware calls memory (any
            // region but device, port and reserved ranges; a hole is none).
            let mut covered = 0;
            for i in 0..size / dsize {
                // SAFETY: as above: descriptors the firmware wrote.
                let d = unsafe { &*((map as usize + i * dsize) as *const efi::MemoryDescriptor) };
                if kind_of(d.ty) == RegionKind::Firmware {
                    continue;
                }
                let (s, e) = (d.phys.max(at), d.phys.saturating_add(d.pages.saturating_mul(PAGE)).min(at + len));
                covered += e.saturating_sub(s);
            }
            match covered {
                0 => Some(false),
                c if c >= len => Some(true),
                _ => None,
            }
        });
        #[cfg(not(target_arch = "x86_64"))]
        tables.map_huge(PHYS_MAP_BASE, 0, top, true, true);
    } else {
        for i in 0..size / dsize {
            // SAFETY: as above.
            let d = unsafe { &*((map as usize + i * dsize) as *const efi::MemoryDescriptor) };
            if d.ty != efi::MEM_MMIO && d.ty != efi::MEM_MMIO_PORT && d.ty != efi::MEM_RESERVED {
                let start = d.phys & !(HUGE - 1);
                let end = (d.phys + d.pages * PAGE).div_ceil(HUGE) * HUGE;
                tables.map_huge(PHYS_MAP_BASE + start, start, end - start, true, true);
            }
        }
    }
    // SAFETY: both entries are in root tables allocated above.
    unsafe { *tables.entry(tables.low, 0) = *tables.entry(tables.root, 256) };
    for s in elf.segments() {
        let (write, exec) = (s.flags & PF_W != 0, s.flags & PF_X != 0);
        if write && exec {
            fail("kernel image has a segment that is writable and executable");
        }
        let mut v = s.vaddr & !(PAGE - 1);
        while v < s.vaddr + s.memsz {
            tables.map_page(v + slide, kphys + (v - lo), write, exec);
            v += PAGE;
        }
    }

    // Where the loader's time went is said once the firmware is left.
    let t_end = arch::counter();

    // The final memory map and the point of no return. Nothing may allocate
    // between GetMemoryMap and ExitBootServices; if the map changed anyway
    // (the firmware allows one retry), read it again.
    let mut tries = 0;
    loop {
        size = MAP_BYTES as usize;
        // SAFETY: as for the first call.
        if unsafe { ((*bs).get_memory_map)(&mut size, map as *mut u8, &mut key, &mut dsize, &mut dver) } != efi::SUCCESS {
            fail("cannot read the firmware memory map");
        }
        // SAFETY: `image` is this image's handle and `key` is current.
        if unsafe { ((*bs).exit_boot_services)(image, key) } == efi::SUCCESS {
            arch::console_after_exit();
            break;
        }
        tries += 1;
        if tries == 3 {
            fail("the firmware refused ExitBootServices");
        }
    }
    // Where the loader's time went, in counter ticks (the kernel says the
    // counter's rate): finding the volume, reading the files, entropy, the
    // wall clock, laying out the kernel, and the firmware's own exit (on a
    // virtual machine of eight processors most of the boot: it parks the
    // other processors).
    let t_exit = arch::counter();
    print("loader: ticks: volume ");
    print_u64(t_open.wrapping_sub(loader_tsc));
    print(", files ");
    print_u64(t_files.wrapping_sub(t_open));
    print(", entropy ");
    print_u64(t_entropy.wrapping_sub(t_files));
    print(", wall clock ");
    print_u64(wall_tsc.wrapping_sub(t_entropy));
    print(", layout ");
    print_u64(t_end.wrapping_sub(wall_tsc));
    print(", the firmware's exit ");
    print_u64(t_exit.wrapping_sub(t_end));
    print("\r\n");

    // Convert the firmware map: sort by address, merge neighbours of one kind.
    let count = size / dsize;
    let desc = |i: usize| {
        // SAFETY: as above; the map is no longer changing.
        unsafe { &*((map as usize + i * dsize) as *const efi::MemoryDescriptor) }
    };
    let out = regions as *mut Region;
    let mut n = 0usize;
    let mut floor = 0u64;
    loop {
        // Selection sort over at most a few hundred descriptors.
        let mut best: Option<&efi::MemoryDescriptor> = None;
        for i in 0..count {
            let d = desc(i);
            if d.pages != 0 && d.phys >= floor && best.is_none_or(|b| d.phys < b.phys) {
                best = Some(d);
            }
        }
        let Some(d) = best else { break };
        floor = d.phys + d.pages * PAGE;
        let kind = kind_of(d.ty) as u32;
        // SAFETY: `out` has room for MAX_REGIONS entries; `n` is checked below.
        unsafe {
            if n > 0 && (*out.add(n - 1)).kind == kind && (*out.add(n - 1)).base + (*out.add(n - 1)).len == d.phys {
                (*out.add(n - 1)).len += d.pages * PAGE;
            } else {
                if n == MAX_REGIONS {
                    fail("the firmware memory map has too many regions");
                }
                *out.add(n) = Region { base: d.phys, len: d.pages * PAGE, kind, reserved: 0 };
                n += 1;
            }
        }
    }

    let info = BootInfo {
        magic: BOOT_MAGIC,
        version: BOOT_VERSION,
        reserved: 0,
        phys_map_len: top,
        kernel_phys: kphys,
        kernel_virt: lo + slide,
        kernel_len: hi - lo,
        kernel_slide: slide,
        regions,
        region_count: n as u64,
        ramdisk: ramdisk.0,
        ramdisk_len: ramdisk.1,
        cmdline: cmdline.0,
        cmdline_len: cmdline.1,
        rsdp,
        fb_base: fb.0,
        fb_width: fb.1,
        fb_height: fb.2,
        fb_stride: fb.3,
        fb_format: fb.4,
        entropy,
        entropy_sources,
        wall_secs,
        wall_tsc,
        loader_tsc,
        crash,
    };
    // SAFETY: `info_page` is a page allocated above; the tables map this
    // code at its address through `low` and the kernel at its entry.
    unsafe {
        (info_page as *mut BootInfo).write(info);
        arch::handover(tables.root, tables.low, elf.entry.wrapping_add(slide), info_page)
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    fail("internal error (panic)")
}
