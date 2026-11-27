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
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, KERNEL_BASE, PHYS_MAP_BASE, Region, RegionKind};
use kernel_core::elf::{Elf, PF_W, PF_X};
use silica_abi::root::{ENTROPY_CPU, ENTROPY_FIRMWARE};
use silica_formats::blake2s::Blake2s;

const PAGE: u64 = 4096;
const HUGE: u64 = 2 << 20;

/// A test image's loader (the `harness` feature) was told a test harness
/// listens for the exit status: the command line says how (`exit=semihosting`
/// on aarch64, see `arch/aarch64.rs`; `exit=port`, QEMU's debug-exit device,
/// on x86_64). A loader built for a machine has neither word nor call.
#[cfg(feature = "harness")]
static HARNESS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Stop for good after a failure; a test image's loader first tells a
/// listening harness.
fn stop() -> ! {
    #[cfg(feature = "harness")]
    if HARNESS.load(core::sync::atomic::Ordering::Relaxed) {
        arch::tell(kernel_core::ExitStatus::Failure.code());
    }
    arch::halt()
}

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

/// Report a fatal error and stop.
fn fail(what: &str) -> ! {
    print("ERROR: loader: ");
    print(what);
    print("\r\n");
    stop()
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

/// The boot archive's entry that carries the machine's secret to the
/// service manager: added here, in memory, never on the boot volume.
const SECRET_ENTRY: &str = "machine.secret";
/// The UEFI variable that keeps it, UCS-2 and NUL-terminated.
const SECRET_VARIABLE: [u16; 20] = {
    let name = b"SilicaMachineSecret";
    let mut w = [0u16; 20];
    let mut i = 0;
    while i < name.len() {
        w[i] = name[i] as u16;
        i += 1;
    }
    w
};

/// This machine's own secret, 32 bytes that last across boots, into `out`:
/// what the keys of its services that must outlive a boot are made from
/// (the remote service's machine key), so that no such key is stored on a
/// disk, where every program that reaches the disk's blocks could read it.
/// Kept in a non-volatile UEFI variable with boot-services access alone,
/// which the firmware answers only before ExitBootServices: to a loader,
/// not to any operating system.
///
/// - What is kept, exactly so (32 bytes, those attributes), is used.
/// - Not found, or found otherwise (another size, or runtime access, which
///   a running system could have set or read): made from `seed` and kept,
///   but only when the loader trusts a source of randomness that went into
///   the seed (`trusted`): a seed of timing jitter alone would be a
///   guessable secret for good. Untrusted, one for this boot alone, said.
/// - Any other answer (the store could not be read): one for this boot
///   alone, said, and what is kept is left as it is, so that a passing
///   fault does not change the machine's keys.
/// - Where the firmware cannot keep one it made: this boot's alone, said.
///
/// No copy of the secret is left here: it is written into `out` alone.
fn machine_secret(rt: *const efi::RuntimeServices, seed: &[u8; 32], trusted: bool, out: &mut [u8; 32]) {
    let want = efi::VARIABLE_NON_VOLATILE | efi::VARIABLE_BOOTSERVICE_ACCESS;
    let (mut size, mut attrs) = (out.len(), 0u32);
    // SAFETY: runtime services are valid while boot services run; the name
    // is NUL-terminated, and `out` holds `size` bytes.
    let st =
        unsafe { ((*rt).get_variable)(SECRET_VARIABLE.as_ptr(), &efi::SILICA_VARIABLES, &mut attrs, &mut size, out.as_mut_ptr().cast()) };
    use kernel_core::boot::{SecretKept, SecretPlan, secret_plan};
    let kept = match st {
        efi::SUCCESS if size == out.len() && attrs == want => SecretKept::Exactly,
        efi::SUCCESS | efi::BUFFER_TOO_SMALL => SecretKept::Otherwise,
        efi::NOT_FOUND => SecretKept::Missing,
        _ => SecretKept::Unreadable,
    };
    let plan = secret_plan(kept, trusted);
    if plan == SecretPlan::Use {
        return;
    }
    let mut made = Blake2s::keyed(seed).update(b"machine secret").finish();
    out.copy_from_slice(&made);
    silica_formats::blake2s::zero(&mut made);
    if plan == SecretPlan::ThisBoot {
        print(if kept == SecretKept::Unreadable {
            "WARNING: loader: the firmware could not say what it keeps of this machine's secret: keys made from it last this boot only\r\n"
        } else {
            "WARNING: loader: no source of randomness the loader trusts: this machine's secret is not made to last; keys made from it last this boot only\r\n"
        });
        return;
    }
    // SAFETY: as above. A variable of the name with other attributes is
    // deleted first (size 0), as SetVariable cannot change attributes.
    let stored = unsafe {
        let _ = ((*rt).set_variable)(SECRET_VARIABLE.as_ptr(), &efi::SILICA_VARIABLES, 0, 0, ptr::null());
        ((*rt).set_variable)(SECRET_VARIABLE.as_ptr(), &efi::SILICA_VARIABLES, want, out.len(), out.as_ptr().cast())
    };
    if stored == efi::SUCCESS {
        print("loader: made this machine's secret; the firmware keeps it for its loader alone\r\n");
    } else {
        print("WARNING: loader: the firmware cannot keep this machine's secret: keys made from it last this boot only\r\n");
    }
}

/// The boot archive to hand over: the one read from the boot volume
/// (`read`: its address and length, in loader memory) with the machine's
/// secret added as [`SECRET_ENTRY`], in memory of the ramdisk's kind. An
/// entry of that name on the volume is dropped, and said: the secret is the
/// firmware's to give. An archive that does not pass the format's check is
/// handed over unchanged, for the kernel to refuse and say why; the secret
/// is then not added. No copy of the secret is made here but the archive's;
/// the caller wipes `secret`.
fn boot_archive(boot: &Boot, read: (u64, u64), secret: &mut [u8; 32]) -> (u64, u64) {
    use silica_formats::ramdisk;
    let as_ramdisk = |bytes: &[u8]| {
        let at = boot.pages(efi::MEM_SILICA_RAMDISK, bytes.len() as u64, "out of memory for the boot archive");
        // SAFETY: the pages were just allocated, at least `bytes.len()` long.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len()) };
        (at, bytes.len() as u64)
    };
    if read.1 == 0 {
        return (0, 0);
    }
    // SAFETY: `read` is the file just read into loader memory.
    let disk = unsafe { core::slice::from_raw_parts(read.0 as *const u8, read.1 as usize) };
    if ramdisk::check(disk, ramdisk::crc32c_update).is_err() {
        return as_ramdisk(disk);
    }
    // The entries, in scratch memory (the loader has no heap): written one
    // by one before they are read as a slice.
    let n = ramdisk::entries(disk).count() + 1;
    let scratch =
        boot.pages(efi::MEM_LOADER_DATA, (n * core::mem::size_of::<(&str, &[u8])>()) as u64, "out of memory for the boot archive");
    let list = scratch as *mut (&str, &[u8]);
    let mut k = 0;
    for (name, data) in ramdisk::entries(disk) {
        if name == SECRET_ENTRY {
            print("WARNING: loader: the boot volume's archive has its own machine.secret, which is not used\r\n");
            continue;
        }
        // SAFETY: `k < n` places, each the size of an entry, allocated above.
        unsafe { list.add(k).write((name, data)) };
        k += 1;
    }
    // SAFETY: as above.
    unsafe { list.add(k).write((SECRET_ENTRY, &secret[..])) };
    k += 1;
    // SAFETY: the first `k` places were written above.
    let entries = unsafe { core::slice::from_raw_parts(list, k) };
    let made = ramdisk::archive_len(entries).map(|len| {
        let at = boot.pages(efi::MEM_SILICA_RAMDISK, len as u64, "out of memory for the boot archive");
        // SAFETY: the pages were just allocated, zeroed, `len` bytes or more.
        let out = unsafe { core::slice::from_raw_parts_mut(at as *mut u8, len) };
        ramdisk::write(entries, out, ramdisk::crc32c_update).map(|len| (at, len as u64))
    });
    silica_formats::blake2s::zero(secret);
    match made.flatten() {
        Some(a) => a,
        None => {
            print("WARNING: loader: the boot archive has no room for the machine's secret; it is handed over without it\r\n");
            as_ramdisk(disk)
        }
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

/// The kind a firmware memory descriptor's range has in the region table
/// the kernel hands memory out by: [`kind_of`] its type, but memory-mapped
/// I/O the firmware keeps for its runtime services (EFI_MEMORY_RUNTIME: on
/// a PC its flash, the variable store, the machine's secret in it) is
/// withheld from every program, as Silica calls no runtime service (QEMU's
/// q35 with OVMF lists one such range, the 4 MiB of flash below 4 GiB; the
/// PCI configuration space is not one). How the loader maps a range is
/// still its type's ([`kind_of`]: device memory, uncached).
fn region_kind(d: &efi::MemoryDescriptor) -> RegionKind {
    if matches!(d.ty, efi::MEM_MMIO | efi::MEM_MMIO_PORT) && d.attr & efi::MEMORY_RUNTIME != 0 {
        return RegionKind::Withheld;
    }
    kind_of(d.ty)
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
        // The firmware's runtime services, which nothing calls once Silica
        // runs (it turns the machine off and resets it through ACPI): kept
        // from every program like the kernel's own memory. Their data holds
        // the firmware's copy of its variables, boot-services-only ones
        // too: the machine's secret (`machine_secret`). Lent as firmware
        // memory, the device manager could read it there.
        efi::MEM_RUNTIME_CODE | efi::MEM_RUNTIME_DATA => RegionKind::Kernel,
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
/// early in its boot, and all inside the 16 MiB below the 64 MiB mark, so
/// the 64 MiB block above stays whole for a service that needs it.
const CRASH_AT: [u64; 3] = [0x0300_0000, 0x0380_0000, 0x03C0_0000];

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
    // The command line first: in a test image, whether a harness hears the
    // status decides how every failure after this one ends (a missing kernel
    // included).
    let cmdline = boot.read(root, "SILICA\\CMDLINE.TXT", efi::MEM_LOADER_DATA).unwrap_or((0, 0));
    #[cfg(feature = "harness")]
    {
        // SAFETY: the command line was read into `cmdline.1` bytes at `cmdline.0`.
        let words = unsafe { core::slice::from_raw_parts(cmdline.0 as *const u8, cmdline.1 as usize) };
        HARNESS.store(
            words.split(|b| b.is_ascii_whitespace()).any(|w| w == b"exit=semihosting" || w == b"exit=port"),
            core::sync::atomic::Ordering::Relaxed,
        );
    }
    let (kfile, klen) = boot.read(root, "SILICA\\KERNEL.ELF", efi::MEM_LOADER_DATA).unwrap_or_else(|| fail("SILICA\\KERNEL.ELF not found"));
    let ramdisk = boot.read(root, "SILICA\\RAMDISK.IMG", efi::MEM_LOADER_DATA).unwrap_or((0, 0));
    let t_files = arch::counter();
    // Refused here, once the command line said whether a harness hears the
    // status (the loader is built for plain x86-64 and Armv8.0, so reading
    // two files first is safe on any processor of either).
    if let Some(what) = arch::unsupported() {
        print("ERROR: loader: this processor lacks ");
        print(what);
        print(", which Silica requires\r\n");
        stop()
    }

    // Entropy and time, while the firmware can still be asked.
    let (seed, entropy_sources) = gather_entropy(&boot);
    // This machine's secret, from the firmware (made the first time),
    // given to the service manager in the boot archive in memory alone.
    // Made to last only from a seed with a trusted source in it.
    let mut secret = [0u8; 32];
    machine_secret(rt, &seed, entropy_sources != 0, &mut secret);
    let ramdisk = boot_archive(&boot, ramdisk, &mut secret);
    silica_formats::blake2s::zero(&mut secret);
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
    // Overlaps take the most restrictive kind; a range that wraps is
    // refused by name (`kernel_core::boot::convert_map`, host-tested).
    // SAFETY: `regions` was allocated for MAX_REGIONS entries, the loader's own.
    let out = unsafe { core::slice::from_raw_parts_mut(regions as *mut Region, MAX_REGIONS) };
    let n = match kernel_core::boot::convert_map(
        count,
        |i| {
            let d = desc(i);
            (d.phys, d.pages.checked_mul(PAGE).unwrap_or(u64::MAX), region_kind(d) as u32)
        },
        out,
    ) {
        Ok(n) => n,
        Err(why) => fail(why),
    };

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
