//! Creating the root task: the one thread the kernel starts itself, holding
//! a capability to everything (component D01's kernel half).
//!
//! The root task's objects are made the way any others are, by retyping an
//! Untyped, so it owns and is accounted for its own memory.

use crate::boot::{Region, RegionKind};
use crate::cap::{Cap, FLAG_DEVICE, FLAG_IRQS, FLAG_PORTS};
use crate::elf::{Elf, ElfError, PF_W, PF_X};
use crate::hal::Hal;
use crate::kernel::{Exit, Kernel};
use crate::mem::Paddr;
use crate::thread::Thread;
use core::mem::{offset_of, size_of};
use silica_abi::root::*;
use silica_abi::{Error, ObjectType, PAGE_BITS, Rights, SLOT_BITS};

const CS_BITS: u8 = 12;
const AS_BITS: u8 = 18;
const STACK_BITS: u8 = 16;
const INFO_BITS: u8 = 15;
const _: () = assert!(size_of::<RootInfo>() <= 1 << INFO_BITS);

/// Where the kernel maps the [`RootInfo`].
pub const INFO_VADDR: u64 = 0x7F00_0000_0000;
/// Where the kernel maps the root thread's IPC buffer.
pub const IPCBUF_VADDR: u64 = 0x7F00_0001_0000;
/// Lowest address of the root thread's stack.
pub const STACK_VADDR: u64 = 0x7F00_0010_0000;

/// What the architecture knows that the root task needs.
pub struct RootSpec<'a> {
    /// Physical memory, sorted by base, not overlapping.
    pub regions: &'a [Region],
    /// The root task's ELF image.
    pub elf: &'a [u8],
    /// `e_machine` the image must have.
    pub machine: u16,
    /// Device ranges the kernel keeps (interrupt controllers), sorted.
    pub reserved: &'a [(u64, u64)],
    /// Device memory is handed out up to this address.
    pub device_top: u64,
    /// Whether the machine has I/O ports.
    pub ports: bool,
    /// Ramdisk archive address and length.
    pub ramdisk: [u64; 2],
    /// ACPI RSDP.
    pub rsdp: u64,
    /// Framebuffer, as in [`RootInfo::framebuffer`].
    pub framebuffer: [u64; 5],
    /// As in [`RootInfo::clock`].
    pub clock: [u64; 3],
    /// Wall-clock time, as in [`RootInfo::wall_clock`].
    pub wall_clock: [u64; 2],
    /// Firmware entropy.
    pub entropy: [u8; 32],
    /// Kernel command line.
    pub cmdline: &'a [u8],
}

/// Why the root task could not be created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootError {
    /// No free region is large enough for the root task's objects.
    NoMemory,
    /// The root image is not a loadable ELF file.
    Image(ElfError),
    /// A segment is writable and executable, overlaps another or lies outside user space.
    Segment,
    /// A kernel operation failed while building the task.
    Kernel(Error),
}

impl From<Error> for BootError {
    fn from(e: Error) -> BootError {
        BootError::Kernel(e)
    }
}

impl<H: Hal> Kernel<H> {
    fn boot_retype(&self, us: Paddr, ty: ObjectType, bits: u8, cs: &Cap, index: u64, count: u64) -> Result<(), Error> {
        while self.retype(us, ty, bits as u64, cs, index, count)? != Exit::Done {}
        Ok(())
    }

    /// Create the root task and make its thread runnable. Returns the thread.
    pub fn boot_root(&self, spec: &RootSpec<'_>) -> Result<Paddr, BootError> {
        let elf = Elf::parse(spec.elf, spec.machine).map_err(BootError::Image)?;
        let (_, hi) = elf.span().ok_or(BootError::Image(ElfError::BadSegment))?;
        if hi > INFO_VADDR {
            return Err(BootError::Segment);
        }
        // An Untyped twice the size of what the task needs to start, at least 4 MiB.
        let image: u64 = elf.segments().map(|s| (s.vaddr + s.memsz).div_ceil(4096) - s.vaddr / 4096).sum::<u64>() << PAGE_BITS;
        let need = (1u64 << (CS_BITS + SLOT_BITS)) + (1 << AS_BITS) + image + (1 << STACK_BITS) + (1 << INFO_BITS) + (4 << PAGE_BITS);
        let bits = (64 - (2 * need - 1).leading_zeros()).max(22) as u8;
        let size = 1u64 << bits;
        let base = spec
            .regions
            .iter()
            .filter(|r| r.kind == RegionKind::Usable as u32)
            .filter_map(|r| {
                let b = (r.base.max(1 << 20) + size - 1) & !(size - 1);
                (b.checked_add(size)? <= r.base + r.len).then_some(b)
            })
            .next_back()
            .ok_or(BootError::NoMemory)?;

        // The CapSpace comes first, by hand: until it exists there is no slot
        // to hold the Untyped it is made from.
        let cs_size = 1u64 << (CS_BITS + SLOT_BITS);
        self.phys.zero(base, cs_size);
        let cs = Cap { ty: ObjectType::CapSpace, rights: Rights::ALL, bits: CS_BITS, flags: 0, obj: base, badge: 0 };
        let slot = |i: u64| base + (i << SLOT_BITS);
        let us = slot(SLOT_BOOT_UNTYPED);
        self.insert_root(us, &Cap { ty: ObjectType::Untyped, rights: Rights::ALL, bits, flags: 0, obj: base, badge: cs_size });
        self.slot(us).aux.set(cs_size);
        self.insert_child(us, slot(SLOT_CAPSPACE), &cs)?;

        self.boot_retype(us, ObjectType::Thread, 0, &cs, SLOT_THREAD, 1)?;
        self.boot_retype(us, ObjectType::AddressSpace, AS_BITS, &cs, SLOT_ASPACE, 1)?;
        self.boot_retype(us, ObjectType::Frame, PAGE_BITS, &cs, SLOT_IPCBUF, 1)?;
        let hw = |flags, end| Cap { ty: ObjectType::Hw, rights: Rights::ALL, bits: 0, flags, obj: 0, badge: end };
        if spec.ports {
            self.insert_root(slot(SLOT_PORTS), &hw(FLAG_PORTS, 1 << 16));
        }
        self.insert_root(slot(SLOT_IRQS), &hw(FLAG_IRQS, 256));
        let asp = self.slot(slot(SLOT_ASPACE)).obj.get();

        // Every other piece of memory, as Untyped capabilities.
        let info_slot = SLOT_FIRST_UNTYPED - 1;
        self.boot_retype(us, ObjectType::Frame, INFO_BITS, &cs, info_slot, 1)?;
        let info = self.slot(slot(info_slot)).obj.get();
        let mut count = 0u64;
        let mut emit = |mut a: u64, b: u64, kind: u8| {
            while a < b && count < MAX_UNTYPED as u64 {
                let bits = (a.trailing_zeros().min(63 - (b - a).leading_zeros()) as u8).min(silica_abi::UNTYPED_MAX_BITS);
                let flags = if kind == KIND_RAM { 0 } else { FLAG_DEVICE };
                let cap = Cap { ty: ObjectType::Untyped, rights: Rights::ALL, bits, flags, obj: a, badge: 0 };
                self.insert_root(slot(SLOT_FIRST_UNTYPED + count), &cap);
                let d = UntypedDesc { base: a, bits, kind, reserved: [0; 6] };
                self.phys.write_val(info + (offset_of!(RootInfo, untyped) + count as usize * size_of::<UntypedDesc>()) as u64, &d);
                count += 1;
                a += 1u64 << bits;
            }
        };
        let device = |mut a: u64, b: u64, emit: &mut dyn FnMut(u64, u64, u8)| {
            for &(ra, rb) in spec.reserved {
                if rb <= a || ra >= b {
                    continue;
                }
                emit(a, ra.max(a), KIND_DEVICE);
                a = rb.min(b);
            }
            emit(a, b, KIND_DEVICE);
        };
        let mut cursor = 0u64;
        for r in spec.regions {
            let (a, b) = (r.base, r.base + r.len);
            if a > cursor {
                device(cursor, a, &mut emit);
            }
            cursor = b;
            if r.kind == RegionKind::Usable as u32 || r.kind == RegionKind::Loader as u32 {
                // Page 0 is never ordinary memory: address 0 means "none" in the kernel.
                let a = if a == 0 {
                    emit(0, 4096.min(b), KIND_DEVICE);
                    4096.min(b)
                } else {
                    a
                };
                emit(a, base.clamp(a, b), KIND_RAM);
                emit((base + size).clamp(a, b), b, KIND_RAM);
            } else if r.kind == RegionKind::Ramdisk as u32 {
                emit(a, b, KIND_RAMDISK);
            } else if r.kind == RegionKind::Firmware as u32 {
                device(a, b, &mut emit);
            }
        }
        if spec.device_top > cursor {
            device(cursor, spec.device_top, &mut emit);
        }
        if count == MAX_UNTYPED as u64 {
            crate::klog!(self, "boot: more than {} memory ranges; the rest are not handed to the root task", MAX_UNTYPED);
        }

        // The image, one frame per page.
        let mut next = SLOT_FIRST_UNTYPED + count;
        for seg in elf.segments() {
            if seg.flags & PF_W != 0 && seg.flags & PF_X != 0 {
                return Err(BootError::Segment);
            }
            let rights = Rights(Rights::READ.0 | if seg.flags & PF_W != 0 { 2 } else { 0 } | if seg.flags & PF_X != 0 { 4 } else { 0 });
            let first = seg.vaddr & !0xFFF;
            let pages = (seg.vaddr + seg.memsz - first).div_ceil(4096);
            for p in 0..pages {
                let s = self.slot_in(&cs, next)?;
                self.boot_retype(us, ObjectType::Frame, PAGE_BITS, &cs, next, 1)?;
                next += 1;
                let frame = self.slot(s).obj.get();
                let page = first + p * 4096;
                // The part of the segment's file data that falls in this page.
                let from = page.max(seg.vaddr);
                let off = (from - seg.vaddr) as usize;
                if off < seg.data.len() {
                    let n = (seg.data.len() - off).min((page + 4096 - from) as usize);
                    self.phys.write_bytes(frame + (from - page), &seg.data[off..off + n]);
                }
                self.map(asp, s, page, rights).map_err(|_| BootError::Segment)?;
            }
        }
        let rw = Rights(Rights::READ.0 | Rights::WRITE.0);
        let stack = self.slot_in(&cs, next)?;
        self.boot_retype(us, ObjectType::Frame, STACK_BITS, &cs, next, 1)?;
        next += 1;
        self.map(asp, stack, STACK_VADDR, rw)?;
        self.map(asp, slot(SLOT_IPCBUF), IPCBUF_VADDR, rw)?;
        self.map(asp, slot(info_slot), INFO_VADDR, Rights::READ)?;

        let put = |off: usize, v: &[u64]| {
            for (i, w) in v.iter().enumerate() {
                self.phys.write_val(info + (off + i * 8) as u64, w);
            }
        };
        put(offset_of!(RootInfo, magic), &[ROOT_MAGIC, 1 << CS_BITS, next, count, IPCBUF_VADDR]);
        put(offset_of!(RootInfo, ramdisk), &spec.ramdisk);
        put(offset_of!(RootInfo, rsdp), &[spec.rsdp]);
        put(offset_of!(RootInfo, framebuffer), &spec.framebuffer);
        put(offset_of!(RootInfo, clock), &spec.clock);
        put(offset_of!(RootInfo, wall_clock), &spec.wall_clock);
        self.phys.write_bytes(info + offset_of!(RootInfo, entropy) as u64, &spec.entropy);
        let cmd = &spec.cmdline[..spec.cmdline.len().min(256)];
        put(offset_of!(RootInfo, cmdline_len), &[cmd.len() as u64]);
        self.phys.write_bytes(info + offset_of!(RootInfo, cmdline) as u64, cmd);

        // The thread: its CapSpace, address space and IPC buffer, then go.
        let t = self.slot(slot(SLOT_THREAD)).obj.get();
        let emb = Thread::embedded_slots(t);
        for (src, dst) in [(SLOT_CAPSPACE, emb[0]), (SLOT_ASPACE, emb[1]), (SLOT_IPCBUF, emb[3])] {
            let mut cap = self.slot(slot(src)).cap().ok_or(Error::BadCap)?;
            if cap.ty == ObjectType::Frame {
                cap.badge = 0;
            }
            self.insert_sibling(slot(src), dst, &cap);
        }
        let r = &self.thread(t).regs;
        r[H::IP].set(elf.entry);
        r[H::SP].set(STACK_VADDR + (1 << STACK_BITS));
        r[H::ARG[0]].set(INFO_VADDR);
        r[H::FLAGS].set(H::FLAGS_INIT);
        self.make_runnable(t);
        Ok(t)
    }
}
