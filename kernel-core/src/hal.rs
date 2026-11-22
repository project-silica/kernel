//! What the arch-free kernel needs from an architecture
//! (`spec/ARCHITECTURE.md` section 6, HAL). Implemented once per
//! architecture in `kernel/arch/` and once for the host tests.

use crate::mem::{Paddr, Phys};
use silica_abi::{Error, Rights};

/// Number of words in a thread's saved register frame.
pub const NREGS: usize = 40;

/// Architecture operations and constants.
pub trait Hal {
    /// Frame index of the instruction pointer.
    const IP: usize;
    /// Frame index of the stack pointer.
    const SP: usize;
    /// Frame index of the flags register.
    const FLAGS: usize;
    /// Frame index of the syscall number, and of the status on return.
    const SYS: usize;
    /// Frame indices of the six syscall argument registers.
    const ARG: [usize; 6];
    /// Frame index of a word that is not 0 when a system call saved the
    /// frame (an interrupt or a fault saved it otherwise). The entry marks
    /// it before the processor says it is in the kernel, so another
    /// processor holding this one's lock, and finding its thread in the
    /// kernel, knows whether it is waiting to make a system call.
    const BY_SYSCALL: usize;
    /// Flags a new user thread starts with.
    const FLAGS_INIT: u64;
    /// Length of the syscall instruction, to restart a preempted syscall.
    const SYSCALL_LEN: u64;
    /// One past the highest user virtual address.
    const USER_TOP: u64;

    /// Make a user-supplied instruction pointer safe to return to.
    fn user_ip_ok(ip: u64) -> bool {
        ip < Self::USER_TOP
    }

    /// A page-table entry pointing to the next-level table at `table`.
    fn pte_table(table: Paddr) -> u64;
    /// A leaf entry mapping `frame` at `level` (0: 4 KiB, 1: 2 MiB).
    fn pte_leaf(frame: Paddr, rights: Rights, level: u8, device: bool) -> u64;
    /// Whether the entry maps or points to anything.
    fn pte_present(pte: u64) -> bool;
    /// Whether a present entry at `level` is a leaf.
    fn pte_is_leaf(pte: u64, level: u8) -> bool;
    /// The physical address in a present entry.
    fn pte_addr(pte: u64) -> Paddr;

    /// An entry of a device address space's table (the IOMMU's format)
    /// pointing to the next-level table at `table`. Entries of device
    /// address spaces are read with [`Hal::pte_present`],
    /// [`Hal::pte_is_leaf`] and [`Hal::pte_addr`] like a processor's.
    fn io_pte_table(table: Paddr) -> u64 {
        Self::pte_table(table)
    }
    /// A leaf entry of a device address space: devices may read `frame`,
    /// and write it with `Rights::WRITE`.
    fn io_pte_leaf(frame: Paddr, rights: Rights, level: u8) -> u64 {
        Self::pte_leaf(frame, rights, level, false)
    }
    /// One past the highest address a device address space translates; 0
    /// when the machine has no IOMMU the kernel drives.
    fn io_top(&self) -> u64 {
        0
    }
    /// Whether a device address space maps 2 MiB with one entry.
    fn io_large(&self) -> bool {
        false
    }
    /// From now on the device with identity `id` reaches memory through the
    /// device address space whose root table is `root` (0: reaches nothing),
    /// and nothing it cached of its previous one remains. Returns the root
    /// it was attached to before (0: none). Holding every lock.
    fn io_attach(&self, _phys: &Phys, _id: u64, _root: Paddr) -> Result<Paddr, Error> {
        Err(Error::Unsupported)
    }
    /// Mappings of the device address space `root` were removed: its
    /// devices' cached translations are dropped before this returns.
    fn io_flush(&self, _phys: &Phys, _root: Paddr) {}
    /// A frame was mapped in the device address space `root`, which has
    /// devices attached (an IOMMU that caches missing translations must be
    /// told).
    fn io_mapped(&self, _phys: &Phys, _root: Paddr) {}
    /// The device address space `root` is going away: every device attached
    /// to it reaches nothing from now on.
    fn io_release(&self, _phys: &Phys, _root: Paddr) {}
    /// The first interrupt line that is a message, not a wire.
    fn msi_first(&self) -> usize {
        usize::MAX
    }
    /// Deliver message line `line` (at least [`Hal::msi_first`]) to
    /// processor `cpu`, raised only by the device with identity `source`
    /// where the machine can enforce that. Returns the message's address
    /// and data for the device's registers; `None` when the machine cannot.
    fn msi_route(&self, _line: usize, _cpu: usize, _source: u64) -> Option<(u64, u64)> {
        None
    }

    /// Fill a new root table with the kernel's half of the address space.
    fn init_root(&self, phys: &Phys, root: Paddr);
    /// Give a new thread's extended-state area its initial contents. The
    /// area is the part of the Thread object from [`crate::thread::FPU_OFFSET`].
    fn init_thread(&self, phys: &Phys, thread: Paddr);
    /// Make `to` the thread this processor returns to: save `from`'s extended
    /// state (when not 0), load `to`'s, and switch to address-space root
    /// `root` (0: kernel mappings only).
    fn switch_thread(&self, phys: &Phys, from: Paddr, to: Paddr, root: Paddr);
    /// Leave address-space root `root` if this processor has it loaded.
    fn leave_aspace(&self, root: Paddr);
    /// Drop cached translations for `[vaddr, vaddr + len)` in `root`.
    fn tlb_flush(&self, root: Paddr, vaddr: u64, len: u64);

    /// Index of the processor running this code.
    fn cpu(&self) -> usize;
    /// [`crate::smp::cpu_slot`] of that index, which an architecture keeps
    /// beside it: finding this processor's state is then one load.
    fn cpu_slot(&self) -> usize {
        crate::smp::cpu_slot(self.cpu())
    }
    /// Interrupt processor `cpu` so that it enters the kernel. With `wait`,
    /// return only when it is no longer running user code: it is in the
    /// kernel, has saved its thread's registers, and will take its lock
    /// (which the caller holds) before doing anything else.
    fn kick(&self, cpu: usize, wait: bool);
    /// On this processor, drop every cached user translation and load the
    /// kernel-only address space. Another processor changed or destroyed
    /// mappings this one may have cached.
    fn tlb_reset(&self);
    /// Processor `cpu` (whose lock the caller holds) does not have `root`
    /// loaded: make its next load of `root` start with nothing cached.
    fn tlb_forget(&self, root: Paddr, cpu: usize);

    /// Monotonic nanoseconds since boot.
    fn now(&self) -> u64;
    /// Ask for a timer interrupt at `deadline` (`u64::MAX`: none).
    fn set_timer(&self, deadline: u64);
    /// Whether an interrupt is waiting to be taken. Long operations poll this
    /// and restart later when it is true.
    fn irq_pending(&self) -> bool;
    /// This processor is about to wait for an interrupt and let go of its
    /// lock: from now until `idle` returns it counts as idle, not as in the
    /// kernel, so a `kick` interrupts it.
    fn idle_begin(&self);
    /// Arm this processor's timer for `until` (monotonic nanoseconds;
    /// `u64::MAX`: no timer) and wait for an interrupt with interrupts
    /// enabled; return after its handler has run. When it was the timer,
    /// return no earlier than `until`: an architecture may arm the timer
    /// ahead of `until` by the time this processor takes to come out of the
    /// wait, and spin for the rest, so a deadline is met on time rather
    /// than late by the wake-up latency. Called without this processor's
    /// lock, so the handler must not touch kernel state.
    fn idle(&self, until: u64);
    /// Route interrupt line `line` to processor `cpu`, with the given
    /// trigger and polarity, and let it interrupt. `false` when the machine
    /// has no such line.
    fn irq_route(&self, line: usize, cpu: usize, level: bool, active_low: bool) -> bool;
    /// Hold interrupt line `line` off, or let it interrupt again.
    fn irq_mask(&self, line: usize, masked: bool);
    /// Port I/O (x86_64). `width` is 1, 2 or 4.
    fn port_io(&self, port: u16, width: u8, write: bool, value: u32) -> u32;
    /// Write to the kernel's console.
    fn console(&self, bytes: &[u8]);
    /// Turn the machine off, telling a test harness `status` (0: success).
    /// Returns only where that is not possible.
    fn power_off(&self, status: u64);
}
