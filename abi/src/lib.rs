//! The kernel ABI: everything the kernel and `libsys` must agree on.
//!
//! Applications never use these numbers directly (`spec/ARCHITECTURE.md`
//! section 4); they call `libsys`. The numbers may change between kernel
//! versions, together with the `libsys` that ships beside the kernel.
#![no_std]
#![forbid(unsafe_code)]

pub mod root;

/// Kernel operations. Every operation is its own syscall: there is no
/// multiplexed "invoke", so this list is the kernel's whole surface.
///
/// Register convention (x86_64 names; other architectures map the same
/// roles): number in `rax`, arguments `a0..a5` in `rdi rsi rdx r10 r8 r9`.
/// On return `rax` holds an [`Error`] code (0 is success) and results come
/// back in the argument registers as documented per call.
///
/// A system call preserves what a function call preserves: `rbx`, `rbp`,
/// `rsp`, `r12`..`r15`, MXCSR and the x87 control word. Vector and x87 data
/// registers are not preserved (they come back cleared or unchanged, never
/// holding another thread's data).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Syscall {
    /// `a0` endpoint (its low 32 bits), `a1` [`MsgInfo`], `a2..a5` message
    /// words 0..3. Sends and blocks until the receiver replies. Returns the
    /// reply the same way. Bits 32 to 62 of `a0` ([`CALL_TIMEOUT_SHIFT`]) are
    /// the longest the whole call may take, in microseconds (0: no limit):
    /// past it the call fails with `Timeout`, whether it still waits for a
    /// receiver or was received and waits for its answer (which then goes
    /// nowhere, as an answer to a caller that has gone does). How a program
    /// calls one it trusts less than itself without being held by it.
    Call = 0,
    /// `a0` endpoint or notification, `a1` [`MsgInfo`], `a2..a5` words,
    /// `a0` bit 63 set: do not block. On a notification, ORs the capability's
    /// badge into the pending word (bit 0 for a capability without a badge),
    /// or answers the call parked there (see `Recv`).
    Send = 1,
    /// `a0` endpoint or notification (its low 32 bits), `a1` deadline in
    /// monotonic nanoseconds (0 or `u64::MAX`: wait for ever, 1: do not
    /// block), `a2` for an endpoint: a notification to park the next call on
    /// (0: none). Returns `a0` badge, `a1` [`MsgInfo`], `a2..a5` words.
    /// For a notification `a0` is the pending word, cleared by the read.
    ///
    /// Receiving on an endpoint can listen on a notification as well: bits
    /// 32 to 62 of `a0` name it (0: none; [`LISTEN_SHIFT`]). If it has been
    /// signalled, or is signalled while the thread waits, the receive
    /// returns its pending word in `a0` (cleared by the read) with `a1`
    /// [`MsgInfo::NOTIFIED`], which no message carries. So one thread, in
    /// one wait, serves callers and hears interrupts and wake-ups: the loop
    /// of every driver and server. One thread listens on a notification at
    /// a time, and none while another waits on it directly (`InUse`). When
    /// the notification's capability leaves its slot, or the notification
    /// is destroyed, while the thread waits, the receive fails with `Gone`.
    ///
    /// A call received with a parking notification waits there instead of
    /// on the receiving thread: any thread answers it by a `Send` to that
    /// notification, at any later time, so a server can take further calls
    /// first. One call parks on one notification; while it does, signals to
    /// the notification accumulate and waiting on it is refused (`InUse`).
    Recv = 2,
    /// Reply to the caller received last (`a1` [`MsgInfo`], `a2..a5` words),
    /// then, when the low 32 bits of `a0` are not 0, `Recv` on `a0` (an
    /// endpoint, with a notification to listen on in its upper bits as for
    /// `Recv`) with no deadline.
    ReplyRecv = 3,
    /// Give up the processor to another runnable thread.
    Yield = 4,
    /// `a0` untyped, `a1` [`ObjectType`], `a2` size in bits (ignored for
    /// fixed-size types), `a3` destination CapSpace, `a4` first destination
    /// slot, `a5` count. Creates `count` objects in consecutive empty slots.
    Retype = 5,
    /// `a0` destination CapSpace, `a1` destination slot, `a2` source slot
    /// (in the caller's CapSpace), `a3` rights mask, `a4` badge (0 keeps the
    /// source's), `a5` 0 to derive, 1 to duplicate. A derived capability is
    /// a child of the source: revoking or deleting the source deletes it. A
    /// duplicate is independent of the source and shares the source's parent.
    CapCopy = 6,
    /// `a0` destination CapSpace, `a1` destination slot, `a2` source slot.
    /// The capability keeps its place in the derivation tree.
    CapMove = 7,
    /// `a0` slot. Deletes the capability and every capability derived from
    /// it; an object is destroyed with its last capability. A large CapSpace
    /// is emptied in pieces between pending interrupts (the call is made
    /// again until it is done, invisibly to the caller); what is left when
    /// its caller is stopped is finished by the next system call that takes
    /// every lock, before any memory can be reused.
    CapDelete = 8,
    /// `a0` slot. Deletes every capability derived from this one.
    CapRevoke = 9,
    /// `a0` address space, `a1` frame, `a2` virtual address, `a3` rights.
    /// In an address space made for devices ([`ASPACE_IO`]) the address is
    /// the one the attached devices use to reach the frame, `READ` lets them
    /// read it and `WRITE` write it; it must lie below the machine's
    /// `RootInfo::dma_top`.
    Map = 10,
    /// `a0` frame: removes this capability's mapping. Or `a0` an address
    /// space (write), `a1` an address, `a2` a length in bytes: removes
    /// every mapping in that space that overlaps the range, whole, whoever's
    /// frame it is; returns how many in `a0`. What a program does before
    /// it maps something new where lent memory was: a frame its lender
    /// took back is unmapped already, but a page its supervisor mapped in
    /// its place (`silica_rt::mem`) is not.
    Unmap = 11,
    /// `a0` thread (write), `a1` CapSpace, `a2` address space, `a3` fault
    /// endpoint, `a4` IPC buffer frame. [`KEEP`] leaves a field unchanged, 0
    /// clears it. Each capability must carry the use the thread makes of
    /// it: read and write on the CapSpace, the address space and the IPC
    /// buffer (running in them is full use of their slots and mappings),
    /// write on the fault endpoint (faults are sent through it); `NoRights`
    /// otherwise, with nothing changed.
    ThreadConfigure = 12,
    /// `a0` thread, `a1` instruction pointer, `a2` stack pointer, `a3`
    /// first argument, `a4` not 0: make the thread runnable.
    ThreadStart = 13,
    /// `a0` thread, `a1` 0 suspend, 1 resume, 2 move to processor `a2`
    /// (it carries on there with what it was doing: a call waiting for its
    /// reply is answered there, a send or receive not yet delivered is made
    /// again there, a suspended thread stays suspended; `InUse` while it is
    /// bound to a deadline context), 3 bind it to the SchedContext `a2`
    /// (read; 0: unbind, back to the default fair share; a deadline context
    /// must be on the thread's processor).
    ThreadControl = 14,
    /// `a0` port capability, `a1` port, `a2` width in bytes (1, 2, 4) with
    /// bit 8 set for a write, `a3` value. Returns `a0` value read.
    PortIo = 15,
    /// `a0` interrupt-line capability, `a1` line, `a2` notification (0:
    /// unbind), `a3` flags ([`IRQ_LEVEL`], [`IRQ_ACTIVE_LOW`], [`IRQ_MSI`]).
    /// From now on an interrupt on the line signals the notification with
    /// the capability's badge, on the processor of the calling thread. A
    /// level-triggered line is held off after each interrupt until `IrqAck`.
    ///
    /// With [`IRQ_MSI`] the line is not a wire: a device raises it by
    /// writing a message, and the call returns that message (`a0` the
    /// address, `a1` the data) for the driver to put in its device's MSI or
    /// MSI-X registers. `a4` is then a device-identity capability and `a5`
    /// the identity of the one device allowed to raise the line (enforced
    /// where the machine remaps interrupts). `BadArg` for a line that is a
    /// wire, `Unsupported` where the machine has no message interrupts.
    ///
    /// A bind that fails changes nothing: the line keeps its binding.
    IrqBind = 16,
    /// `a0` interrupt-line capability, `a1` line. The device has been dealt
    /// with: let the line interrupt again.
    IrqAck = 17,
    /// Look at kernel state through a capability (which needs its read
    /// right: `NoRights` without it), changing nothing.
    /// `a0` a CapSpace and `a1` a slot index: what that slot holds. Returns
    /// `a0` object type (0: empty), `a1` rights, `a2` size in bits, `a3` the
    /// object's identity (equal for capabilities to one object), `a4` badge,
    /// `a5` depth in the derivation tree.
    /// `a0` a Thread and `a1` 0: returns `a0` state (the `THREAD_` values),
    /// `a1` processor, `a2` instruction pointer, `a3` stack pointer, `a4`
    /// the identity of what it waits on.
    /// `a0` a device-identity capability and `a1` a Frame, on a machine
    /// that cannot confine DMA (`RootInfo::dma_top` is 0): returns `a0` the
    /// frame's physical address, which is the address a device reaches it
    /// at. `Unsupported` on a machine that confines DMA: there a device's
    /// addresses are the ones its driver mapped, and physical addresses are
    /// nobody's business.
    /// `a0` a SchedContext: returns `a0` class, `a1` weight (fair) or
    /// budget (deadline), `a2` period, `a3` budget left this period, `a4`
    /// nanoseconds charged to it so far, `a5` processor (deadline).
    /// `a0` a CapSpace, `a1` a slot index and `a2` 1: the same about the
    /// thread in that slot (`WrongType` if it holds something else), so a
    /// holder of a CapSpace sees every thread it holds without copying.
    /// (`a2` must otherwise be 0.)
    Inspect = 18,
    /// `a0` console capability (the root task's [`root::SLOT_CONSOLE`] or
    /// one derived from it), `a1` number of bytes (at most 32), with
    /// [`CONSOLE_MORE`] when more of the same text follows, `a2` to `a5`
    /// the bytes, eight to a register, lowest address in the low byte.
    /// Writes them to the console the kernel logs to. A piece with more to
    /// follow is held (up to 192 bytes) and written out with the piece
    /// that ends the text, so text of up to 224 bytes reaches the console
    /// whole: nobody else's (a kernel line, another program's) comes in
    /// between, and a writer that finds another's line unfinished starts a
    /// fresh one. For bring-up and panics; the logging service replaces it
    /// for everything else.
    Console = 19,
    /// `a0` power capability (the root task's [`root::SLOT_POWER`]), `a1`
    /// status (0: success), `a2` 0 to turn the machine off, 1 to reset it.
    /// A test harness (QEMU's debug-exit device on x86_64, semihosting on
    /// aarch64 with `exit=semihosting` on the kernel command line) is told
    /// the status instead. Turning off is ACPI S5 on x86_64 (the FADT's
    /// control registers and the DSDT's `\_S5` values) and PSCI SYSTEM_OFF
    /// on aarch64; resetting is the FADT's reset register on x86_64 and PSCI
    /// SYSTEM_RESET on aarch64. Returns `Unsupported` where the firmware
    /// describes no way to do it. Nothing is stopped or flushed first: that
    /// is the caller's business (the shutdown service's).
    PowerOff = 20,
    /// `a0` SchedContext (write), `a1` class. [`SCHED_FAIR`]: `a2` weight
    /// (1 to [`SCHED_MAX_WEIGHT`]; a thread with none weighs
    /// [`SCHED_DEFAULT_WEIGHT`]): bound threads share the processor in
    /// proportion to their weights. [`SCHED_DEADLINE`]: `a2` budget and `a3`
    /// period in nanoseconds, `a4` processor: bound threads run before any
    /// fair work, earliest deadline first, for `a2` of every `a3`, and are
    /// held back once it is spent until the next period. `Overload` when the
    /// processor's reservations would take more than 90% of it. `InUse` to
    /// change the class, or a deadline context's processor, while threads
    /// are bound to it.
    ///
    /// Processor time is authority, as memory is: `a5` is a processor-time
    /// capability (write; [`root::SLOT_SCHED`] or one narrowed from it to
    /// a range of processors), or 0 for none. Without one a context may be
    /// configured fair with at most [`SCHED_DEFAULT_WEIGHT`]: memory buys a
    /// share no better than a thread without a context has. A greater
    /// weight needs one covering every processor (a fair thread runs
    /// wherever it is moved); a reservation, one covering `a4`. `NoRights`
    /// otherwise, `WrongType` for a capability that is not processor time.
    SchedConfigure = 21,
    /// `a0` device-identity capability, `a1` a device's identity (its PCI
    /// segment, bus, device and function: `segment << 16 | bus << 8 |
    /// device << 3 | function`), `a2` an address space made for devices
    /// ([`ASPACE_IO`]; write), or 0. From now on the device reaches exactly
    /// the frames mapped in that address space, at the addresses they are
    /// mapped at, and nothing else; with 0 it reaches nothing, which is how
    /// every device starts and what it returns to when the address space is
    /// destroyed. `Unsupported` on a machine that cannot confine DMA
    /// (`RootInfo::dma_top` is 0) or for a device its IOMMU does not cover;
    /// `NoMemory` when the kernel's device tables are full.
    DeviceAttach = 22,
    /// Signal the notification `a3` (OR the capability's badge into its
    /// word, as `Send` does; a call parked on it is not answered: a signal
    /// carries no message, and accumulates for whoever answers), then `Recv` with
    /// `a0..a2` as `Recv` takes them: the wake-up a thread gives its peer
    /// and its own wait for the answer, in one kernel entry, so a peer on
    /// the same processor runs at once. A signal that cannot be made (not a
    /// notification, not writable) fails the call before anything else; once
    /// made it stands, and a receive made again (preempted, restarted) is a
    /// plain `Recv`. A call of its own rather than an argument of `Recv`, so
    /// that every other receive pays nothing for it.
    SignalRecv = 23,
    /// `a0` interrupt-line capability, `a1` a message-signalled line bound
    /// with `IrqBind`, `a2` a Frame (readable; 0: stop watching), `a3` the
    /// offset of a 32-bit word in it (a multiple of 4), `a4` a mask (its low
    /// 32 bits; not 0). Says that the line's news is that word: the device
    /// changes `word & mask` whenever it raises the line, and the driver
    /// takes all of it each time it is signalled (virtio's used index, which
    /// the device advances before it interrupts). From then on a processor
    /// with nothing to run, watching for work before it halts, watches the
    /// word as well, and a change is delivered at once as the interrupt
    /// would be: the line's notification signalled with its badge. The
    /// interrupt that follows is delivered only if the word changed again
    /// since: one news, one signal. A processor that halts (its watch ended
    /// with nothing) is woken by the interrupt as before, so nothing waits
    /// longer than without the watch. The watch lasts while the frame's
    /// capability stays in slot `a2`: once that slot holds anything else it
    /// is inert (every interrupt delivered, nothing read). Rebinding the
    /// line, or destroying the notification or the CapSpace, ends it. The word is watched by
    /// the processor of the calling thread: call it from the thread that
    /// bound the line, which is where the line is delivered. `BadArg` for a line that
    /// is a wire or not bound, a word outside the frame, or a mask of 0;
    /// `NoMemory` when the kernel's watches ([`IRQ_WATCHES`]) are all in use.
    IrqWatch = 24,
    /// `a0` interrupt-line capability, `a1` a line watched with `IrqWatch`,
    /// `a2` a Frame (writable), `a3` the offset of a 32-bit word in it (a
    /// multiple of 4), `a4` bits of that word (not 0) that tell the device
    /// not to interrupt (virtio's `VRING_AVAIL_F_NO_INTERRUPT`). From then
    /// on those bits are the kernel's to write, and the program must not
    /// store to them (it may store to the word's other bits): they are set
    /// while a processor watches the line's word and clear at every other
    /// time, so the device interrupts exactly when nobody is watching for
    /// it, and is told before the watch ends. Only for a device that writes
    /// its news before it reads these bits and interrupts whenever they are
    /// clear (virtio without event indices). Lasts while the frame's
    /// capability stays in slot `a2` (keep it there: a device left quiet
    /// interrupts nobody), and ends with the watch, its bits left clear. `BadArg` for a line not watched, a word outside the
    /// frame, or no bits.
    IrqQuiet = 25,
    /// `a0` a slot holding a capability to watch (not a Frame), `a1` a
    /// Notification (write, and no badge of its own), `a2` an empty slot,
    /// `a3` bits (not 0). Puts in `a2` a notice: a capability derived from
    /// `a1` whose badge is `a3`. When the capability in `a0` is deleted, by
    /// whatever means (its object's memory taken back because its owner
    /// ended, a revocation of what it derives from, or the holder's own
    /// `CapDelete`), the notification is signalled with `a3` once and the
    /// notice is deleted: how a program learns at once that a peer it holds
    /// a capability to has gone, with nothing to poll. Deleting the notice
    /// (or the capability it derives from) ends the watch without a signal.
    /// `CapMove` of either slot keeps the watch. One watch per capability:
    /// `InUse` for one already watched or a notice; `WrongType` for a Frame
    /// or a notification that is not one; `BadArg` for bits of 0 or a
    /// badged notification; `SlotFull` when `a2` is not empty.
    CapNotice = 26,
}

/// Interrupt lines that can be watched (`IrqWatch`) at once.
pub const IRQ_WATCHES: usize = 16;

impl Syscall {
    /// One past the highest syscall number.
    pub const COUNT: u64 = 27;

    /// Decode a syscall number.
    pub const fn from_u64(n: u64) -> Option<Syscall> {
        Some(match n {
            0 => Syscall::Call,
            1 => Syscall::Send,
            2 => Syscall::Recv,
            3 => Syscall::ReplyRecv,
            4 => Syscall::Yield,
            5 => Syscall::Retype,
            6 => Syscall::CapCopy,
            7 => Syscall::CapMove,
            8 => Syscall::CapDelete,
            9 => Syscall::CapRevoke,
            10 => Syscall::Map,
            11 => Syscall::Unmap,
            12 => Syscall::ThreadConfigure,
            13 => Syscall::ThreadStart,
            14 => Syscall::ThreadControl,
            15 => Syscall::PortIo,
            16 => Syscall::IrqBind,
            17 => Syscall::IrqAck,
            18 => Syscall::Inspect,
            19 => Syscall::Console,
            20 => Syscall::PowerOff,
            21 => Syscall::SchedConfigure,
            22 => Syscall::DeviceAttach,
            23 => Syscall::SignalRecv,
            24 => Syscall::IrqWatch,
            25 => Syscall::IrqQuiet,
            26 => Syscall::CapNotice,
            _ => return None,
        })
    }
}

/// `IrqBind` flag: the line is level-triggered (default: edge).
pub const IRQ_LEVEL: u64 = 1;
/// `IrqBind` flag: the line is active low (default: active high).
pub const IRQ_ACTIVE_LOW: u64 = 2;
/// `IrqBind` flag: a message-signalled interrupt (MSI, MSI-X).
pub const IRQ_MSI: u64 = 4;
/// `Retype` of an AddressSpace, added to the size in bits: an address space
/// for devices. Frames are mapped in it for the devices attached to it
/// (`DeviceAttach`) to reach by DMA; no thread can run in it.
pub const ASPACE_IO: u64 = 1 << 8;
/// `Retype` of an AddressSpace, added to the size in bits: an address space
/// isolated from the others against speculative execution. Whenever a
/// processor switches into or out of it, the indirect branch predictors are
/// flushed (IBPB, or the firmware's equivalent) and the return stack
/// refilled, and while it runs the processor's sibling thread cannot steer
/// its branches (STIBP) nor its loads bypass earlier stores (SSBD). For
/// programs that hold secrets (keys, credentials); it costs every switch
/// into and out of them a few microseconds, and others nothing. Not with
/// [`ASPACE_IO`].
pub const ASPACE_ISOLATED: u64 = 1 << 9;
/// `Console`: more of the same text follows this piece (bit 63 of `a1`).
pub const CONSOLE_MORE: u64 = 1 << 63;

/// Number of interrupt lines the kernel can route.
pub const IRQ_LINES: u64 = 192;

/// `Inspect` thread state: not runnable (new, suspended, or faulted with nobody to tell).
pub const THREAD_INACTIVE: u64 = 0;
/// `Inspect` thread state: running or ready to run.
pub const THREAD_RUNNABLE: u64 = 1;
/// `Inspect` thread state: waiting to send.
pub const THREAD_SENDING: u64 = 2;
/// `Inspect` thread state: waiting to receive.
pub const THREAD_RECEIVING: u64 = 3;
/// `Inspect` thread state: waiting for the answer to a call (or to a fault).
pub const THREAD_REPLY: u64 = 4;
/// `Inspect` thread state: waiting to make a system call again that would
/// take every processor's lock, while its processor has used up its share
/// of the machine's time spent so (the kernel's `Kernel::throttled`).
pub const THREAD_THROTTLED: u64 = 5;

/// `Recv`, `ReplyRecv`: the notification to listen on while receiving on an
/// endpoint is named in `a0` from this bit up (31 bits).
pub const LISTEN_SHIFT: u32 = 32;
/// `ThreadConfigure`: leave this field as it is.
pub const KEEP: u64 = u64::MAX;
/// `Send`: flag in `a0` asking not to block.
pub const NONBLOCK: u64 = 1 << 63;
/// `Call`: the call's time limit, in microseconds, is in `a0` from this
/// bit up (31 bits, about 35 minutes; 0: none).
pub const CALL_TIMEOUT_SHIFT: u32 = 32;
/// `Call`: the bits of `a0` that name the endpoint.
pub const CALL_SLOT: u64 = (1 << CALL_TIMEOUT_SHIFT) - 1;

/// The kernel object types (`spec/ARCHITECTURE.md` section 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjectType {
    /// Memory not yet given a type. Move-only: split it by retyping.
    Untyped = 1,
    /// A table of capability slots; slot 0 is always empty.
    CapSpace = 2,
    /// A thread of execution.
    Thread = 3,
    /// A virtual address space with the memory for its page tables.
    AddressSpace = 4,
    /// Mappable memory, any power-of-two size from one page up.
    Frame = 5,
    /// Synchronous IPC rendezvous point.
    Endpoint = 6,
    /// Asynchronous signal word; also carries interrupts.
    Notification = 7,
    /// Processor time: a weighted fair share or a deadline reservation,
    /// which threads bound to it run on (`SchedConfigure`, `ThreadControl`).
    SchedContext = 8,
    /// A range of hardware resources that are not memory: I/O ports,
    /// interrupt lines or device identities. Created at boot, narrowed by
    /// `CapCopy`.
    Hw = 9,
    /// A virtual CPU (M9).
    Vcpu = 10,
}

impl ObjectType {
    /// Log2 of the bytes one object requires for a Retype with this size
    /// argument. CapSpace sizes count slots. Other variable-size objects
    /// count bytes. Fixed-size objects ignore the size argument.
    pub fn storage_bits(self, bits: u64) -> Result<u8, Error> {
        match self {
            Self::Untyped if (SLOT_BITS as u64..=UNTYPED_MAX_BITS as u64).contains(&bits) => Ok(bits as u8),
            Self::Frame if (PAGE_BITS as u64..=UNTYPED_MAX_BITS as u64).contains(&bits) => Ok(bits as u8),
            Self::CapSpace if (1..=32).contains(&bits) => Ok(bits as u8 + SLOT_BITS),
            Self::AddressSpace if (ASPACE_MIN_BITS as u64..=30).contains(&(bits & !(ASPACE_IO | ASPACE_ISOLATED))) => Ok(bits as u8),
            Self::Thread => Ok(THREAD_BITS),
            Self::Endpoint | Self::Notification => Ok(ENDPOINT_BITS),
            Self::SchedContext => Ok(SCHED_BITS),
            Self::Vcpu => Err(Error::Unsupported),
            _ => Err(Error::BadArg),
        }
    }

    /// Decode an object type number.
    pub const fn from_u64(n: u64) -> Option<ObjectType> {
        Some(match n {
            1 => ObjectType::Untyped,
            2 => ObjectType::CapSpace,
            3 => ObjectType::Thread,
            4 => ObjectType::AddressSpace,
            5 => ObjectType::Frame,
            6 => ObjectType::Endpoint,
            7 => ObjectType::Notification,
            8 => ObjectType::SchedContext,
            9 => ObjectType::Hw,
            10 => ObjectType::Vcpu,
            _ => return None,
        })
    }
}

/// Smallest page and Frame: 4 KiB.
pub const PAGE_BITS: u8 = 12;
/// A large page: 2 MiB. A Frame this size or larger is mapped with large
/// pages and its virtual address must be aligned to this.
pub const LARGE_BITS: u8 = 21;
/// Size of one capability slot in bits (64 bytes).
pub const SLOT_BITS: u8 = 6;
/// Size of a Thread object in bits.
pub const THREAD_BITS: u8 = 12;
/// Size of an Endpoint or Notification in bits.
pub const ENDPOINT_BITS: u8 = 6;
/// Size in bits of a SchedContext object.
pub const SCHED_BITS: u8 = 7;
/// `SchedConfigure` class: weighted fair share.
pub const SCHED_FAIR: u64 = 0;
/// `SchedConfigure` class: a deadline reservation.
pub const SCHED_DEADLINE: u64 = 1;
/// The weight of a thread with no scheduling context, and of a new one.
pub const SCHED_DEFAULT_WEIGHT: u64 = 1024;
/// The largest weight.
pub const SCHED_MAX_WEIGHT: u64 = 1 << 20;
/// Smallest AddressSpace: a header page and the root table. Each further
/// page is one page table.
pub const ASPACE_MIN_BITS: u8 = 13;
/// Largest single Untyped.
pub const UNTYPED_MAX_BITS: u8 = 47;
/// Most objects one `Retype` creates.
pub const RETYPE_MAX: u64 = 256;
/// Most page-table entries one `Map` of a frame writes (and its `Unmap`
/// clears): a frame that would need more, because only small pages are
/// available where it is mapped (128 MiB in 4 KiB pages; 64 GiB in 2 MiB
/// pages), is refused with `BadArg`. The bound is what keeps one mapping
/// from holding the machine.
pub const MAP_MAX_ENTRIES: u64 = 1 << 15;

/// What a capability lets its holder do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rights(pub u8);

impl Rights {
    /// Receive from an endpoint or notification; map a frame readable.
    pub const READ: Rights = Rights(1);
    /// Send to an endpoint or notification; map a frame writable.
    pub const WRITE: Rights = Rights(2);
    /// Map a frame executable.
    pub const EXEC: Rights = Rights(4);
    /// Pass capabilities through an endpoint.
    pub const GRANT: Rights = Rights(8);
    /// Everything.
    pub const ALL: Rights = Rights(15);
    /// Nothing.
    pub const NONE: Rights = Rights(0);

    /// Whether every right in `other` is present.
    pub const fn has(self, other: Rights) -> bool {
        self.0 & other.0 == other.0
    }
}

/// Why a kernel operation failed. Returned in the syscall status register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Error {
    /// The syscall number names no operation.
    BadSyscall = 1,
    /// A capability argument names an empty slot or a slot outside the CapSpace.
    BadCap = 2,
    /// The capability is of the wrong type for this operation.
    WrongType = 3,
    /// The capability lacks a right the operation needs.
    NoRights = 4,
    /// An argument is out of range, misaligned or inconsistent.
    BadArg = 5,
    /// The Untyped (or the address space's table pool) has too little room.
    NoMemory = 6,
    /// A destination slot already holds a capability.
    SlotFull = 7,
    /// The frame capability is already mapped; copy it to map it twice.
    Mapped = 8,
    /// Part of the virtual range is already in use.
    Overlap = 9,
    /// The operation would have to wait and was asked not to.
    WouldBlock = 10,
    /// The deadline passed.
    Timeout = 11,
    /// The thread was waiting and can no longer be served: the capability
    /// it sent, called or received through left its slot (deleted, revoked
    /// or moved), the object was destroyed, or the thread that owed it a
    /// reply was destroyed or took another call without answering.
    Gone = 12,
    /// There is no caller to reply to.
    NoCaller = 13,
    /// The Untyped still has objects made from it (it cannot be moved or
    /// retyped as a whole), or a capability that must be unique was copied.
    InUse = 14,
    /// The operation is not available on this machine or in this build.
    Unsupported = 15,
    /// The capability has been derived through too many generations.
    TooDeep = 16,
    /// The processor's time is already promised: a deadline reservation
    /// would take more than the share deadline work may have.
    Overload = 17,
}

impl Error {
    /// Decode a status register value; 0 is success.
    pub const fn result(code: u64) -> Result<(), Error> {
        Err(match code {
            0 => return Ok(()),
            2 => Error::BadCap,
            3 => Error::WrongType,
            4 => Error::NoRights,
            5 => Error::BadArg,
            6 => Error::NoMemory,
            7 => Error::SlotFull,
            8 => Error::Mapped,
            9 => Error::Overlap,
            10 => Error::WouldBlock,
            11 => Error::Timeout,
            12 => Error::Gone,
            13 => Error::NoCaller,
            14 => Error::InUse,
            15 => Error::Unsupported,
            16 => Error::TooDeep,
            17 => Error::Overload,
            _ => Error::BadSyscall,
        })
    }
}

/// The message descriptor word: bits 0..9 the number of message words
/// (at most [`MsgInfo::MAX_WORDS`]), bits 10..12 the number of capabilities
/// (at most [`MsgInfo::MAX_CAPS`]), bits 16..63 a label the kernel does not
/// interpret. Words 0..3 travel in registers; the rest in the IPC buffer.
/// A message is at most 4000 bytes, which is what an IDL message may be
/// (`spec/IDL.md` section 3); anything larger belongs in a shared Frame.
///
/// The IPC buffer is one page: message words 0..499 at byte 0; at
/// [`IPC_CAPS`] the capabilities a sender passes (a slot index with one of
/// the `IPC_CAP_` modes in the top four bits); at [`IPC_RECV`] the slots, in
/// the receiver's own CapSpace, that capabilities it is sent are put in.
/// Sending capabilities needs the `GRANT` right on the endpoint capability;
/// a reply may always carry them. Capabilities are passed in order and the
/// first that cannot be (no such capability, receiving slot missing or
/// full) ends the transfer; the receiver's descriptor says how many arrived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsgInfo(pub u64);

impl MsgInfo {
    /// What a receive that listened on a notification returns instead of a
    /// message's descriptor when the notification was signalled: bit 13,
    /// which the kernel clears in every descriptor it delivers for a
    /// sender, so no sender can make a message look like one.
    pub const NOTIFIED: MsgInfo = MsgInfo(1 << 13);
    /// Most words in one message.
    pub const MAX_WORDS: usize = 500;
    /// Most capabilities in one message.
    pub const MAX_CAPS: usize = 4;
    /// Words carried in registers.
    pub const REG_WORDS: usize = 4;

    /// A descriptor with `label` and `words` message words.
    pub const fn new(label: u64, words: usize) -> MsgInfo {
        MsgInfo(label << 16 | (words as u64 & 0x3ff))
    }
    /// A descriptor with `label`, `words` message words and `caps` capabilities.
    pub const fn with_caps(label: u64, words: usize, caps: usize) -> MsgInfo {
        MsgInfo(label << 16 | (caps as u64 & 7) << 10 | (words as u64 & 0x3ff))
    }
    /// The label.
    pub const fn label(self) -> u64 {
        self.0 >> 16
    }
    /// The number of capabilities, clamped to the maximum.
    pub const fn caps(self) -> usize {
        let n = (self.0 >> 10 & 7) as usize;
        if n > Self::MAX_CAPS { Self::MAX_CAPS } else { n }
    }
    /// The number of message words, clamped to the maximum.
    pub const fn words(self) -> usize {
        let n = (self.0 & 0x3ff) as usize;
        if n > Self::MAX_WORDS { Self::MAX_WORDS } else { n }
    }
}

/// Byte offset in the IPC buffer of the capabilities being sent.
pub const IPC_CAPS: usize = 4000;
/// Byte offset in the IPC buffer of the slots received capabilities go to.
pub const IPC_RECV: usize = 4032;
/// The slot index part of an entry at [`IPC_CAPS`].
pub const IPC_CAP_SLOT: u64 = (1 << 60) - 1;
/// Pass a capability derived from the sender's: the sender can revoke it.
pub const IPC_CAP_DERIVE: u64 = 0;
/// Pass an independent duplicate.
pub const IPC_CAP_DUPLICATE: u64 = 1;
/// Pass the capability itself; the sender no longer has it.
pub const IPC_CAP_MOVE: u64 = 2;

/// Label of a fault message, sent on behalf of a faulting thread to its
/// fault endpoint as a call. Words: 0 instruction pointer, 1 fault address,
/// 2 for a [`Fault::Page`] the access ([`FAULT_PRESENT`], [`FAULT_WRITE`],
/// [`FAULT_FETCH`]; the same on every architecture), for other faults the
/// architecture's own detail (x86_64: vector in word 1, error code here;
/// aarch64: the exception syndrome here), 3 stack pointer. Replying resumes the thread
/// when reply word 0 is [`FAULT_RESUME`] or [`FAULT_MAP`]; otherwise it stays
/// suspended.
pub const FAULT_LABEL: u64 = 0xFFFF_0000_0000;

/// Kinds of fault, added to [`FAULT_LABEL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Fault {
    /// Access to an address with no mapping or without the needed right.
    Page = 1,
    /// An instruction the processor does not accept in user mode.
    Instruction = 2,
    /// Any other processor exception (divide error, alignment, FP).
    Exception = 3,
}

/// Page fault access bit: a mapping exists but lacks the right needed (a
/// protection fault); clear when nothing is mapped there.
pub const FAULT_PRESENT: u64 = 1;
/// Page fault access bit: the access was a write.
pub const FAULT_WRITE: u64 = 2;
/// Page fault access bit: the access was an instruction fetch.
pub const FAULT_FETCH: u64 = 4;

/// Reply word 0 that resumes a faulted thread.
pub const FAULT_RESUME: u64 = 1;
/// Reply word 0 that maps a frame and then resumes the faulted thread, in
/// one kernel entry: word 1 the address space, word 2 the frame (both
/// capabilities of the replying thread), word 3 the page-aligned virtual
/// address with the rights in its low four bits. What `Map` followed by a
/// `FAULT_RESUME` reply does; if the mapping fails, the reply fails with
/// that error and the fault stays unanswered.
pub const FAULT_MAP: u64 = 2;
