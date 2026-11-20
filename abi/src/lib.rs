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
    /// `a0` endpoint, `a1` [`MsgInfo`], `a2..a5` message words 0..3. Sends
    /// and blocks until the receiver replies. Returns the reply the same way.
    Call = 0,
    /// `a0` endpoint or notification, `a1` [`MsgInfo`], `a2..a5` words,
    /// `a0` bit 63 set: do not block. On a notification, ORs the capability's
    /// badge into the pending word (bit 0 for a capability without a badge).
    Send = 1,
    /// `a0` endpoint or notification, `a1` deadline in monotonic nanoseconds
    /// (0: wait for ever, 1: do not block). Returns `a0` badge, `a1`
    /// [`MsgInfo`], `a2..a5` words. For a notification `a0` is the pending
    /// word, cleared by the read.
    Recv = 2,
    /// Reply to the caller received last (`a1` [`MsgInfo`], `a2..a5` words),
    /// then, when `a0` is not 0, `Recv` on `a0` with no deadline.
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
    /// it; an object is destroyed with its last capability.
    CapDelete = 8,
    /// `a0` slot. Deletes every capability derived from this one.
    CapRevoke = 9,
    /// `a0` address space, `a1` frame, `a2` virtual address, `a3` rights.
    Map = 10,
    /// `a0` frame. Removes this capability's mapping.
    Unmap = 11,
    /// `a0` thread, `a1` CapSpace, `a2` address space, `a3` fault endpoint,
    /// `a4` IPC buffer frame. [`KEEP`] leaves a field unchanged, 0 clears it.
    ThreadConfigure = 12,
    /// `a0` thread, `a1` instruction pointer, `a2` stack pointer, `a3`
    /// first argument, `a4` not 0: make the thread runnable.
    ThreadStart = 13,
    /// `a0` thread, `a1` 0 suspend, 1 resume, 2 move to processor `a2`
    /// (the thread is suspended first and stays suspended).
    ThreadControl = 14,
    /// `a0` port capability, `a1` port, `a2` width in bytes (1, 2, 4) with
    /// bit 8 set for a write, `a3` value. Returns `a0` value read.
    PortIo = 15,
}

impl Syscall {
    /// One past the highest syscall number.
    pub const COUNT: u64 = 16;

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
            _ => return None,
        })
    }
}

/// `ThreadConfigure`: leave this field as it is.
pub const KEEP: u64 = u64::MAX;
/// `Send`: flag in `a0` asking not to block.
pub const NONBLOCK: u64 = 1 << 63;

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
    /// Processor time budget.
    SchedContext = 8,
    /// A range of hardware resources that are not memory: I/O ports or
    /// interrupt lines. Created at boot, narrowed by `CapCopy`.
    Hw = 9,
    /// A virtual CPU (M9).
    Vcpu = 10,
}

impl ObjectType {
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
/// Smallest AddressSpace: a header page and the root table. Each further
/// page is one page table.
pub const ASPACE_MIN_BITS: u8 = 13;
/// Largest single Untyped.
pub const UNTYPED_MAX_BITS: u8 = 47;
/// Most objects one `Retype` creates.
pub const RETYPE_MAX: u64 = 256;

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
    /// The object the thread was waiting on, or the thread that owed it a
    /// reply, was destroyed.
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
            _ => Error::BadSyscall,
        })
    }
}

/// The message descriptor word: bits 0..6 the number of message words
/// (at most [`MsgInfo::MAX_WORDS`]), bits 16..63 a label the kernel does not
/// interpret. Words 0..3 travel in registers; the rest in the IPC buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsgInfo(pub u64);

impl MsgInfo {
    /// Most words in one message.
    pub const MAX_WORDS: usize = 64;
    /// Words carried in registers.
    pub const REG_WORDS: usize = 4;

    /// A descriptor with `label` and `words` message words.
    pub const fn new(label: u64, words: usize) -> MsgInfo {
        MsgInfo(label << 16 | (words as u64 & 0x7f))
    }
    /// The label.
    pub const fn label(self) -> u64 {
        self.0 >> 16
    }
    /// The number of message words, clamped to the maximum.
    pub const fn words(self) -> usize {
        let n = (self.0 & 0x7f) as usize;
        if n > Self::MAX_WORDS { Self::MAX_WORDS } else { n }
    }
}

/// Label of a fault message, sent on behalf of a faulting thread to its
/// fault endpoint as a call. Words: 0 instruction pointer, 1 fault address,
/// 2 architecture error code, 3 stack pointer. Replying resumes the thread
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

/// Reply word 0 that resumes a faulted thread.
pub const FAULT_RESUME: u64 = 1;
/// Reply word 0 that maps a frame and then resumes the faulted thread, in
/// one kernel entry: word 1 the address space, word 2 the frame (both
/// capabilities of the replying thread), word 3 the page-aligned virtual
/// address with the rights in its low four bits. What `Map` followed by a
/// `FAULT_RESUME` reply does; if the mapping fails, the reply fails with
/// that error and the fault stays unanswered.
pub const FAULT_MAP: u64 = 2;
