//! `libsys`: the system library (`spec/ARCHITECTURE.md` section 4). Programs
//! reach the kernel only through the functions here; syscall numbers and
//! register layouts are private to this crate and the kernel.
//!
//! A capability is named by the index of its slot in the calling thread's
//! CapSpace ([`Cap`]). Slot 0 is never a capability.
#![no_std]
#![deny(unsafe_code)]

#[cfg(target_arch = "x86_64")]
#[path = "arch/x86_64.rs"]
#[allow(unsafe_code)]
mod arch;
#[cfg(target_arch = "aarch64")]
#[path = "arch/aarch64.rs"]
#[allow(unsafe_code)]
mod arch;

mod buffer;

pub use buffer::{IpcBuffer, Pass};
use silica_abi::NONBLOCK;
pub use silica_abi::Syscall;
pub use silica_abi::root;
pub use silica_abi::{
    Error, FAULT_FETCH, FAULT_LABEL, FAULT_MAP, FAULT_PRESENT, FAULT_RESUME, FAULT_WRITE, Fault, IRQ_ACTIVE_LOW, IRQ_LEVEL, KEEP, MsgInfo,
    ObjectType, Rights, THREAD_INACTIVE, THREAD_RECEIVING, THREAD_REPLY, THREAD_RUNNABLE, THREAD_SENDING,
};

/// Platform API level (`spec/CI_RELEASE.md` section 7).
pub const API_LEVEL: u32 = 0;

/// The result of a kernel operation.
pub type Result<T> = core::result::Result<T, Error>;

/// A capability: a slot index in the caller's CapSpace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cap(pub u64);

impl Cap {
    /// "No capability", where an operation accepts one.
    pub const NONE: Cap = Cap(0);
    /// `thread_configure`: leave the field as it is.
    pub const KEEP: Cap = Cap(KEEP);
}

/// A received message: up to four words in registers. Longer messages
/// continue in the IPC buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Message {
    /// The badge of the capability the sender used (for a notification, the
    /// pending signals). 0 in the reply to a call.
    pub badge: u64,
    /// Label and length.
    pub info: MsgInfo,
    /// Message words 0..3.
    pub words: [u64; 4],
}

/// How long a receive may wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// Until a message arrives.
    Forever,
    /// Not at all: fail with `WouldBlock` when nothing is waiting.
    Poll,
    /// Until this monotonic time in nanoseconds (see [`now`]), then fail with `Timeout`.
    Until(u64),
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn sys(n: Syscall, a: [u64; 6]) -> Result<[u64; 6]> {
    let (status, out) = arch::syscall(n as u64, a);
    Error::result(status).map(|()| out)
}

/// System call `n` with six argument words, as the ABI defines it
/// (`silica_abi::Syscall`): the six result words, or the error. The typed
/// functions of this crate are built on it; use it for what they do not
/// cover, such as testing how the kernel answers malformed arguments.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn raw(n: Syscall, a: [u64; 6]) -> Result<[u64; 6]> {
    sys(n, a)
}

fn message(r: [u64; 6]) -> Message {
    Message { badge: r[0], info: MsgInfo(r[1]), words: [r[2], r[3], r[4], r[5]] }
}

/// Send a message to an endpoint and wait for the reply.
pub fn call(ep: Cap, info: MsgInfo, w: [u64; 4]) -> Result<Message> {
    sys(Syscall::Call, [ep.0, info.0, w[0], w[1], w[2], w[3]]).map(message)
}

/// Send a message to an endpoint, waiting for a receiver; or signal a notification.
pub fn send(to: Cap, info: MsgInfo, w: [u64; 4]) -> Result<()> {
    sys(Syscall::Send, [to.0, info.0, w[0], w[1], w[2], w[3]]).map(|_| ())
}

/// Like [`send`], but fails with `WouldBlock` when no receiver is waiting.
pub fn try_send(to: Cap, info: MsgInfo, w: [u64; 4]) -> Result<()> {
    sys(Syscall::Send, [to.0 | NONBLOCK, info.0, w[0], w[1], w[2], w[3]]).map(|_| ())
}

/// Signal a notification: OR the capability's badge into its pending word
/// (bit 0 when the capability has no badge).
pub fn signal(notification: Cap) -> Result<()> {
    send(notification, MsgInfo(0), [0; 4])
}

/// Receive from an endpoint, or take a notification's pending signals.
pub fn recv(from: Cap, wait: Wait) -> Result<Message> {
    let deadline = match wait {
        Wait::Forever => 0,
        Wait::Poll => 1,
        Wait::Until(t) => t.max(2),
    };
    sys(Syscall::Recv, [from.0, deadline, 0, 0, 0, 0]).map(message)
}

/// Receive from an endpoint; if what arrives is a call, park it on the
/// notification `park` instead of holding it on this thread. A parked call
/// is answered, by any thread and at any later time, with [`answer`]; until
/// then this thread is free to receive and answer others. One call parks on
/// one notification at a time.
pub fn recv_parking(from: Cap, wait: Wait, park: Cap) -> Result<Message> {
    let deadline = match wait {
        Wait::Forever => 0,
        Wait::Poll => 1,
        Wait::Until(t) => t.max(2),
    };
    sys(Syscall::Recv, [from.0, deadline, park.0, 0, 0, 0]).map(message)
}

/// Answer the call parked on notification `parked`.
pub fn answer(parked: Cap, info: MsgInfo, w: [u64; 4]) -> Result<()> {
    send(parked, info, w)
}

/// Reply to the call received last. Fails with `NoCaller` when there is none.
pub fn reply(info: MsgInfo, w: [u64; 4]) -> Result<()> {
    sys(Syscall::ReplyRecv, [0, info.0, w[0], w[1], w[2], w[3]]).map(|_| ())
}

/// Reply to the call received last (dropped silently if that caller is
/// gone), then receive the next message from `from`. The server loop.
pub fn reply_recv(from: Cap, info: MsgInfo, w: [u64; 4]) -> Result<Message> {
    sys(Syscall::ReplyRecv, [from.0, info.0, w[0], w[1], w[2], w[3]]).map(message)
}

/// A pager's answer to a page fault: map `frame` at `vaddr` in `aspace`,
/// resume the faulted thread, and receive the next message from `from`, all
/// in one kernel entry. If the mapping fails its error is returned and the
/// fault is still unanswered.
pub fn fault_map_recv(from: Cap, aspace: Cap, frame: Cap, vaddr: usize, rights: Rights) -> Result<Message> {
    let at = (vaddr as u64 & !0xFFF) | rights.0 as u64;
    sys(Syscall::ReplyRecv, [from.0, MsgInfo::new(0, 4).0, FAULT_MAP, aspace.0, frame.0, at]).map(message)
}

/// Like [`fault_map_recv`] without receiving again.
pub fn fault_map(aspace: Cap, frame: Cap, vaddr: usize, rights: Rights) -> Result<()> {
    let at = (vaddr as u64 & !0xFFF) | rights.0 as u64;
    sys(Syscall::ReplyRecv, [0, MsgInfo::new(0, 4).0, FAULT_MAP, aspace.0, frame.0, at]).map(|_| ())
}

/// Let another runnable thread use the processor.
pub fn yield_now() {
    let _ = sys(Syscall::Yield, [0; 6]);
}

/// Create `count` objects of type `ty` from `untyped`, in slots
/// `first..first + count` of the CapSpace `space`. `bits` is the size for
/// Untyped, Frame and AddressSpace, the number of slots (in bits) for a
/// CapSpace, and ignored otherwise.
pub fn retype(untyped: Cap, ty: ObjectType, bits: u8, space: Cap, first: u64, count: u64) -> Result<()> {
    sys(Syscall::Retype, [untyped.0, ty as u64, bits as u64, space.0, first, count]).map(|_| ())
}

/// Derive a capability from `src` into slot `dst` of `space`, keeping only
/// the rights in `rights`. A non-zero `badge` badges an unbadged endpoint or
/// notification capability, or narrows a Hw range (start in the low 32
/// bits, end in the high 32). The new capability depends on `src`: revoking
/// or deleting `src` deletes it. This is how to lend authority.
pub fn cap_derive(space: Cap, dst: u64, src: Cap, rights: Rights, badge: u64) -> Result<()> {
    sys(Syscall::CapCopy, [space.0, dst, src.0, rights.0 as u64, badge, 0]).map(|_| ())
}

/// Like [`cap_derive`], but the new capability is independent of `src`: it
/// lives until it is deleted itself or whatever `src` was derived from is
/// revoked. This is how to hand authority on for good.
pub fn cap_duplicate(space: Cap, dst: u64, src: Cap, rights: Rights, badge: u64) -> Result<()> {
    sys(Syscall::CapCopy, [space.0, dst, src.0, rights.0 as u64, badge, 1]).map(|_| ())
}

/// Move `src` into slot `dst` of `space`.
pub fn cap_move(space: Cap, dst: u64, src: Cap) -> Result<()> {
    sys(Syscall::CapMove, [space.0, dst, src.0, 0, 0, 0]).map(|_| ())
}

/// Delete a capability and everything derived from it. An object is
/// destroyed with its last capability.
///
/// # Safety
/// Deleting the last capability to memory, an address space or a thread
/// that running code depends on removes it from under that code.
#[allow(unsafe_code)]
pub unsafe fn cap_delete(cap: Cap) -> Result<()> {
    sys(Syscall::CapDelete, [cap.0, 0, 0, 0, 0, 0]).map(|_| ())
}

/// Delete every capability derived from `cap`; for an Untyped, destroy
/// everything made from it and make its memory available again.
///
/// # Safety
/// As for [`cap_delete`], for everything derived from `cap`.
#[allow(unsafe_code)]
pub unsafe fn cap_revoke(cap: Cap) -> Result<()> {
    sys(Syscall::CapRevoke, [cap.0, 0, 0, 0, 0, 0]).map(|_| ())
}

/// Map `frame` at `vaddr` in `aspace`. Never replaces an existing mapping
/// (`Overlap`), and never maps writable and executable at once (`BadArg`).
/// On error, every mapping and all remaining page-table capacity are preserved.
pub fn map(aspace: Cap, frame: Cap, vaddr: usize, rights: Rights) -> Result<()> {
    sys(Syscall::Map, [aspace.0, frame.0, vaddr as u64, rights.0 as u64, 0, 0]).map(|_| ())
}

/// Remove the mapping made through `frame`.
///
/// # Safety
/// Nothing may use the mapped memory afterwards.
#[allow(unsafe_code)]
pub unsafe fn unmap(frame: Cap) -> Result<()> {
    sys(Syscall::Unmap, [frame.0, 0, 0, 0, 0, 0]).map(|_| ())
}

/// Give a thread its CapSpace, address space, fault endpoint and IPC buffer
/// frame. [`Cap::KEEP`] leaves a field unchanged; [`Cap::NONE`] clears it.
pub fn thread_configure(thread: Cap, space: Cap, aspace: Cap, fault: Cap, ipc_buffer: Cap) -> Result<()> {
    sys(Syscall::ThreadConfigure, [thread.0, space.0, aspace.0, fault.0, ipc_buffer.0, 0]).map(|_| ())
}

/// Set a thread's instruction pointer, stack pointer and first argument,
/// clearing its other registers, and make it runnable when `run` is set.
///
/// # Safety
/// `ip` must be code that may run on the stack at `sp` in the thread's
/// address space.
#[allow(unsafe_code)]
pub unsafe fn thread_start(thread: Cap, ip: usize, sp: usize, arg: u64, run: bool) -> Result<()> {
    sys(Syscall::ThreadStart, [thread.0, ip as u64, sp as u64, arg, run as u64, 0]).map(|_| ())
}

/// Stop a thread where it is. A call it was blocked in starts again when
/// the thread is resumed, unless its message was already delivered.
pub fn thread_suspend(thread: Cap) -> Result<()> {
    sys(Syscall::ThreadControl, [thread.0, 0, 0, 0, 0, 0]).map(|_| ())
}

/// Move a thread to processor `cpu` (an index below `RootInfo::cpus`). The
/// thread is suspended first and stays suspended; resume it to run there.
pub fn thread_set_cpu(thread: Cap, cpu: usize) -> Result<()> {
    sys(Syscall::ThreadControl, [thread.0, 2, cpu as u64, 0, 0, 0]).map(|_| ())
}

/// Make a suspended thread runnable.
pub fn thread_resume(thread: Cap) -> Result<()> {
    sys(Syscall::ThreadControl, [thread.0, 1, 0, 0, 0, 0]).map(|_| ())
}

/// Read `width` bytes (1, 2 or 4) from an I/O port covered by `ports`.
pub fn port_in(ports: Cap, port: u16, width: u8) -> Result<u32> {
    sys(Syscall::PortIo, [ports.0, port as u64, width as u64, 0, 0, 0]).map(|r| r[0] as u32)
}

/// Write `bytes` to the kernel's console through a console capability, 32
/// bytes per system call.
pub fn console_write(console: Cap, bytes: &[u8]) -> Result<()> {
    for chunk in bytes.chunks(32) {
        let mut w = [0u64; 4];
        for (i, b) in chunk.iter().enumerate() {
            w[i / 8] |= (*b as u64) << (i % 8 * 8);
        }
        sys(Syscall::Console, [console.0, chunk.len() as u64, w[0], w[1], w[2], w[3]])?;
    }
    Ok(())
}

/// Turn the machine off through a power capability, telling a test harness
/// `status` (0: success). Returns only with the reason it could not.
pub fn power_off(power: Cap, status: u64) -> Error {
    match sys(Syscall::PowerOff, [power.0, status, 0, 0, 0, 0]) {
        Ok(_) => Error::Unsupported,
        Err(e) => e,
    }
}

/// Write `width` bytes (1, 2 or 4) to an I/O port covered by `ports`.
pub fn port_out(ports: Cap, port: u16, width: u8, value: u32) -> Result<()> {
    sys(Syscall::PortIo, [ports.0, port as u64, width as u64 | 0x100, value as u64, 0, 0]).map(|_| ())
}

/// From now on an interrupt on line `line` (a global system interrupt
/// number covered by `lines`) signals `notification` with that capability's
/// badge, on the calling thread's processor. `flags`: [`IRQ_LEVEL`],
/// [`IRQ_ACTIVE_LOW`]. After each interrupt of a level-triggered line the
/// line is held off until [`irq_ack`].
pub fn irq_bind(lines: Cap, line: u32, notification: Cap, flags: u64) -> Result<()> {
    sys(Syscall::IrqBind, [lines.0, line as u64, notification.0, flags, 0, 0]).map(|_| ())
}

/// Stop delivering interrupt line `line`.
pub fn irq_unbind(lines: Cap, line: u32) -> Result<()> {
    sys(Syscall::IrqBind, [lines.0, line as u64, 0, 0, 0, 0]).map(|_| ())
}

/// The device behind `line` has been dealt with: let it interrupt again.
pub fn irq_ack(lines: Cap, line: u32) -> Result<()> {
    sys(Syscall::IrqAck, [lines.0, line as u64, 0, 0, 0, 0]).map(|_| ())
}

/// What a capability slot holds, as [`inspect_slot`] reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotInfo {
    /// The type of object.
    pub ty: ObjectType,
    /// What the capability permits.
    pub rights: Rights,
    /// The object's size in bits (for a CapSpace, its slots).
    pub bits: u8,
    /// Equal for all capabilities to one object, and to nothing else.
    pub object: u64,
    /// The badge (endpoints and notifications).
    pub badge: u64,
    /// Generations between this capability and the one it descends from.
    pub depth: u16,
}

/// What slot `index` of the CapSpace `space` holds (`None`: nothing).
pub fn inspect_slot(space: Cap, index: u64) -> Result<Option<SlotInfo>> {
    let r = sys(Syscall::Inspect, [space.0, index, 0, 0, 0, 0])?;
    Ok(ObjectType::from_u64(r[0]).map(|ty| SlotInfo {
        ty,
        rights: Rights(r[1] as u8),
        bits: r[2] as u8,
        object: r[3],
        badge: r[4],
        depth: r[5] as u16,
    }))
}

/// A thread, as [`inspect_thread`] reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadInfo {
    /// One of the `THREAD_` values.
    pub state: u64,
    /// The processor it runs on.
    pub cpu: usize,
    /// Its instruction pointer and stack pointer when it last entered the kernel.
    pub ip: u64,
    /// See `ip`.
    pub sp: u64,
    /// The identity (as in [`SlotInfo::object`]) of what it waits on, or 0.
    pub waits_on: u64,
}

/// The state of `thread`.
pub fn inspect_thread(thread: Cap) -> Result<ThreadInfo> {
    let r = sys(Syscall::Inspect, [thread.0, 0, 0, 0, 0, 0])?;
    Ok(ThreadInfo { state: r[0], cpu: r[1] as usize, ip: r[2], sp: r[3], waits_on: r[4] })
}

/// The monotonic clock: converts the processor's timestamp counter with the
/// calibration from [`root::RootInfo::clock`]. Reading it is not a syscall.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    hz: u64,
    zero: u64,
}

impl Clock {
    /// A clock from `[ticks per second, _, counter at time 0]`.
    pub const fn new(clock: [u64; 3]) -> Clock {
        Clock { hz: clock[0], zero: clock[2] }
    }
    /// Nanoseconds between counter value `from` and now.
    pub fn since(&self, from: u64) -> u64 {
        let ticks = arch::timestamp().wrapping_sub(from);
        (ticks as u128 * 1_000_000_000 / self.hz.max(1) as u128) as u64
    }
    /// Monotonic nanoseconds, on the scale [`Wait::Until`] uses.
    pub fn now(&self) -> u64 {
        self.since(self.zero)
    }
}
