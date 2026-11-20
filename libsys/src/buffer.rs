//! The IPC buffer: one page a thread shares with the kernel for the parts
//! of a message that do not fit in registers.
#![allow(unsafe_code)]

use crate::Cap;
use silica_abi::{IPC_CAP_DERIVE, IPC_CAP_DUPLICATE, IPC_CAP_MOVE, IPC_CAPS, IPC_RECV, MsgInfo};

/// How a capability is passed in a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// The receiver gets a capability derived from the sender's: the sender
    /// can take it back by revoking its own.
    Lend,
    /// The receiver gets an independent duplicate.
    Duplicate,
    /// The receiver gets the capability itself; the sender no longer has it.
    Give,
}

/// A thread's IPC buffer.
#[derive(Clone, Copy, Debug)]
pub struct IpcBuffer(*mut u64);

impl IpcBuffer {
    /// The buffer at `addr`.
    ///
    /// # Safety
    /// `addr` must be where the calling thread's IPC buffer frame (the one
    /// given to `thread_configure`, or `RootInfo::ipc_buffer`) is mapped
    /// read-write, and stay so while the value is used.
    pub unsafe fn new(addr: usize) -> IpcBuffer {
        IpcBuffer(addr as *mut u64)
    }

    fn at(&self, byte: usize) -> *mut u64 {
        debug_assert!(byte < 4096 && byte.is_multiple_of(8));
        // SAFETY: inside the one page the constructor's caller vouched for.
        unsafe { self.0.add(byte / 8) }
    }

    /// Message word `i` (words 0..3 travel in registers instead).
    pub fn word(&self, i: usize) -> u64 {
        assert!(i < MsgInfo::MAX_WORDS);
        // SAFETY: a word of the buffer page; the kernel writes it only while
        // this thread is inside a system call.
        unsafe { self.at(i * 8).read_volatile() }
    }

    /// Set message word `i`.
    pub fn set_word(&self, i: usize, v: u64) {
        assert!(i < MsgInfo::MAX_WORDS);
        // SAFETY: as for `word`.
        unsafe { self.at(i * 8).write_volatile(v) }
    }

    /// Copy `bytes` into the message area starting at byte `offset`.
    pub fn copy_in(&self, offset: usize, bytes: &[u8]) {
        assert!(offset.checked_add(bytes.len()).is_some_and(|end| end <= MsgInfo::MAX_WORDS * 8));
        // SAFETY: the range was just checked to lie in the message area of
        // the buffer page; `bytes` is ordinary memory elsewhere.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.0.cast::<u8>().add(offset), bytes.len()) }
    }

    /// Copy from the message area starting at byte `offset` into `out`.
    pub fn copy_out(&self, offset: usize, out: &mut [u8]) {
        assert!(offset.checked_add(out.len()).is_some_and(|end| end <= MsgInfo::MAX_WORDS * 8));
        // SAFETY: as for `copy_in`.
        unsafe { core::ptr::copy_nonoverlapping(self.0.cast::<u8>().add(offset), out.as_mut_ptr(), out.len()) }
    }

    /// Name the capability to pass as number `i` of the next message that
    /// says it carries more than `i` capabilities.
    pub fn pass(&self, i: usize, cap: Cap, how: Pass) {
        assert!(i < MsgInfo::MAX_CAPS);
        let mode = match how {
            Pass::Lend => IPC_CAP_DERIVE,
            Pass::Duplicate => IPC_CAP_DUPLICATE,
            Pass::Give => IPC_CAP_MOVE,
        };
        // SAFETY: as for `word`.
        unsafe { self.at(IPC_CAPS + i * 8).write_volatile(cap.0 | mode << 60) }
    }

    /// Have capability number `i` of received messages put in the empty
    /// slot `slot` of this thread's CapSpace (0: accept none).
    pub fn accept(&self, i: usize, slot: u64) {
        assert!(i < MsgInfo::MAX_CAPS);
        // SAFETY: as for `word`.
        unsafe { self.at(IPC_RECV + i * 8).write_volatile(slot) }
    }
}
