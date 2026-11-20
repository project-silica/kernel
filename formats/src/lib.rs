//! Readers for the two file formats the system boots from: ELF64 images
//! ([`elf`]) and the initial ramdisk archive ([`ramdisk`]). Shared by the
//! loader, the kernel and userspace. They trust nothing in the bytes they
//! are given: every offset is checked before use, and malformed input is an
//! error or "not found", never a panic.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod elf;
pub mod ramdisk;
