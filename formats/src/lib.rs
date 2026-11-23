//! Readers for the two file formats the system boots from: ELF64 images
//! ([`elf`]) and the initial ramdisk archive ([`ramdisk`]); the firmware's
//! ACPI tables ([`acpi`]); the console font ([`font`]); and the hash
//! the loader conditions boot entropy with ([`blake2s`]). Shared by the
//! loader, the kernel and userspace. They trust nothing in the bytes they
//! are given: every offset is checked before use, and malformed input is an
//! error or "not found", never a panic.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod acpi;
pub mod blake2s;
pub mod elf;
pub mod font;
pub mod ramdisk;
