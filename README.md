# kernel

The Silica microkernel: `kernel-core` (arch-free logic, builds and tests on the
host), `kernel` (arch glue, `no_std`, with `arch/{x86_64,aarch64,riscv64}`),
`libsys` (the vDSO-style system library) and `boot-uefi` (the UEFI loader),
plus kernel tests and proofs.

Belongs here: the ten kernel object types and their mechanisms
(`spec/ARCHITECTURE.md` sections 5 and 6), the loader, `libsys`. Does not
belong here: drivers (except interrupt controller, timer, early serial),
filesystems, network, policy, or anything that could run in userspace.
Depends only on `idl`.

Host tests: `cargo test -p kernel-core`. License: MIT OR Apache-2.0.
