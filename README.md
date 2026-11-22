# kernel

The Silica microkernel and what must agree with it byte for byte.

| Crate | What it is |
| --- | --- |
| `abi` | The kernel ABI: syscall numbers, object types, rights, errors, the message descriptor and the root task's start-up record. Private to the kernel and `libsys`; applications use `libsys`. |
| `kernel-core` | The kernel itself, free of architecture code: capabilities and the derivation tree, Untyped retype, threads, endpoints, notifications, address spaces, the root task. Builds and tests on the host. |
| `kernel` | The architecture layer and the kernel binary. `arch/x86_64` runs everything; `arch/aarch64` boots through the loader (from EL1 or EL2) to the kernel's report and stops there: no interrupt controller, timer, user mode or root task yet; `riscv64` builds only. |
| `boot-uefi` | The UEFI loader for x86_64 and aarch64: reads kernel, ramdisk and command line from the ESP, applies the KASLR slide, builds the first page tables, hands over a `BootInfo`. What differs per architecture is in `src/arch/`. |
| `libsys` | The system library: one typed function per kernel operation, and the monotonic clock. |
| `roottest` | The root task used by the in-OS tests. |

Belongs here: the ten kernel object types and their mechanisms
(`spec/ARCHITECTURE.md` sections 5 and 6), the loader, `libsys`. Does not
belong here: drivers (except the interrupt controller, the timer and the
kernel's own serial console), filesystems, network, policy, or anything that
could run in userspace.

## Reading the kernel

Start with `abi/src/lib.rs` (every operation and error, with its meaning),
then `kernel-core/src/cap.rs` (slots and the derivation list), `untyped.rs`,
`thread.rs`, `ipc.rs`, `vm.rs`, `kernel.rs` (the dispatcher) and `root.rs`.
`kernel/arch/x86_64/cpu.rs` holds every way into and out of the kernel.

## Tests

Which test checks which kind of change, and when a tier is the right one, is
in the workshop's `spec/TESTING.md` section 2. After a change to one in-OS
path, `./tools/boot.sh --test <name>` from the workshop root runs that test
alone with its serial output on standard output.

`kernel-core/tests/objects.rs` are scenario tests through the syscall
interface on a host machine (`tests/support`). `tests/model.rs` compares the
kernel with a reference model after each of thousands of random operations.
`tests/support/fuzz.rs` is the syscall fuzzer's grammar and its invariants.
In-OS tests are the functions listed in `roottest/src/main.rs` and named in
`tests/inos-x86_64.txt`.

License: MIT OR Apache-2.0.
