# kernel

The Silica microkernel and what must agree with it byte for byte.

| Crate | What it is |
| --- | --- |
| `abi` | The kernel ABI: syscall numbers, object types, rights, errors, the message descriptor and the root task's start-up record. Private to the kernel and `libsys`; applications use `libsys`. |
| `kernel-core` | The kernel itself, free of architecture code: capabilities and the derivation tree, Untyped retype, threads, endpoints, notifications, address spaces, the root task. Builds and tests on the host. |
| `kernel` | The architecture layer (`arch/x86_64`; `aarch64` and `riscv64` build but do not boot yet) and the kernel binary. |
| `boot-uefi` | The UEFI loader: reads kernel, ramdisk and command line from the ESP, applies the KASLR slide, builds the first page tables, hands over a `BootInfo`. |
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

## Testing

From the workshop root:

    ./tools/test.sh --fast            # host tests, builds, one boot, size gates
    ./tools/test.sh --full            # adds every in-OS test, the boot matrix, Miri
    ./tools/boot.sh --test ipc        # one in-OS test, serial on standard output
    ./tools/fuzz.sh --repo kernel     # the syscall and ELF fuzzers

`kernel-core/tests/objects.rs` are scenario tests through the syscall
interface on a host machine (`tests/support`). `tests/model.rs` compares the
kernel with a reference model after each of thousands of random operations.
`tests/support/fuzz.rs` is the syscall fuzzer's grammar and its invariants.
In-OS tests are the functions listed in `roottest/src/main.rs` and named in
`tests/inos-x86_64.txt`.

License: MIT OR Apache-2.0.
