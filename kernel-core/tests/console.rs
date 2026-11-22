//! The kernel console as several writers share it (`kernel_core::log`).
//! One test: the console's state is the whole program's.

mod support;

use kernel_core::log::write;
use support::Machine;

#[test]
fn a_line_cut_short_by_another_writer_is_shown_again_after_it() {
    let m = Machine::new(16);
    let hal = &m.k.hal;
    let out = || String::from_utf8_lossy(&hal.console.borrow()).into_owned();
    let (shell, service) = (1, 3);

    // A prompt, a service's line, the prompt again, then what is typed.
    write(hal, shell, b"silica> ");
    write(hal, service, b"partitions: 1\n");
    write(hal, shell, b"services\n");
    assert_eq!(out(), "silica> \r\npartitions: 1\nsilica> services\n");

    // A line finished by its own writer is not shown again.
    hal.console.borrow_mut().clear();
    write(hal, service, b"done\n");
    write(hal, shell, b"x\n");
    assert_eq!(out(), "done\nx\n");

    // What was typed after the prompt comes back with it, and the line goes
    // on from there.
    hal.console.borrow_mut().clear();
    write(hal, shell, b"silica> ser");
    write(hal, service, b"a");
    write(hal, service, b"b\n");
    write(hal, shell, b"vices\n");
    assert_eq!(out(), "silica> ser\r\nab\nsilica> services\n");

    // The cut writer speaking before the other's line is done: a fresh
    // line, its own text again, then the rest.
    hal.console.borrow_mut().clear();
    write(hal, shell, b"p> ");
    write(hal, service, b"half");
    write(hal, shell, b"q\n");
    assert_eq!(out(), "p> \r\nhalf\r\np> q\n");
    write(hal, service, b"next\n");

    // A line too long to keep is not shown again.
    hal.console.borrow_mut().clear();
    let long = [b'x'; 300];
    write(hal, shell, &long);
    write(hal, service, b"z\n");
    write(hal, shell, b"\n");
    let want = format!("{}\r\nz\n\n", "x".repeat(300));
    assert_eq!(out(), want);
}
