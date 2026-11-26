//! The crash record as a running kernel uses it (`crash::Recorder`), the
//! same on every architecture: nothing is written before a panic, the first
//! panic alone writes, and the next boot gets the console's last lines and
//! the panic, once.

use kernel_core::crash::{Recorder, read};

#[test]
fn a_panic_is_recorded_once_and_the_next_boot_takes_it_once() {
    let mut region = vec![0u8; 4096];
    let mut kept = [0u8; 64];
    let r = Recorder::new();
    assert_eq!(r.start(&mut region, &mut kept, |_, _| panic!("no record yet")), b"");
    // Before a panic, what the console says is not recorded.
    r.say(|| panic!("the region is not asked for before a panic"), b"ordinary line\r\n");
    r.begin(Some(&mut region), |last| {
        last[..6].copy_from_slice(b"tail\r\n");
        6
    });
    // A second processor panicking meanwhile starts nothing over.
    r.begin(Some(&mut region), |_| panic!("only the first panic takes the console's lines"));
    r.say(|| Some(&mut region), b"PANIC: deliberate\r\n");
    assert_eq!(read(&region), None, "unsealed");
    r.seal(Some(&mut region));
    // The next boot (the region kept, the kernel's own state new).
    let next = Recorder::new();
    let mut said = Vec::new();
    let got = next.start(&mut region, &mut kept, |was, text| {
        assert_eq!(was, 1);
        said.extend_from_slice(text);
    });
    assert_eq!(said, b"tail\r\nPANIC: deliberate\r\n");
    assert_eq!(got, b"tail\r\nPANIC: deliberate\r\n");
    // Taken once: the boot after that finds none, and counts on.
    let third = Recorder::new();
    assert_eq!(third.start(&mut region, &mut kept, |_, _| panic!("the record was cleared")), b"");
}

/// A machine whose loader kept no region: a panic records nothing and
/// does not fail.
#[test]
fn a_machine_with_no_region_records_nothing() {
    let none = Recorder::new();
    none.begin(None, |_| panic!("no region, no record"));
    none.say(|| None, b"x");
    none.seal(None);
}
