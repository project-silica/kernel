//! The console's transmit ring (`kernel_core::log::TxRing`): bytes go out
//! in order, a piece that does not fit is dropped whole and said.

use kernel_core::log::TxRing;

fn drain<const N: usize>(r: &TxRing<N>) -> Vec<u8> {
    core::iter::from_fn(|| r.take()).collect()
}

#[test]
fn bytes_come_out_in_order_across_the_wrap() {
    let r: TxRing<16> = TxRing::new();
    for round in 0..10u8 {
        r.put(&[round; 5]);
        r.put(&[round + 100; 3]);
        assert_eq!(drain(&r), [[round; 5].as_slice(), &[round + 100; 3]].concat());
        assert!(!r.pending());
    }
}

#[test]
fn a_piece_that_does_not_fit_is_dropped_whole_and_said() {
    let r: TxRing<64> = TxRing::new();
    r.put(b"first line\r\n");
    r.put(&[b'x'; 60]); // no room: dropped whole
    r.put(&[b'y'; 60]); // still none
    assert_eq!(drain(&r), b"first line\r\n");
    r.put(b"after\r\n");
    let out = String::from_utf8(drain(&r)).unwrap();
    assert_eq!(out, "\r\n[console: 120 bytes not written]\r\nafter\r\n");
}

/// The ring keeps what it was given after it is sent: a panic's crash
/// record takes the console's last lines from it.
#[test]
fn the_last_bytes_put_are_kept_after_they_are_sent() {
    let r: TxRing<8> = TxRing::new();
    let mut out = [0u8; 16];
    assert_eq!(r.recent(&mut out), 0);
    r.put(b"abc");
    while r.take().is_some() {}
    r.put(b"defgh");
    let n = r.recent(&mut out);
    assert_eq!(&out[..n], b"abcdefgh");
    while r.take().is_some() {}
    r.put(b"ij");
    let n = r.recent(&mut out);
    assert_eq!(&out[..n], b"cdefghij", "the oldest bytes give way");
    let mut few = [0u8; 3];
    assert_eq!(r.recent(&mut few), 3);
    assert_eq!(&few, b"hij");
}
