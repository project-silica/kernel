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
