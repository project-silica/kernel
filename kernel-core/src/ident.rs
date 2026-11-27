//! What `Inspect` shows of a kernel object: an identity, not where it is.
//!
//! An object's capability holds its physical address, and `Inspect` gave
//! it out as the object's "identity" (equal for every capability to one
//! object, and to nothing else: `SlotInfo::object`, and a thread's
//! `waits_on`). A physical address is more than an identity: it tells the
//! machine's memory layout, which memory a program's objects came from and
//! where they lie beside others' (the 2026-11-27 audit: M13, N4), and no
//! program needs it to tell objects apart. What `Inspect` shows now is a
//! keyed permutation of it ([`Kernel::identity`]): one-to-one, so equal
//! exactly where the addresses are, and with a key made at boot from the
//! boot's entropy that no program learns, unrelated to the address.
//!
//! The permutation is four rounds of a Feistel network over the two
//! halves of the 64-bit address, each round's function SipHash-2-4 of the
//! round and the half (a pseudorandom function, so the four rounds make a
//! pseudorandom permutation: Luby and Rackoff). The key is SipHash of a
//! constant under the boot entropy's two halves: the entropy itself (the
//! root task's boot secret) is not kept.

use crate::hal::Hal;
use crate::kernel::Kernel;

/// SipHash-2-4 of the single 64-bit word `m` under key `k` (Aumasson and
/// Bernstein, 2012; the eight-byte message little-endian).
pub(crate) fn siphash(k: [u64; 2], m: u64) -> u64 {
    let mut v = [k[0] ^ 0x736f_6d65_7073_6575, k[1] ^ 0x646f_7261_6e64_6f6d, k[0] ^ 0x6c79_6765_6e65_7261, k[1] ^ 0x7465_6462_7974_6573];
    fn round(v: &mut [u64; 4]) {
        v[0] = v[0].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(13) ^ v[0];
        v[0] = v[0].rotate_left(32);
        v[2] = v[2].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(16) ^ v[2];
        v[0] = v[0].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(21) ^ v[0];
        v[2] = v[2].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(17) ^ v[2];
        v[2] = v[2].rotate_left(32);
    }
    // The message, then the last block: no bytes left, the length (8).
    for b in [m, 8u64 << 56] {
        v[3] ^= b;
        round(&mut v);
        round(&mut v);
        v[0] ^= b;
    }
    v[2] ^= 0xff;
    for _ in 0..4 {
        round(&mut v);
    }
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

/// The permutation of `x` under key `k` (four Feistel rounds).
pub(crate) fn permute(k: [u64; 2], x: u64) -> u64 {
    let (mut l, mut r) = (x >> 32, x & 0xFFFF_FFFF);
    for i in 0..4u64 {
        (l, r) = (r, l ^ (siphash(k, i << 32 | r) & 0xFFFF_FFFF));
    }
    l << 32 | r
}

/// The key of the identities, from the boot's 32 bytes of entropy.
pub(crate) fn key_from(entropy: &[u8; 32]) -> [u64; 2] {
    let w = |i: usize| u64::from_le_bytes(entropy[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
    // "identity", once under each half of the entropy.
    const LABEL: u64 = 0x7974_6974_6e65_6469;
    [siphash([w(0), w(1)], LABEL), siphash([w(2), w(3)], LABEL ^ 1)]
}

impl<H: Hal> Kernel<H> {
    /// What `Inspect` shows of the object at physical address `p`: equal
    /// for one object and different for different ones, and nothing of
    /// where it is (0 stays 0: "none").
    pub(crate) fn identity(&self, p: u64) -> u64 {
        if p == 0 { 0 } else { permute([self.ident_key[0].get(), self.ident_key[1].get()], p) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::assert_eq;

    /// The reference implementation's vector for an eight-byte message:
    /// key 00 01 .. 0f, message 00 01 .. 07.
    #[test]
    fn siphash_matches_the_reference_vector() {
        let k = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
        assert_eq!(siphash(k, 0x0706_0504_0302_0100), 0x93f5_f579_9a93_2462);
    }

    /// One-to-one over a run of addresses, and none of them shown as itself.
    #[test]
    fn permute_is_one_to_one_and_hides_the_address() {
        let k = key_from(&[7; 32]);
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..10_000u64 {
            let p = 0x10_0000 + i * 64;
            let id = permute(k, p);
            assert!(seen.insert(id), "two addresses share an identity");
            assert!(id != p);
        }
    }
}
