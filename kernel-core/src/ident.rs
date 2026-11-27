//! What `Inspect` shows of a kernel object: an identity, not where it is.
//!
//! An object's capability holds its physical address, and `Inspect` gave
//! it out as the object's "identity" (equal for every capability to one
//! object, and to nothing else: `SlotInfo::object`, and a thread's
//! `waits_on`). A physical address is more than an identity: it tells the
//! machine's memory layout, which memory a program's objects came from and
//! where they lie beside others' (the 2026-11-27 audit: M13, N4), and no
//! program needs it to tell objects apart. What `Inspect` shows now is a
//! keyed permutation of it (`Kernel::identity`): one-to-one, so equal
//! exactly where the addresses are, and with a key made at boot from the
//! boot's entropy that no program learns, unrelated to the address.
//!
//! The permutation is eight rounds of a Feistel network over the two
//! halves of the 64-bit address, each round's function SipHash-2-4 of the
//! round and the half (a pseudorandom function: the rounds make a
//! pseudorandom permutation, Luby and Rackoff; eight, as four over 32-bit
//! halves can be told from random after some 2^16 values). The key is
//! derived from the boot's entropy under a label of its own, and the root
//! task's boot secret under another ([`root_secret`]): the entropy itself
//! is not kept, and the root task cannot work out the key from its secret.

use crate::hal::Hal;
use crate::kernel::Kernel;

/// SipHash-2-4 of the single 64-bit word `m` under key `k` (Aumasson and
/// Bernstein, 2012; the eight-byte message little-endian).
pub fn siphash(k: [u64; 2], m: u64) -> u64 {
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

/// The permutation of `x` under key `k`: eight Feistel rounds (four over
/// 32-bit halves can be told from a random permutation after some 2^16
/// values: the review of audit-kernel-stalls).
pub fn permute(k: [u64; 2], x: u64) -> u64 {
    let (mut l, mut r) = (x >> 32, x & 0xFFFF_FFFF);
    for i in 0..8u64 {
        (l, r) = (r, l ^ (siphash(k, i << 32 | r) & 0xFFFF_FFFF));
    }
    l << 32 | r
}

/// Word `i` of what the boot's 32 bytes of entropy give under `label`:
/// SipHash of the label and `i` under each half of the entropy, the two
/// combined, so each word depends on all of it. Under different labels the
/// words are unrelated: knowing what one label gave tells nothing of
/// another's (SipHash is a pseudorandom function of its key).
fn derive(entropy: &[u8; 32], label: u64, i: u64) -> u64 {
    let w = |j: usize| u64::from_le_bytes(entropy[j * 8..j * 8 + 8].try_into().unwrap_or([0; 8]));
    siphash([w(0), w(1)], label ^ i) ^ siphash([w(2), w(3)], label.rotate_left(32) ^ i)
}

/// "identity" and "rootsecr", little-endian.
const IDENTITY: u64 = 0x7974_6974_6e65_6469;
const ROOT: u64 = 0x7263_6573_746f_6f72;

/// The key of the identities, from the boot's 32 bytes of entropy.
pub fn key_from(entropy: &[u8; 32]) -> [u64; 2] {
    [derive(entropy, IDENTITY, 0), derive(entropy, IDENTITY, 1)]
}

/// The root task's boot secret, from the same entropy under another label:
/// the root task, which holds it, cannot work out the identities' key from
/// it (it was the entropy itself, from which it could: the review of
/// audit-kernel-stalls).
pub fn root_secret(entropy: &[u8; 32]) -> [u64; 4] {
    core::array::from_fn(|i| derive(entropy, ROOT, i as u64))
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
