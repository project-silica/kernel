//! BLAKE2s-256 (RFC 7693): the hash the loader conditions boot entropy
//! with. Every source is fed in whole and the 32-byte digest is the seed, so
//! the result is unpredictable as long as any one source is, and a source
//! that lies (all zeros, a fixed pattern, a value chosen to cancel another)
//! cannot make it weaker than the others.

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// An incremental BLAKE2s-256 hash, optionally keyed.
#[derive(Clone)]
pub struct Blake2s {
    h: [u32; 8],
    t: u64,
    buf: [u8; 64],
    len: usize,
}

impl Blake2s {
    /// An unkeyed hash.
    pub fn new() -> Self {
        Self::keyed(&[])
    }

    /// A hash keyed with up to 32 bytes of `key` (the rest is ignored): a
    /// MAC, or a way to derive independent values from one secret.
    pub fn keyed(key: &[u8]) -> Self {
        let key = &key[..key.len().min(32)];
        let mut h = IV;
        h[0] ^= 0x0101_0000 ^ ((key.len() as u32) << 8) ^ 32;
        let mut s = Blake2s { h, t: 0, buf: [0; 64], len: 0 };
        if !key.is_empty() {
            s.buf[..key.len()].copy_from_slice(key);
            s.len = 64;
        }
        s
    }

    /// Feed `data` in.
    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        while !data.is_empty() {
            // The last block is compressed only at the end (it carries the
            // final flag), so a full buffer waits for more input.
            if self.len == 64 {
                self.t += 64;
                compress(&mut self.h, &self.buf, self.t, false);
                self.len = 0;
            }
            let n = data.len().min(64 - self.len);
            self.buf[self.len..self.len + n].copy_from_slice(&data[..n]);
            self.len += n;
            data = &data[n..];
        }
        self
    }

    /// The 32-byte digest.
    pub fn finish(&self) -> [u8; 32] {
        let mut h = self.h;
        let mut buf = self.buf;
        buf[self.len..].fill(0);
        compress(&mut h, &buf, self.t + self.len as u64, true);
        let mut out = [0; 32];
        for (o, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
            o.copy_from_slice(&w.to_le_bytes());
        }
        zero_words(&mut h);
        zero(&mut buf);
        out
    }

    /// Zero the state: what it holds of what was hashed (its last block,
    /// which for a keyed hash or an HMAC is the key or the padded key) and
    /// the chained value, as dropping it does.
    fn forget(&mut self) {
        zero_words(&mut self.h);
        zero(&mut self.buf);
        self.t = 0;
        self.len = 0;
    }
}

impl Drop for Blake2s {
    /// A hash of a secret (a key, a MAC, the boot's entropy) leaves nothing
    /// of it behind.
    fn drop(&mut self) {
        self.forget();
    }
}

/// Overwrite `b` with zeros in a way the optimizer keeps.
fn zero(b: &mut [u8]) {
    b.fill(0);
    core::hint::black_box(b);
}

/// [`zero`], for words.
fn zero_words(w: &mut [u32]) {
    w.fill(0);
    core::hint::black_box(w);
}

impl Default for Blake2s {
    fn default() -> Self {
        Self::new()
    }
}

fn compress(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
    let mut m = [0u32; 16];
    for (w, b) in m.iter_mut().zip(block.as_chunks::<4>().0) {
        *w = u32::from_le_bytes(*b);
    }
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u32;
    v[13] ^= (t >> 32) as u32;
    if last {
        v[14] = !v[14];
    }
    let g = |v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32| {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    };
    for s in &SIGMA {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
    // The message words and the working state are the block's, which may
    // be a key.
    zero_words(&mut m);
    zero_words(&mut v);
}

#[cfg(test)]
mod tests {
    use super::Blake2s;

    /// A keyed hash (as HMAC and the boot secret's derivations use it)
    /// holds its key in its last block until more comes, and the chained
    /// value after: both are zero once it is done with (`forget`, which
    /// its drop runs), and the digest is the same as before.
    #[test]
    fn its_key_and_state_are_zeroed_when_done_with() {
        let mut s = Blake2s::keyed(&[0x5a; 32]);
        assert_eq!(&s.buf[..32], &[0x5a; 32], "the key waits in the last block");
        let digest = s.finish();
        assert_ne!(digest, [0; 32]);
        assert!(s.buf.contains(&0x5a) && s.h.iter().any(|w| *w != 0), "finish leaves the state as it was");
        s.update(b"more than the key");
        assert!(s.h.iter().any(|w| *w != 0));
        s.forget();
        assert!(s.buf.iter().all(|b| *b == 0) && s.h.iter().all(|w| *w == 0) && (s.t, s.len) == (0, 0));
        assert_eq!(Blake2s::keyed(&[0x5a; 32]).finish(), digest, "wiping changes no digest");
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn known_answers() {
        // RFC 7693 appendix B.
        assert_eq!(hex(&Blake2s::new().update(b"abc").finish()), "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982");
        assert_eq!(hex(&Blake2s::new().finish()), "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9");
    }

    #[test]
    fn keyed_known_answers() {
        // The reference implementation's keyed vectors (blake2s-kat.txt):
        // key 00..1f, input 00..len-1.
        let key: Vec<u8> = (0..32).collect();
        let input: Vec<u8> = (0..=255).collect();
        assert_eq!(hex(&Blake2s::keyed(&key).finish()), "48a8997da407876b3d79c0d92325ad3b89cbb754d86ab71aee047ad345fd2c49");
        assert_eq!(
            hex(&Blake2s::keyed(&key).update(&input[..1]).finish()),
            "40d15fee7c328830166ac3f918650f807e7e01e177258cdc0a39b11f598066f1"
        );
        assert_eq!(
            hex(&Blake2s::keyed(&key).update(&input[..255]).finish()),
            "3fb735061abc519dfe979e54c1ee5bfad0a9d858b3315bad34bde999efd724dd"
        );
    }

    #[test]
    fn split_input_gives_the_same_digest() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let whole = Blake2s::new().update(&data).finish();
        for cut in [0, 1, 63, 64, 65, 128, 999, 1000] {
            let mut h = Blake2s::new();
            h.update(&data[..cut]).update(&data[cut..]);
            assert_eq!(h.finish(), whole, "cut at {cut}");
        }
    }
}
