//! ACPI tables, read from untrusted memory.
//!
//! [`find_table`] walks from the RSDP through the XSDT to a table with a
//! given signature, reading physical memory through a function the caller
//! supplies (the kernel's map of memory, a root task's mapping of firmware
//! memory, a test's byte array). [`devices`] finds, in a DSDT or SSDT, the
//! devices with a given hardware ID whose `_CRS` is a fixed resource
//! template, and the memory range and interrupt it names: the subset of AML
//! that firmware uses for simple on-chip devices. Anything else (a `_CRS`
//! computed by a method, a `_HID` given as an EISA number) is not found here
//! and needs the AML interpreter (E03). Every length and offset is checked;
//! malformed input yields nothing, never a panic.

/// Reads `buf.len()` bytes of physical memory at the address given; false
/// when that memory cannot be read.
pub type Read<'a> = &'a mut dyn FnMut(u64, &mut [u8]) -> bool;

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// The physical address and length of the table with signature `sig`
/// (the first one, in XSDT order), found from the RSDP at `rsdp`.
pub fn find_table(read: Read, rsdp: u64, sig: &[u8; 4]) -> Option<(u64, u32)> {
    let mut r = [0u8; 36];
    if rsdp == 0 || !read(rsdp, &mut r) || &r[..8] != b"RSD PTR " || r[15] < 2 {
        return None;
    }
    let xsdt = u64_at(&r, 24)?;
    let mut h = [0u8; 36];
    if xsdt == 0 || !read(xsdt, &mut h) || &h[..4] != b"XSDT" {
        return None;
    }
    let len = u32_at(&h, 4)?;
    let entries = (len.checked_sub(36)? / 8).min(256);
    for i in 0..entries as u64 {
        let mut e = [0u8; 8];
        if !read(xsdt + 36 + i * 8, &mut e) {
            return None;
        }
        let t = u64::from_le_bytes(e);
        let mut th = [0u8; 8];
        if t != 0 && read(t, &mut th) && &th[..4] == sig {
            let tlen = u32_at(&th, 4)?;
            return (tlen >= 36).then_some((t, tlen));
        }
    }
    None
}

/// The FADT's DSDT: the physical address of the table that describes the
/// machine's devices.
pub fn dsdt(read: Read, rsdp: u64) -> Option<(u64, u32)> {
    let (fadt, len) = find_table(read, rsdp, b"FACP")?;
    let mut f = [0u8; 148];
    let n = (len as usize).min(f.len());
    if !read(fadt, &mut f[..n]) {
        return None;
    }
    // X_DSDT (64-bit, offset 140) when present, else DSDT (offset 40).
    let x = if n >= 148 { u64_at(&f, 140)? } else { 0 };
    let at = if x != 0 { x } else { u32_at(&f, 40)? as u64 };
    let mut h = [0u8; 8];
    if at == 0 || !read(at, &mut h) || &h[..4] != b"DSDT" {
        return None;
    }
    Some((at, u32_at(&h, 4)?))
}

/// A device's resources, from its `_CRS` template.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Resources {
    /// The first memory range: base and length (0, 0 when none).
    pub mem: (u64, u64),
    /// The first interrupt (a global system interrupt number), if any.
    pub irq: Option<u32>,
}

/// Every device in the AML table `aml` (a whole DSDT or SSDT, header
/// included) whose `_HID` is the string `hid`, with the resources its
/// `_CRS` resource template names.
pub fn devices<'a>(aml: &'a [u8], hid: &'a [u8]) -> impl Iterator<Item = Resources> + 'a {
    // A `_HID` declared as a string: NameOp, the name, StringPrefix, the
    // characters, the terminating zero.
    let mut pat = [0u8; 32];
    let plen = 6 + hid.len() + 1;
    let ok = hid.len() <= 24 && !hid.is_empty();
    if ok {
        pat[..6].copy_from_slice(b"\x08_HID\x0D");
        pat[6..6 + hid.len()].copy_from_slice(hid);
    }
    let mut at = 36.min(aml.len());
    core::iter::from_fn(move || {
        if !ok {
            return None;
        }
        while at + plen <= aml.len() {
            let here = at;
            at += 1;
            if aml[here..here + plen] != pat[..plen] {
                continue;
            }
            // The `_CRS` of the same device follows within its scope: the
            // next `_CRS` before the next `_HID`.
            let rest = &aml[here + plen..];
            let next_hid = find(rest, b"\x08_HID").unwrap_or(rest.len());
            let Some(c) = find(&rest[..next_hid], b"\x08_CRS\x11") else { continue };
            if let Some(r) = crs(&rest[c + 6..next_hid]) {
                return Some(r);
            }
        }
        None
    })
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// An AML package length at the start of `b`: (length, bytes it took).
fn pkg_len(b: &[u8]) -> Option<(usize, usize)> {
    let lead = *b.first()?;
    let extra = (lead >> 6) as usize;
    if extra == 0 {
        return Some(((lead & 0x3F) as usize, 1));
    }
    let mut n = (lead & 0x0F) as usize;
    for i in 0..extra {
        n |= (*b.get(1 + i)? as usize) << (4 + 8 * i);
    }
    Some((n, 1 + extra))
}

/// A Buffer (after its BufferOp) holding a resource template.
fn crs(b: &[u8]) -> Option<Resources> {
    let (len, used) = pkg_len(b)?;
    let body = b.get(used..len)?;
    // The buffer size, as a constant (BytePrefix, WordPrefix or DWordPrefix).
    let skip = match *body.first()? {
        0x0A => 2,
        0x0B => 3,
        0x0C => 5,
        _ => return None,
    };
    let mut d = body.get(skip..)?;
    let mut r = Resources::default();
    while let Some(&tag) = d.first() {
        let (kind, n, head) = if tag & 0x80 != 0 {
            (tag, u16::from_le_bytes(d.get(1..3)?.try_into().ok()?) as usize, 3)
        } else {
            (tag >> 3, (tag & 7) as usize, 1)
        };
        let v = d.get(head..head + n)?;
        match kind {
            // End tag.
            0x0F if tag & 0x80 == 0 => break,
            // Memory32Fixed: write flag, base, length.
            0x86 if n >= 9 && r.mem == (0, 0) => r.mem = (u32_at(v, 1)? as u64, u32_at(v, 5)? as u64),
            // QWordMemory: flags, granularity, min, max, translation, length.
            0x8A if n >= 43 && v[0] == 0 && r.mem == (0, 0) => r.mem = (u64_at(v, 11)?, u64_at(v, 35)?),
            // DWordMemory.
            0x87 if n >= 23 && v[0] == 0 && r.mem == (0, 0) => r.mem = (u32_at(v, 7)? as u64, u32_at(v, 19)? as u64),
            // Extended Interrupt: flags, count, the interrupts.
            0x89 if n >= 6 && v[1] >= 1 && r.irq.is_none() => r.irq = Some(u32_at(v, 2)?),
            _ => {}
        }
        d = &d[head + n..];
    }
    Some(r)
}

/// The sleep type values of the sleep state `name` (`b"_S5_"` for soft
/// off), from the package the AML table `aml` declares under that name:
/// `Name (_S5, Package () { a, b, ... })`, the first two elements being the
/// values for the PM1a and PM1b control registers. Only constant elements
/// (Zero, One, a byte, word or dword constant) are understood; `None` for
/// anything else, or when the name is not declared.
pub fn sleep_type(aml: &[u8], name: &[u8; 4]) -> Option<(u8, u8)> {
    let mut from = 36.min(aml.len());
    while let Some(at) = find(&aml[from..], name) {
        let here = from + at;
        from = here + 1;
        // NameOp before the name (possibly with a root prefix), PackageOp after it.
        let named = here >= 1 && (aml[here - 1] == 0x08 || (here >= 2 && aml[here - 1] == b'\\' && aml[here - 2] == 0x08));
        if !named || aml.get(here + 4) != Some(&0x12) {
            continue;
        }
        let rest = aml.get(here + 5..)?;
        let (_, used) = pkg_len(rest)?;
        let mut d = rest.get(used + 1..)?; // after the element count
        let mut values = [0u8; 2];
        for v in &mut values {
            let (value, n) = match *d.first()? {
                0x00 => (0, 1),
                0x01 => (1, 1),
                0xFF => (0xFF, 1),
                0x0A => (*d.get(1)? as u64, 2),
                0x0B => (u16::from_le_bytes(d.get(1..3)?.try_into().ok()?) as u64, 3),
                0x0C => (u32::from_le_bytes(d.get(1..5)?.try_into().ok()?) as u64, 5),
                _ => return None,
            };
            *v = value as u8;
            d = d.get(n..)?;
        }
        return Some((values[0], values[1]));
    }
    None
}

/// What the FADT says about turning the machine off and resetting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Power {
    /// I/O ports of the PM1a and PM1b control registers (0: none).
    pub pm1_control: [u16; 2],
    /// The port to write `acpi_enable` to, to put the machine in ACPI mode
    /// (0: it is in ACPI mode already, or cannot leave it).
    pub smi_command: u16,
    /// See `smi_command`.
    pub acpi_enable: u8,
    /// The reset register, when the firmware offers one in I/O space: its
    /// port and the value to write.
    pub reset: Option<(u16, u8)>,
    /// I/O ports of the PM1a and PM1b event blocks (0: none), and the
    /// length of each: a status register, then an enable register, each
    /// half of it. The fixed power button is bit 8 of both.
    pub pm1_event: [u16; 2],
    /// See `pm1_event`.
    pub pm1_event_len: u8,
    /// The general-purpose event block 0 (port, length: status registers,
    /// then as many enable registers).
    pub gpe0: (u16, u8),
    /// The interrupt the ACPI events arrive on (a global system interrupt).
    pub sci: u16,
}

/// The power registers of a FADT (the whole table).
pub fn power(fadt: &[u8]) -> Option<Power> {
    let u32_at = |at: usize| u32_at(fadt, at);
    let port = |at: usize| u32_at(at).filter(|p| *p != 0 && *p < 0x1_0000).map(|p| p as u16);
    if fadt.len() < 116 || &fadt[..4] != b"FACP" {
        return None;
    }
    let mut p = Power {
        pm1_control: [port(64).unwrap_or(0), port(68).unwrap_or(0)],
        smi_command: port(48).unwrap_or(0),
        acpi_enable: *fadt.get(52)?,
        reset: None,
        pm1_event: [port(56).unwrap_or(0), port(60).unwrap_or(0)],
        pm1_event_len: *fadt.get(88)?,
        gpe0: (port(80).unwrap_or(0), *fadt.get(92)?),
        sci: u16::from_le_bytes(fadt.get(46..48)?.try_into().ok()?),
    };
    // ACPI 2.0 and later: the extended address of PM1a's control block, a
    // generic address in I/O space, when the 32-bit one is not given.
    if p.pm1_control[0] == 0 && fadt.len() >= 184 && fadt[172] == 1 {
        let a = u64_at(fadt, 176)?;
        if a != 0 && a < 0x1_0000 {
            p.pm1_control[0] = a as u16;
        }
    }
    // The reset register: flag RESET_REG_SUP, a generic address in I/O space.
    let flags = u32_at(112)?;
    if flags & 1 << 10 != 0 && fadt.len() >= 129 && fadt[116] == 1 {
        let a = u64_at(fadt, 120)?;
        if a != 0 && a < 0x1_0000 {
            p.reset = Some((a as u16, fadt[128]));
        }
    }
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table with a header of 36 bytes and `body` after it.
    fn table(sig: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut t = vec![0u8; 36];
        t[..4].copy_from_slice(sig);
        t[4..8].copy_from_slice(&((36 + body.len()) as u32).to_le_bytes());
        t.extend_from_slice(body);
        t
    }

    /// A device as firmware describes a UART: `_HID`, then a `_CRS`
    /// buffer with a Memory32Fixed and an Extended Interrupt.
    fn uart(base: u32, gsi: u32) -> Vec<u8> {
        let mut tpl = vec![0x86, 9, 0, 1];
        tpl.extend_from_slice(&base.to_le_bytes());
        tpl.extend_from_slice(&0x1000u32.to_le_bytes());
        tpl.extend_from_slice(&[0x89, 6, 0, 0x01, 1]);
        tpl.extend_from_slice(&gsi.to_le_bytes());
        tpl.extend_from_slice(&[0x79, 0]);
        let mut buf = vec![0x0A, tpl.len() as u8];
        buf.extend_from_slice(&tpl);
        let mut d = b"\x08_HID\x0DARMH0011\x00".to_vec();
        d.extend_from_slice(b"\x08_UID\x0A\x01");
        d.extend_from_slice(b"\x08_CRS\x11");
        d.push((buf.len() + 1) as u8);
        d.extend_from_slice(&buf);
        d
    }

    #[test]
    fn finds_each_device_with_the_id_and_its_resources() {
        let mut body = uart(0x0900_0000, 33);
        body.extend_from_slice(b"\x08_HID\x0DQEMU0002\x00\x08_CRS\x11\x05\x0A\x02\x79\x00");
        body.extend_from_slice(&uart(0x0904_0000, 40));
        let dsdt = table(b"DSDT", &body);
        let found: Vec<_> = devices(&dsdt, b"ARMH0011").collect();
        assert_eq!(
            found,
            [Resources { mem: (0x0900_0000, 0x1000), irq: Some(33) }, Resources { mem: (0x0904_0000, 0x1000), irq: Some(40) }]
        );
        assert_eq!(devices(&dsdt, b"ARMH0061").count(), 0);
    }

    #[test]
    fn malformed_templates_yield_nothing_and_never_panic() {
        let good = table(b"DSDT", &uart(0x0900_0000, 33));
        for cut in 0..good.len() {
            let _ = devices(&good[..cut], b"ARMH0011").count();
        }
        let mut bad = good.clone();
        for i in 36..bad.len() {
            for v in [0x00, 0x7F, 0x80, 0xFF] {
                bad[i] = v;
                let _ = devices(&bad, b"ARMH0011").count();
                bad[i] = good[i];
            }
        }
    }

    #[test]
    fn the_soft_off_sleep_type_is_read_from_its_package() {
        // Name (_S5, Package (0x04) { 0x05, Zero, Zero, Zero }), as OVMF has it,
        // after an unrelated _S4 declared with a word constant.
        let mut body = b"\x08_S4_\x12\x08\x04\x0B\x06\x00\x00\x00\x00".to_vec();
        body.extend_from_slice(b"\x08\\_S5_\x12\x06\x04\x0A\x05\x00\x00\x00");
        let dsdt = table(b"DSDT", &body);
        assert_eq!(sleep_type(&dsdt, b"_S5_"), Some((5, 0)));
        assert_eq!(sleep_type(&dsdt, b"_S4_"), Some((6, 0)));
        assert_eq!(sleep_type(&dsdt, b"_S3_"), None);
        // A method computing it, or a name that is not a declaration, is not understood.
        let odd = table(b"DSDT", b"\x14\x0A_S5_\x00\xA4\x0A\x05xx\x08_S5_\x0A\x05");
        assert_eq!(sleep_type(&odd, b"_S5_"), None);
        for cut in 0..dsdt.len() {
            let _ = sleep_type(&dsdt[..cut], b"_S5_");
        }
    }

    #[test]
    fn the_power_registers_are_read_from_the_fadt() {
        let mut f = table(b"FACP", &[0; 244 - 36]);
        f[48..52].copy_from_slice(&0xB2u32.to_le_bytes());
        f[52] = 0xF1;
        f[64..68].copy_from_slice(&0x604u32.to_le_bytes());
        f[56..60].copy_from_slice(&0x600u32.to_le_bytes());
        f[88] = 4;
        f[80..84].copy_from_slice(&0x620u32.to_le_bytes());
        f[92] = 16;
        f[46..48].copy_from_slice(&9u16.to_le_bytes());
        f[112..116].copy_from_slice(&(1u32 << 10).to_le_bytes());
        f[116] = 1;
        f[120..128].copy_from_slice(&0xCF9u64.to_le_bytes());
        f[128] = 6;
        let p = power(&f).unwrap();
        assert_eq!(
            p,
            Power {
                pm1_control: [0x604, 0],
                smi_command: 0xB2,
                acpi_enable: 0xF1,
                reset: Some((0xCF9, 6)),
                pm1_event: [0x600, 0],
                pm1_event_len: 4,
                gpe0: (0x620, 16),
                sci: 9,
            }
        );
        // No reset register offered: none used, whatever the address says.
        f[112..116].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(power(&f).unwrap().reset, None);
        assert_eq!(power(&f[..100]), None);
    }

    #[test]
    fn walks_from_the_rsdp_to_a_table() {
        // Memory: RSDP at 0x100, XSDT at 0x200 naming FACP at 0x300 and
        // SPCR at 0x500; the FACP names the DSDT at 0x600.
        let mut mem = vec![0u8; 0x1000];
        let mut rsdp = b"RSD PTR ".to_vec();
        rsdp.resize(36, 0);
        rsdp[15] = 2;
        rsdp[24..32].copy_from_slice(&0x200u64.to_le_bytes());
        mem[0x100..0x124].copy_from_slice(&rsdp);
        let mut x = Vec::new();
        x.extend_from_slice(&0x300u64.to_le_bytes());
        x.extend_from_slice(&0x500u64.to_le_bytes());
        let xsdt = table(b"XSDT", &x);
        mem[0x200..0x200 + xsdt.len()].copy_from_slice(&xsdt);
        let mut f = vec![0u8; 112];
        f[104..112].copy_from_slice(&0x600u64.to_le_bytes());
        let facp = table(b"FACP", &f);
        mem[0x300..0x300 + facp.len()].copy_from_slice(&facp);
        let spcr = table(b"SPCR", &[0; 44]);
        mem[0x500..0x500 + spcr.len()].copy_from_slice(&spcr);
        let d = table(b"DSDT", &uart(0x0904_0000, 40));
        mem[0x600..0x600 + d.len()].copy_from_slice(&d);
        let mut read = |a: u64, b: &mut [u8]| match mem.get(a as usize..a as usize + b.len()) {
            Some(s) => {
                b.copy_from_slice(s);
                true
            }
            None => false,
        };
        assert_eq!(find_table(&mut read, 0x100, b"SPCR"), Some((0x500, 80)));
        assert_eq!(find_table(&mut read, 0x100, b"MCFG"), None);
        assert_eq!(dsdt(&mut read, 0x100), Some((0x600, d.len() as u32)));
        assert_eq!(find_table(&mut read, 0x101, b"SPCR"), None);
    }
}
