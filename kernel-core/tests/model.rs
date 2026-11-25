//! The reference model for capabilities, Untyped memory and address spaces
//! (`spec/TESTING.md` section 3): a slow, plain description of what every
//! capability and memory syscall must do, compared with the kernel after
//! each of a long random sequence of operations.
//!
//! The model keeps a map from slot to capability with an explicit parent
//! link, and for each address space a map from virtual page to frame. Where
//! the kernel walks linked lists and page tables, the model scans its maps.
mod support;

use silica_abi::root::{SLOT_CAPSPACE, SLOT_FIRST_UNTYPED};
use silica_abi::{Error, KEEP, ObjectType as T, Syscall as S};
use std::collections::{BTreeMap, BTreeSet};
use support::*;

/// A slot: in a CapSpace (object address, index) or inside a Thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Sid {
    Cs(u64, u64),
    Th(u64, u64),
}

#[derive(Clone, Debug, PartialEq)]
struct MCap {
    ty: T,
    rights: u8,
    bits: u8,
    obj: u64,
    badge: u64,
    parent: Option<Sid>,
    /// Frame: address space and virtual address it is mapped at.
    mapped: Option<(u64, u64)>,
    /// Untyped: bytes used.
    free: u64,
}

#[derive(Default)]
struct MAs {
    pages: u64,
    /// Page tables that exist: (level of the entry pointing to it, vaddr >> shift).
    tables: BTreeSet<(u8, u64)>,
    /// Leaves: (level, vaddr) -> physical address.
    leaves: BTreeMap<(u8, u64), u64>,
}

struct Model {
    root_cs: u64,
    slots: BTreeMap<Sid, MCap>,
    spaces: BTreeMap<u64, MAs>,
    /// Live CapSpaces made by the test: object -> slot count in bits.
    capspaces: BTreeMap<u64, u8>,
    threads: BTreeSet<u64>,
}

const USER_TOP: u64 = 0x7FFF_FFFF_F000;

impl Model {
    fn root(&self, i: u64) -> Result<Sid, Error> {
        if i == 0 || i >= 4096 { Err(Error::BadCap) } else { Ok(Sid::Cs(self.root_cs, i)) }
    }
    fn typed(&self, i: u64, ty: T, rights: u8) -> Result<(Sid, MCap), Error> {
        let s = self.root(i)?;
        let c = self.slots.get(&s).ok_or(Error::BadCap)?;
        if c.ty != ty {
            return Err(Error::WrongType);
        }
        if c.rights & rights != rights {
            return Err(Error::NoRights);
        }
        Ok((s, c.clone()))
    }
    /// Slot `index` of the CapSpace capability `cs`.
    fn slot_in(&self, cs: &MCap, index: u64) -> Result<Sid, Error> {
        if index == 0 || index >= 1 << cs.bits { Err(Error::BadCap) } else { Ok(Sid::Cs(cs.obj, index)) }
    }
    fn children(&self, s: Sid) -> Vec<Sid> {
        self.slots.iter().filter(|(_, c)| c.parent == Some(s)).map(|(k, _)| *k).collect()
    }
    fn depth(&self, s: Sid) -> u16 {
        self.slots[&s].parent.map_or(0, |p| 1 + self.depth(p))
    }
    fn unmap(&mut self, s: Sid) {
        let c = self.slots.get_mut(&s).unwrap();
        if let Some((a, v)) = c.mapped.take() {
            let (level, step) = if c.bits >= 21 { (1, 1u64 << 21) } else { (0, 4096) };
            let sp = self.spaces.get_mut(&a).unwrap();
            for i in 0..(1u64 << c.bits) / step {
                sp.leaves.remove(&(level, v + i * step));
            }
            // A table with nothing below it goes back to the pool.
            loop {
                let empty = sp.tables.iter().copied().find(|&(l, i)| {
                    let shift = 12 + 9 * l as u64;
                    !sp.leaves.keys().any(|&(lev, va)| lev < l && va >> shift == i)
                        && !sp.tables.iter().any(|&(l2, j)| l2 < l && (j << (12 + 9 * l2 as u64)) >> shift == i)
                });
                match empty {
                    Some(t) => drop(sp.tables.remove(&t)),
                    None => break,
                }
            }
        }
    }
    /// Delete a capability: its descendants first, then the object if this
    /// was the last capability to it, then whatever the object held.
    fn kill(&mut self, s: Sid) {
        if !self.slots.contains_key(&s) {
            return;
        }
        for c in self.children(s) {
            self.kill(c);
        }
        // Killing a descendant can destroy the CapSpace this very slot is
        // in (a capability stored in an object made from itself).
        if !self.slots.contains_key(&s) {
            return;
        }
        self.unmap(s);
        let cap = self.slots.remove(&s).unwrap();
        if self.slots.values().any(|c| c.ty == cap.ty && c.obj == cap.obj) {
            return;
        }
        match cap.ty {
            T::CapSpace => {
                if let Some(bits) = self.capspaces.remove(&cap.obj) {
                    for i in 1..1u64 << bits {
                        self.kill(Sid::Cs(cap.obj, i));
                    }
                }
            }
            T::Thread => {
                self.threads.remove(&cap.obj);
                for i in 0..4 {
                    self.kill(Sid::Th(cap.obj, i));
                }
            }
            T::AddressSpace => {
                self.spaces.remove(&cap.obj);
                for c in self.slots.values_mut() {
                    if c.mapped.is_some_and(|(a, _)| a == cap.obj) {
                        c.mapped = None;
                    }
                }
            }
            _ => {}
        }
    }
    fn revoke(&mut self, s: Sid) {
        for c in self.children(s) {
            self.kill(c);
        }
        if let Some(c) = self.slots.get_mut(&s)
            && c.ty == T::Untyped
        {
            c.free = 0;
        }
    }

    /// What the kernel must answer to syscall `n` with arguments `a` made
    /// by the root thread, with the model updated to match.
    fn step(&mut self, n: S, a: [u64; 6]) -> Result<(), Error> {
        match n {
            S::Retype => {
                let us = self.root(a[0])?;
                let ty = T::from_u64(a[1]).ok_or(Error::BadArg)?;
                let (_, dst) = self.typed(a[3], T::CapSpace, 2)?;
                let ut = self.slots.get(&us).ok_or(Error::BadCap)?.clone();
                if ut.ty != T::Untyped {
                    return Err(Error::WrongType);
                }
                let bits = a[2];
                let (obits, cbits) = match ty {
                    T::Untyped if (6..=47).contains(&bits) => (bits, bits),
                    T::Frame if (12..=47).contains(&bits) => (bits, bits),
                    T::CapSpace if (1..=32).contains(&bits) => (bits + 6, bits),
                    T::AddressSpace if (13..=30).contains(&bits) => (bits, bits),
                    T::Thread => (12, 12),
                    T::Endpoint | T::Notification => (6, 6),
                    T::SchedContext => (7, 7),
                    T::Vcpu => return Err(Error::Unsupported),
                    _ => return Err(Error::BadArg),
                };
                if a[5] == 0 || a[5] > 256 {
                    return Err(Error::BadArg);
                }
                if obits > ut.bits as u64 || (ty == T::Untyped && obits == ut.bits as u64) {
                    return Err(Error::NoMemory);
                }
                for i in 0..a[5] {
                    let s = self.slot_in(&dst, a[4].checked_add(i).ok_or(Error::BadCap)?)?;
                    if self.slots.contains_key(&s) {
                        return Err(Error::SlotFull);
                    }
                }
                let mut free = if self.children(us).is_empty() { 0 } else { ut.free };
                self.slots.get_mut(&us).unwrap().free = free;
                let size = 1u64 << obits;
                let start = (free + size - 1) & !(size - 1);
                let end = start + (a[5] << obits);
                if end > 1 << ut.bits {
                    return Err(Error::NoMemory);
                }
                for i in 0..a[5] {
                    let obj = ut.obj + start + (i << obits);
                    let rights = if ty == T::Frame { 7 } else { 15 };
                    let cap = MCap { ty, rights, bits: cbits as u8, obj, badge: 0, parent: Some(us), mapped: None, free: 0 };
                    self.slots.insert(self.slot_in(&dst, a[4] + i)?, cap);
                    match ty {
                        T::CapSpace => drop(self.capspaces.insert(obj, cbits as u8)),
                        T::Thread => drop(self.threads.insert(obj)),
                        T::AddressSpace => drop(self.spaces.insert(obj, MAs { pages: size / 4096, ..MAs::default() })),
                        _ => {}
                    }
                }
                free = end;
                self.slots.get_mut(&us).unwrap().free = free;
                Ok(())
            }
            S::CapCopy => {
                if a[5] > 1 {
                    return Err(Error::BadArg);
                }
                let (_, dst) = self.typed(a[0], T::CapSpace, 2)?;
                let d = self.slot_in(&dst, a[1])?;
                let s = self.root(a[2])?;
                let mut cap = self.slots.get(&s).ok_or(Error::BadCap)?.clone();
                if self.slots.contains_key(&d) {
                    return Err(Error::SlotFull);
                }
                cap.rights &= a[3] as u8;
                match cap.ty {
                    T::Untyped => return Err(Error::InUse),
                    T::Endpoint | T::Notification if a[4] != 0 => {
                        if cap.badge != 0 {
                            return Err(Error::BadArg);
                        }
                        cap.badge = a[4];
                    }
                    T::Frame => cap.mapped = None,
                    _ if a[4] != 0 => return Err(Error::BadArg),
                    _ => {}
                }
                if a[5] == 0 {
                    cap.parent = Some(s);
                }
                self.slots.insert(d, cap);
                Ok(())
            }
            S::CapMove => {
                let (_, dst) = self.typed(a[0], T::CapSpace, 2)?;
                let d = self.slot_in(&dst, a[1])?;
                let s = if a[3] == 0 {
                    self.root(a[2])?
                } else {
                    let (_, src) = self.typed(a[3], T::CapSpace, 2)?;
                    self.slot_in(&src, a[2])?
                };
                if !self.slots.contains_key(&s) {
                    return Err(Error::BadCap);
                }
                if self.slots.contains_key(&d) {
                    return Err(Error::SlotFull);
                }
                let cap = self.slots.remove(&s).unwrap();
                self.slots.insert(d, cap);
                for c in self.slots.values_mut() {
                    if c.parent == Some(s) {
                        c.parent = Some(d);
                    }
                }
                Ok(())
            }
            S::CapDelete => {
                let s = self.root(a[0])?;
                self.kill(s);
                Ok(())
            }
            S::CapRevoke => {
                let s = self.root(a[0])?;
                if !self.slots.contains_key(&s) {
                    return Err(Error::BadCap);
                }
                self.revoke(s);
                Ok(())
            }
            S::Map => {
                let (_, asp) = self.typed(a[0], T::AddressSpace, 2)?;
                let (fs, f) = self.typed(a[1], T::Frame, 0)?;
                let (vaddr, rights) = (a[2], a[3] as u8);
                if f.mapped.is_some() {
                    return Err(Error::Mapped);
                }
                if f.rights & rights != rights {
                    return Err(Error::NoRights);
                }
                if rights & 1 == 0 || rights & 6 == 6 {
                    return Err(Error::BadArg);
                }
                let (level, step) = if f.bits >= 21 { (1u8, 1u64 << 21) } else { (0u8, 4096) };
                let size = 1u64 << f.bits;
                if vaddr % step != 0 || vaddr >= USER_TOP || size > USER_TOP - vaddr {
                    return Err(Error::BadArg);
                }
                let sp = self.spaces.get_mut(&asp.obj).unwrap();
                let mut tables = sp.tables.clone();
                for i in 0..size / step {
                    let v = vaddr + i * step;
                    for l in (level + 1..4).rev() {
                        let key = (l, v >> (12 + 9 * l as u64));
                        if l == 1 && sp.leaves.contains_key(&(1, v & !((1 << 21) - 1))) {
                            return Err(Error::Overlap);
                        }
                        if !tables.contains(&key) {
                            if tables.len() as u64 + 2 >= sp.pages {
                                return Err(Error::NoMemory);
                            }
                            tables.insert(key);
                        }
                    }
                    if sp.leaves.contains_key(&(level, v)) || (level == 1 && tables.contains(&(1, v >> 21))) {
                        return Err(Error::Overlap);
                    }
                }
                sp.tables = tables;
                for i in 0..size / step {
                    sp.leaves.insert((level, vaddr + i * step), f.obj + i * step);
                }
                self.slots.get_mut(&fs).unwrap().mapped = Some((asp.obj, vaddr));
                Ok(())
            }
            S::Unmap => {
                let s = self.root(a[0])?;
                if self.slots.get(&s).is_some_and(|c| c.ty == T::AddressSpace) {
                    // A range of an address space (write): every frame mapped
                    // there that overlaps it, whole.
                    let (_, sp) = self.typed(a[0], T::AddressSpace, 2)?;
                    let end = a[1].checked_add(a[2]).ok_or(Error::BadArg)?;
                    let hit: Vec<Sid> = self
                        .slots
                        .iter()
                        .filter(|(_, c)| c.mapped.is_some_and(|(asp, v)| asp == sp.obj && v < end && v + (1u64 << c.bits) > a[1]))
                        .map(|(s, _)| *s)
                        .collect();
                    for s in hit {
                        self.unmap(s);
                    }
                    return Ok(());
                }
                let (fs, _) = self.typed(a[0], T::Frame, 0)?;
                self.unmap(fs);
                Ok(())
            }
            S::ThreadConfigure => {
                let (_, th) = self.typed(a[0], T::Thread, 2)?;
                let want = [T::CapSpace, T::AddressSpace, T::Endpoint, T::Frame];
                // Read and write to run in a CapSpace or an address space
                // or use an IPC buffer; write to send faults to an endpoint.
                let need = [3, 3, 2, 3];
                let mut src = [None; 4];
                for i in 0..4 {
                    if a[i + 1] == KEEP || a[i + 1] == 0 {
                        continue;
                    }
                    let (s, cap) = self.typed(a[i + 1], want[i], need[i])?;
                    if i == 3 && cap.bits != 12 {
                        return Err(Error::BadArg);
                    }
                    src[i] = Some(s);
                }
                for i in 0..4 {
                    if a[i + 1] == KEEP {
                        continue;
                    }
                    self.kill(Sid::Th(th.obj, i as u64));
                    if let Some(mut cap) = src[i].and_then(|s| self.slots.get(&s).cloned()) {
                        cap.mapped = None;
                        self.slots.insert(Sid::Th(th.obj, i as u64), cap);
                    }
                }
                Ok(())
            }
            _ => unreachable!("not modelled"),
        }
    }
}

const WINDOW: u64 = 2000;
const NWIN: u64 = 28;

/// Compare every slot the model knows about, and every slot of every live
/// container, with the kernel.
fn compare(m: &Machine, model: &Model, what: &str) {
    let addr = |s: Sid| match s {
        Sid::Cs(obj, i) => obj + i * 64,
        Sid::Th(t, i) => m.k.thread_view(t).slots[i as usize],
    };
    let mut expected: Vec<Sid> = (WINDOW..WINDOW + NWIN).map(|i| Sid::Cs(model.root_cs, i)).collect();
    for (&obj, &bits) in &model.capspaces {
        expected.extend((1..1u64 << bits).map(|i| Sid::Cs(obj, i)));
    }
    for &t in &model.threads {
        expected.extend((0..4).map(|i| Sid::Th(t, i)));
    }
    for s in expected {
        let v = m.k.view(addr(s));
        match (model.slots.get(&s), v) {
            (None, None) => {}
            (Some(c), Some(v)) => {
                let got = (v.cap.ty, v.cap.rights.0, v.cap.bits, v.cap.obj);
                assert_eq!(got, (c.ty, c.rights, c.bits, c.obj), "{what}: slot {s:?}");
                if matches!(c.ty, T::Endpoint | T::Notification) {
                    assert_eq!(v.cap.badge, c.badge, "{what}: badge of {s:?}");
                }
                assert_eq!(v.mapped, c.mapped, "{what}: mapping of {s:?}");
                assert_eq!(v.depth, model.depth(s), "{what}: depth of {s:?}");
                if c.ty == T::Untyped {
                    assert_eq!(v.watermark.0, c.free, "{what}: watermark of {s:?}");
                    assert!(v.watermark.1 <= 1 << c.bits, "{what}: cleared mark of {s:?}");
                }
                // The derivation list is consistent around this slot.
                if v.next != 0 {
                    assert_eq!(m.k.view(v.next).expect("next is a capability").prev, addr(s), "{what}: links after {s:?}");
                }
                if let Some(p) = c.parent {
                    // The parent is the nearest earlier slot with a smaller depth.
                    let mut at = v.prev;
                    while m.k.view(at).unwrap().depth >= v.depth {
                        at = m.k.view(at).unwrap().prev;
                    }
                    assert_eq!(at, addr(p), "{what}: parent of {s:?}");
                }
            }
            (want, got) => panic!("{what}: slot {s:?}: model has {want:?}, kernel has {got:?}"),
        }
    }
    for (&a, sp) in &model.spaces {
        for (&(level, v), &p) in &sp.leaves {
            let probe = v + if level == 1 { 0x12_3456 } else { 0x456 };
            assert_eq!(m.k.translate(a, probe), Some(p + (probe - v)), "{what}: translation of {probe:#x}");
        }
        for v in VADDRS {
            let covered = sp.leaves.contains_key(&(0, v)) || sp.leaves.contains_key(&(1, v & !((1 << 21) - 1)));
            assert_eq!(m.k.translate(a, v).is_some(), covered, "{what}: {v:#x} mapped or not");
        }
    }
}

const VADDRS: [u64; 8] = [0x10000, 0x11000, 0x12000, 0x20_0000, 0x40_0000, 0x4000_0000, 0x80_4020_0000, 0x7FFF_FFE0_0000];

fn run(seed: u64, ops: u32) {
    let m = Machine::new(if cfg!(miri) { 8 } else { 64 });
    let (ut, d) = m.big_untyped();
    let mut model =
        Model { root_cs: m.cs.obj, slots: BTreeMap::new(), spaces: BTreeMap::new(), capspaces: BTreeMap::new(), threads: BTreeSet::new() };
    let rootcap = |ty: T, bits: u8, obj: u64| MCap { ty, rights: 15, bits, obj, badge: 0, parent: None, mapped: None, free: 0 };
    model.slots.insert(Sid::Cs(m.cs.obj, ut), rootcap(T::Untyped, d.bits, d.base));
    // The root CapSpace capability is used as a destination; the model
    // only needs to know it is there.
    model.slots.insert(Sid::Cs(m.cs.obj, SLOT_CAPSPACE), rootcap(T::CapSpace, 12, m.cs.obj));
    assert!(ut >= SLOT_FIRST_UNTYPED);

    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for step in 0..ops {
        let has = |model: &Model, i: u64| model.slots.contains_key(&Sid::Cs(m.cs.obj, i));
        // A window slot: usually one that holds a capability (`full`) or one that does not.
        let w = |r: &mut Rng, model: &Model, full: bool| {
            let fit: Vec<u64> = (WINDOW..WINDOW + NWIN).filter(|&i| has(model, i) == full).collect();
            if fit.is_empty() || r.below(8) == 0 { WINDOW + r.below(NWIN) } else { r.pick(&fit) }
        };
        // A window slot holding a capability of type `ty`, most of the time.
        let of = |r: &mut Rng, model: &Model, ty: T| {
            let have: Vec<u64> =
                (WINDOW..WINDOW + NWIN).filter(|&i| model.slots.get(&Sid::Cs(m.cs.obj, i)).is_some_and(|c| c.ty == ty)).collect();
            if have.is_empty() || r.below(8) == 0 { WINDOW + r.below(NWIN) } else { r.pick(&have) }
        };
        let cs = |r: &mut Rng, model: &Model| {
            if r.below(3) == 0 { of(r, model, T::CapSpace) } else { SLOT_CAPSPACE }
        };
        let idx = |r: &mut Rng, model: &Model, c: u64| {
            if c == SLOT_CAPSPACE { w(r, model, false) } else { r.below(9) }
        };
        let (n, a) = match r.below(100) {
            0..30 => {
                let ty = r.pick(&[
                    T::Untyped,
                    T::Frame,
                    T::Frame,
                    T::Frame,
                    T::CapSpace,
                    T::Thread,
                    T::Endpoint,
                    T::Notification,
                    T::AddressSpace,
                    T::AddressSpace,
                ]);
                let bits = match ty {
                    T::Untyped => r.pick(&[12, 14, 16, 20, 22, 22, 5, d.bits as u64]),
                    T::Frame => r.pick(&[12, 12, 12, 12, 13, 21, 21, 11]),
                    T::CapSpace => r.pick(&[1, 2, 3, 3, 0]),
                    T::AddressSpace => r.pick(&[13, 14, 15, 16, 16, 12]),
                    _ => 0,
                };
                let from = if r.below(3) == 0 { of(&mut r, &model, T::Untyped) } else { ut };
                let c = cs(&mut r, &model);
                let i = idx(&mut r, &model, c);
                let tyn = match r.below(40) {
                    0 => 77,
                    1 => T::SchedContext as u64,
                    _ => ty as u64,
                };
                (S::Retype, [from, tyn, bits, c, i, r.pick(&[1, 1, 1, 1, 2, 3, 0, 300])])
            }
            30..55 => {
                let c = cs(&mut r, &model);
                let i = idx(&mut r, &model, c);
                let badge = r.pick(&[0, 0, 0, 5, 9]);
                (S::CapCopy, [c, i, w(&mut r, &model, true), r.pick(&[15, 15, 15, 3, 1, 2, 7]), badge, r.pick(&[0, 0, 0, 1, 1, 1, 2])])
            }
            55..63 => {
                let c = cs(&mut r, &model);
                let i = idx(&mut r, &model, c);
                if r.below(3) == 0 {
                    // Out of a CapSpace named, at one of its indexes.
                    let from = cs(&mut r, &model);
                    let at = idx(&mut r, &model, from);
                    (S::CapMove, [c, i, at, from, 0, 0])
                } else {
                    (S::CapMove, [c, i, w(&mut r, &model, true), 0, 0, 0])
                }
            }
            63..70 => (S::CapDelete, [w(&mut r, &model, true), 0, 0, 0, 0, 0]),
            70..73 => (S::CapRevoke, [if r.below(10) == 0 { ut } else { w(&mut r, &model, true) }, 0, 0, 0, 0, 0]),
            73..90 => {
                let (a, f) = (of(&mut r, &model, T::AddressSpace), of(&mut r, &model, T::Frame));
                (S::Map, [a, f, r.pick(&VADDRS) + r.pick(&[0, 0, 0, 0, 0, 0x800]), r.pick(&[1, 3, 3, 3, 5, 7, 2]), 0, 0])
            }
            90..93 => (S::Unmap, [of(&mut r, &model, T::Frame), 0, 0, 0, 0, 0]),
            93 => {
                let a = of(&mut r, &model, T::AddressSpace);
                (
                    S::Unmap,
                    [a, r.pick(&VADDRS) + r.pick(&[0, 0x1000, 0x800]), r.pick(&[0, 1, 0x1000, 0x20_0000, 1 << 30, u64::MAX]), 0, 0, 0],
                )
            }
            _ => {
                let mut f = |ty| match r.below(4) {
                    0 => KEEP,
                    1 => 0,
                    _ => of(&mut r, &model, ty),
                };
                let (c, a, e, b) = (f(T::CapSpace), f(T::AddressSpace), f(T::Endpoint), f(T::Frame));
                (S::ThreadConfigure, [of(&mut r, &model, T::Thread), c, a, e, b, 0])
            }
        };
        // Sometimes an interrupt is pending, so long operations restart.
        if r.below(10) == 0 {
            m.k.hal.pending.set(r.below(4) as u32);
        }
        if std::env::var_os("MODEL_DUMP").is_some_and(|v| v.to_str() == Some(&step.to_string())) {
            for (k, c) in &model.slots {
                eprintln!("model {k:x?}: {:?} obj {:x} parent {:x?}", c.ty, c.obj, c.parent);
            }
            let mut at = m.k.slot_addr(&m.cs, ut).unwrap();
            while at != 0 {
                let v = m.k.view(at).unwrap();
                eprintln!("kernel {at:x}: depth {} {:?} obj {:x}", v.depth, v.cap.ty, v.cap.obj);
                at = v.next;
            }
        }
        let want = model.step(n, a);
        let got = m.sys(n, a);
        assert!(!got.blocked);
        let what = format!("seed {seed} step {step} {n:?}{a:x?}");
        assert_eq!(Error::result(got.status), want, "{what}");
        let trace = std::env::var_os("MODEL_TRACE").is_some();
        if trace {
            eprintln!("{what} -> {want:?}");
        }
        if trace || step % 16 == 0 || step + 1 == ops {
            compare(&m, &model, &what);
        }
    }
    // Nothing leaks and the Untyped balances: after revoking it, every
    // window slot is empty and it can be retyped whole again.
    model.step(S::CapRevoke, [ut, 0, 0, 0, 0, 0]).unwrap();
    m.ok(S::CapRevoke, [ut, 0, 0, 0, 0, 0]);
    compare(&m, &model, "after the final revoke");
    for i in WINDOW..WINDOW + NWIN {
        assert!(m.cap(i).is_none(), "seed {seed}: slot {i} survived revoking the Untyped everything came from");
    }
    m.retype(ut, T::Frame, d.bits as u64, WINDOW, 1);
}

#[test]
fn kernel_matches_reference_model() {
    let (seeds, ops) = if cfg!(miri) { (2, 150) } else { (48, 3000) };
    if let Some(seed) = std::env::var("MODEL_SEED").ok().and_then(|s| s.parse().ok()) {
        return run(seed, ops);
    }
    for seed in 1..=seeds {
        run(seed, ops);
    }
}
