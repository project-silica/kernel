//! An intrusive balanced tree of threads in order of a key: a processor's
//! fair run queue (by virtual runtime) and its timers (by deadline).
//!
//! Both were lists kept in order, so making a thread runnable, or blocking
//! one with a deadline, walked over the threads already there with
//! interrupts off; any program can create threads from its own memory, so
//! any program could make that walk as long as it liked on the processor
//! every service shares (the 2026-11-24 audit). An AVL tree bounds it to
//! the logarithm of the number of threads, and keeps the operations the
//! scheduler needs at the cost they had: the first and last threads are
//! kept, not searched for.
//!
//! The tree is threaded through two links each thread already has for the
//! queue concerned (its left and right child), a parent link, and a word
//! holding the node's height (its low 8 bits) and its place among equal
//! keys (the rest). Equal keys keep the order they were inserted in, or a
//! thread inserted `first` goes before every equal one already there: the
//! sequence number of a root's insertions grows from the middle of the
//! range, and of its `first` insertions shrinks from there (2^55 of each,
//! a century at ten million a second).
//!
//! Everything here runs under the lock of the processor whose tree it is.

use crate::mem::Paddr;
use core::cell::Cell;

/// One tree's state on a processor.
pub struct Root {
    /// The root node (0: empty).
    pub(crate) top: Cell<Paddr>,
    /// The next sequence numbers for an insertion after equals (counting
    /// up) and before them (counting down).
    hi: Cell<u64>,
    lo: Cell<u64>,
}

const MIDDLE: u64 = 1 << 55;

impl Root {
    ctor! {
        /// An empty tree.
        pub fn new() -> Root {
            Root { top: Cell::new(0), hi: Cell::new(MIDDLE), lo: Cell::new(MIDDLE - 1) }
        }
    }
}

impl Default for Root {
    fn default() -> Root {
        Root::new()
    }
}

/// A node's links, as the tree's user finds them in a thread.
pub(crate) struct Node<'a> {
    pub left: &'a Cell<Paddr>,
    pub right: &'a Cell<Paddr>,
    pub up: &'a Cell<Paddr>,
    /// Height in the low 8 bits, sequence number above.
    pub meta: &'a Cell<u64>,
}

/// Where a tree's nodes and keys are, and where it keeps its first and
/// last nodes (which the scheduler reads directly).
pub(crate) trait Tree {
    fn node(&self, t: Paddr) -> Node<'_>;
    fn key(&self, t: Paddr) -> u64;
    fn root(&self) -> &Root;
    fn first(&self) -> &Cell<Paddr>;
    fn last(&self) -> &Cell<Paddr>;
}

fn height<T: Tree>(tr: &T, t: Paddr) -> u64 {
    if t == 0 { 0 } else { tr.node(t).meta.get() & 0xFF }
}

fn seq<T: Tree>(tr: &T, t: Paddr) -> u64 {
    tr.node(t).meta.get() >> 8
}

/// Whether `a` comes before `b`.
fn before<T: Tree>(tr: &T, a: Paddr, b: Paddr) -> bool {
    let (ka, kb) = (tr.key(a), tr.key(b));
    ka < kb || (ka == kb && seq(tr, a) < seq(tr, b))
}

fn fix_height<T: Tree>(tr: &T, t: Paddr) -> u64 {
    let n = tr.node(t);
    let h = 1 + height(tr, n.left.get()).max(height(tr, n.right.get()));
    n.meta.set((n.meta.get() & !0xFF) | h);
    h
}

/// Put `new` where `old` hangs from `parent` (0: the root).
fn replace_child<T: Tree>(tr: &T, parent: Paddr, old: Paddr, new: Paddr) {
    if parent == 0 {
        tr.root().top.set(new);
    } else {
        let p = tr.node(parent);
        if p.left.get() == old { p.left.set(new) } else { p.right.set(new) }
    }
    if new != 0 {
        tr.node(new).up.set(parent);
    }
}

/// Rotate `x` down to the left (its right child takes its place); returns
/// the node now in its place.
fn rotate_left<T: Tree>(tr: &T, x: Paddr) -> Paddr {
    let nx = tr.node(x);
    let y = nx.right.get();
    let ny = tr.node(y);
    let b = ny.left.get();
    replace_child(tr, nx.up.get(), x, y);
    nx.right.set(b);
    if b != 0 {
        tr.node(b).up.set(x);
    }
    ny.left.set(x);
    nx.up.set(y);
    fix_height(tr, x);
    fix_height(tr, y);
    y
}

/// The mirror of [`rotate_left`].
fn rotate_right<T: Tree>(tr: &T, x: Paddr) -> Paddr {
    let nx = tr.node(x);
    let y = nx.left.get();
    let ny = tr.node(y);
    let b = ny.right.get();
    replace_child(tr, nx.up.get(), x, y);
    nx.left.set(b);
    if b != 0 {
        tr.node(b).up.set(x);
    }
    ny.right.set(x);
    nx.up.set(y);
    fix_height(tr, x);
    fix_height(tr, y);
    y
}

/// Restore the balance from `t` up to the root.
fn rebalance<T: Tree>(tr: &T, mut t: Paddr) {
    while t != 0 {
        let n = tr.node(t);
        let (l, r) = (n.left.get(), n.right.get());
        let (hl, hr) = (height(tr, l), height(tr, r));
        if hl > hr + 1 {
            let nl = tr.node(l);
            if height(tr, nl.left.get()) < height(tr, nl.right.get()) {
                rotate_left(tr, l);
            }
            t = rotate_right(tr, t);
        } else if hr > hl + 1 {
            let nr = tr.node(r);
            if height(tr, nr.right.get()) < height(tr, nr.left.get()) {
                rotate_right(tr, r);
            }
            t = rotate_left(tr, t);
        } else {
            fix_height(tr, t);
        }
        t = tr.node(t).up.get();
    }
}

/// The node after `t` in order (0: none).
pub(crate) fn next<T: Tree>(tr: &T, t: Paddr) -> Paddr {
    let mut r = tr.node(t).right.get();
    if r != 0 {
        while tr.node(r).left.get() != 0 {
            r = tr.node(r).left.get();
        }
        return r;
    }
    let mut c = t;
    let mut p = tr.node(t).up.get();
    while p != 0 && tr.node(p).right.get() == c {
        c = p;
        p = tr.node(p).up.get();
    }
    p
}

/// The node before `t` in order (0: none).
fn prev<T: Tree>(tr: &T, t: Paddr) -> Paddr {
    let mut l = tr.node(t).left.get();
    if l != 0 {
        while tr.node(l).right.get() != 0 {
            l = tr.node(l).right.get();
        }
        return l;
    }
    let mut c = t;
    let mut p = tr.node(t).up.get();
    while p != 0 && tr.node(p).left.get() == c {
        c = p;
        p = tr.node(p).up.get();
    }
    p
}

/// Insert `t` (in no tree), after every node of an equal key or, with
/// `first`, before them.
pub(crate) fn insert<T: Tree>(tr: &T, t: Paddr, first: bool) {
    let root = tr.root();
    let s = if first {
        let s = root.lo.get();
        root.lo.set(s - 1);
        s
    } else {
        let s = root.hi.get();
        root.hi.set(s + 1);
        s
    };
    let n = tr.node(t);
    n.left.set(0);
    n.right.set(0);
    n.meta.set(s << 8 | 1);
    let mut at = root.top.get();
    if at == 0 {
        n.up.set(0);
        root.top.set(t);
        tr.first().set(t);
        tr.last().set(t);
        return;
    }
    // The usual cases, a new last or first, without a walk from the top.
    let last = tr.last().get();
    let head = tr.first().get();
    if !before(tr, t, last) {
        at = last;
    } else if before(tr, t, head) {
        at = head;
    }
    loop {
        let a = tr.node(at);
        let link = if before(tr, t, at) { a.left } else { a.right };
        if link.get() == 0 {
            link.set(t);
            n.up.set(at);
            break;
        }
        at = link.get();
    }
    if before(tr, t, head) {
        tr.first().set(t);
    }
    if !before(tr, t, last) {
        tr.last().set(t);
    }
    rebalance(tr, n.up.get());
}

/// Take `t` (in the tree) out of it.
pub(crate) fn remove<T: Tree>(tr: &T, t: Paddr) {
    if tr.first().get() == t {
        tr.first().set(next(tr, t));
    }
    if tr.last().get() == t {
        tr.last().set(prev(tr, t));
    }
    let n = tr.node(t);
    let (l, r, up) = (n.left.get(), n.right.get(), n.up.get());
    let from;
    if l == 0 || r == 0 {
        // At most one child: it takes `t`'s place.
        replace_child(tr, up, t, if l != 0 { l } else { r });
        from = up;
    } else {
        // Two: the next node in order (the leftmost of the right subtree,
        // which has no left child) takes `t`'s place.
        let mut s = r;
        while tr.node(s).left.get() != 0 {
            s = tr.node(s).left.get();
        }
        let ns = tr.node(s);
        if s == r {
            from = s;
        } else {
            from = ns.up.get();
            replace_child(tr, from, s, ns.right.get());
            ns.right.set(r);
            tr.node(r).up.set(s);
        }
        ns.left.set(l);
        tr.node(l).up.set(s);
        replace_child(tr, up, t, s);
        ns.meta.set((ns.meta.get() & !0xFF) | (n.meta.get() & 0xFF));
    }
    n.left.set(0);
    n.right.set(0);
    n.up.set(0);
    rebalance(tr, from);
}

/// Take the first node out and return it (0: the tree is empty).
pub(crate) fn pop<T: Tree>(tr: &T) -> Paddr {
    let t = tr.first().get();
    if t != 0 {
        remove(tr, t);
    }
    t
}

/// Whether the tree is well formed: parent links, heights, balance, order,
/// first and last, at most `most` nodes. For tests and fuzzing.
pub(crate) fn check<T: Tree>(tr: &T, most: usize) -> Result<usize, &'static str> {
    fn walk<T: Tree>(tr: &T, t: Paddr, up: Paddr, n: &mut usize, most: usize) -> Result<u64, &'static str> {
        if t == 0 {
            return Ok(0);
        }
        *n += 1;
        if *n > most {
            return Err("an ordered tree does not end");
        }
        let node = tr.node(t);
        if node.up.get() != up {
            return Err("an ordered tree's parent link is wrong");
        }
        let (l, r) = (node.left.get(), node.right.get());
        if (l != 0 && !before(tr, l, t)) || (r != 0 && !before(tr, t, r)) {
            return Err("an ordered tree is out of order");
        }
        let (hl, hr) = (walk(tr, l, t, n, most)?, walk(tr, r, t, n, most)?);
        if hl.abs_diff(hr) > 1 {
            return Err("an ordered tree is out of balance");
        }
        let h = 1 + hl.max(hr);
        if node.meta.get() & 0xFF != h {
            return Err("an ordered tree's height is wrong");
        }
        Ok(h)
    }
    let mut n = 0;
    let top = tr.root().top.get();
    walk(tr, top, 0, &mut n, most)?;
    let (mut f, mut l) = (top, top);
    while f != 0 && tr.node(f).left.get() != 0 {
        f = tr.node(f).left.get();
    }
    while l != 0 && tr.node(l).right.get() != 0 {
        l = tr.node(l).right.get();
    }
    if tr.first().get() != f || tr.last().get() != l {
        return Err("an ordered tree's first or last node is wrong");
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;
    use std::{assert, assert_eq};

    struct Arena {
        links: Vec<[Cell<u64>; 4]>,
        keys: Vec<Cell<u64>>,
        root: Root,
        first: Cell<Paddr>,
        last: Cell<Paddr>,
    }

    impl Tree for Arena {
        fn node(&self, t: Paddr) -> Node<'_> {
            let l = &self.links[t as usize - 1];
            Node { left: &l[0], right: &l[1], up: &l[2], meta: &l[3] }
        }
        fn key(&self, t: Paddr) -> u64 {
            self.keys[t as usize - 1].get()
        }
        fn root(&self) -> &Root {
            &self.root
        }
        fn first(&self) -> &Cell<Paddr> {
            &self.first
        }
        fn last(&self) -> &Cell<Paddr> {
            &self.last
        }
    }

    fn arena(n: usize) -> Arena {
        Arena {
            links: (0..n).map(|_| core::array::from_fn(|_| Cell::new(0))).collect(),
            keys: (0..n).map(|_| Cell::new(0)).collect(),
            root: Root::new(),
            first: Cell::new(0),
            last: Cell::new(0),
        }
    }

    fn in_order(a: &Arena) -> Vec<Paddr> {
        let mut v = Vec::new();
        let mut t = a.first.get();
        while t != 0 {
            v.push(t);
            t = next(a, t);
        }
        v
    }

    /// Random insertions (after or before equals) and removals against a
    /// list kept in order the way the scheduler's list was: same order,
    /// and the tree well formed after every step.
    #[test]
    fn matches_an_ordered_list() {
        let n = 300;
        let a = arena(n);
        let mut model: Vec<Paddr> = Vec::new();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rand = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..if cfg!(miri) { 400 } else { 20_000 } {
            let t = (rand() % n as u64) + 1;
            if let Some(i) = model.iter().position(|m| *m == t) {
                model.remove(i);
                remove(&a, t);
            } else {
                let k = rand() % 16;
                let first = rand() % 3 == 0;
                a.keys[t as usize - 1].set(k);
                let at = if first {
                    model.iter().position(|m| a.key(*m) >= k).unwrap_or(model.len())
                } else {
                    model.iter().position(|m| a.key(*m) > k).unwrap_or(model.len())
                };
                model.insert(at, t);
                insert(&a, t, first);
            }
            assert_eq!(check(&a, n), Ok(model.len()));
            assert_eq!(in_order(&a), model);
        }
        while let Some(&f) = model.first() {
            assert_eq!(pop(&a), f);
            model.remove(0);
            assert_eq!(check(&a, n), Ok(model.len()));
        }
        assert_eq!(pop(&a), 0);
    }

    /// Ten thousand nodes in rising order stay a tree of height at most
    /// 1.44 log2(n): no insertion walks far.
    #[test]
    fn stays_shallow() {
        let n = if cfg!(miri) { 300 } else { 10_000 };
        let a = arena(n);
        for t in 1..=n as u64 {
            a.keys[t as usize - 1].set(t / 3);
            insert(&a, t, false);
        }
        assert_eq!(check(&a, n), Ok(n));
        assert!(height(&a, a.root.top.get()) <= 20);
        assert!(height(&a, a.root.top.get()) as f64 <= 1.45 * (n as f64).log2() + 1.0);
    }
}
