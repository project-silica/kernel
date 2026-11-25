//! An intrusive balanced tree of threads in order of a key: a processor's
//! fair run queue (by virtual runtime) and its timers (by deadline).
//!
//! Both were lists kept in order, so making a thread runnable, or blocking
//! one with a deadline, walked over the threads already there with
//! interrupts off; any program can create threads from its own memory, so
//! any program could make that walk as long as it liked on the processor
//! every service shares (the 2026-11-24 audit). An AVL tree bounds it to
//! the logarithm of the number of threads, and keeps the operations the
//! scheduler does on every switch at about the cost the list had: the
//! first and last threads are kept, not searched for; a thread that goes
//! last (a yield, a preempted thread) or first, and taking the first, are
//! attached and detached where they are without a walk from the top; and
//! rebalancing stops at the first subtree whose height did not change,
//! which for those is almost always the first.
//!
//! The tree is threaded through two links each thread already has for the
//! queue concerned (its left and right child), a parent link, and a word
//! holding the node's height. Equal keys keep the order they were inserted
//! in, or a thread inserted `first` goes before every equal one already
//! there: an insertion goes right of an equal key, or left with `first`,
//! and rotations keep the order of the nodes, so no sequence number is
//! kept or compared.
//!
//! Everything here runs under the lock of the processor whose tree it is.

use crate::mem::Paddr;
use core::cell::Cell;

/// One tree's state on a processor.
pub struct Root {
    /// The root node (0: empty).
    pub(crate) top: Cell<Paddr>,
}

impl Root {
    ctor! {
        /// An empty tree.
        pub fn new() -> Root {
            Root { top: Cell::new(0) }
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
    /// The height of the subtree under the node (a leaf's is 1).
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

#[inline(always)]
fn height<T: Tree>(tr: &T, t: Paddr) -> u64 {
    if t == 0 { 0 } else { tr.node(t).meta.get() }
}

/// Whether `t`, being inserted (`first`: before equals), goes before `at`.
#[inline(always)]
fn goes_before<T: Tree>(tr: &T, t: Paddr, at: Paddr, first: bool) -> bool {
    let (k, ka) = (tr.key(t), tr.key(at));
    if first { k <= ka } else { k < ka }
}

/// Put `new` where `old` hangs from `parent` (0: the root).
#[inline(always)]
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

fn set_height<T: Tree>(tr: &T, t: Paddr) -> u64 {
    let n = tr.node(t);
    let h = 1 + height(tr, n.left.get()).max(height(tr, n.right.get()));
    n.meta.set(h);
    h
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
    set_height(tr, x);
    set_height(tr, y);
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
    set_height(tr, x);
    set_height(tr, y);
    y
}

/// Bring the subtree at `t` back into balance if a side is two taller;
/// returns the node now in its place, its height set.
fn balance<T: Tree>(tr: &T, t: Paddr) -> Paddr {
    let n = tr.node(t);
    let (l, r) = (n.left.get(), n.right.get());
    let (hl, hr) = (height(tr, l), height(tr, r));
    if hl > hr + 1 {
        let nl = tr.node(l);
        if height(tr, nl.left.get()) < height(tr, nl.right.get()) {
            rotate_left(tr, l);
        }
        rotate_right(tr, t)
    } else if hr > hl + 1 {
        let nr = tr.node(r);
        if height(tr, nr.right.get()) < height(tr, nr.left.get()) {
            rotate_right(tr, r);
        }
        rotate_left(tr, t)
    } else {
        n.meta.set(1 + hl.max(hr));
        t
    }
}

/// Restore heights and balance from `t` up, after a change below it, as
/// far as a subtree's height changed: above the first one whose height is
/// what it was, nothing did.
fn rebalance<T: Tree>(tr: &T, mut t: Paddr) {
    while t != 0 {
        let before = tr.node(t).meta.get();
        let s = balance(tr, t);
        if tr.node(s).meta.get() == before {
            return;
        }
        t = tr.node(s).up.get();
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
#[inline]
pub(crate) fn insert<T: Tree>(tr: &T, t: Paddr, first: bool) {
    let n = tr.node(t);
    n.left.set(0);
    n.right.set(0);
    n.meta.set(1);
    let last = tr.last().get();
    if last == 0 {
        n.up.set(0);
        tr.root().top.set(t);
        tr.first().set(t);
        tr.last().set(t);
        return;
    }
    // The usual cases, a new last or a new first: the last node has no
    // right child and the first no left one, so it hangs there.
    let parent = if !goes_before(tr, t, last, first) {
        tr.node(last).right.set(t);
        tr.last().set(t);
        last
    } else if goes_before(tr, t, tr.first().get(), first) {
        let head = tr.first().get();
        tr.node(head).left.set(t);
        tr.first().set(t);
        head
    } else {
        insert_inside(tr, t, first)
    };
    n.up.set(parent);
    grown_under(tr, parent);
}

/// Insert `t` (in no tree), whose key is no less than every key in the
/// tree, last: what a yield does, without comparing.
#[inline]
pub(crate) fn push_last<T: Tree>(tr: &T, t: Paddr) {
    let n = tr.node(t);
    n.left.set(0);
    n.right.set(0);
    n.meta.set(1);
    let last = tr.last().get();
    tr.last().set(t);
    n.up.set(last);
    if last == 0 {
        tr.root().top.set(t);
        tr.first().set(t);
        return;
    }
    debug_assert!(tr.key(t) >= tr.key(last));
    tr.node(last).right.set(t);
    grown_under(tr, last);
}

/// A leaf was just hung under `parent`. A parent that had a child already
/// is no taller now. A leaf is now two tall and still balanced (one child,
/// a leaf): what is above it may have grown.
#[inline(always)]
fn grown_under<T: Tree>(tr: &T, parent: Paddr) {
    let p = tr.node(parent);
    if p.meta.get() == 1 {
        p.meta.set(2);
        let above = p.up.get();
        if above != 0 {
            rebalance(tr, above);
        }
    }
}

/// Hang `t` where it goes among the nodes, neither first nor last; returns
/// its parent.
#[inline(never)]
fn insert_inside<T: Tree>(tr: &T, t: Paddr, first: bool) -> Paddr {
    let mut at = tr.root().top.get();
    loop {
        let a = tr.node(at);
        let link = if goes_before(tr, t, at, first) { a.left } else { a.right };
        if link.get() == 0 {
            link.set(t);
            return at;
        }
        at = link.get();
    }
}

/// Take `t` (in the tree) out of it.
pub(crate) fn remove<T: Tree>(tr: &T, t: Paddr) {
    if tr.first().get() == t {
        pop(tr);
        return;
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
        ns.meta.set(n.meta.get());
    }
    n.left.set(0);
    n.right.set(0);
    n.up.set(0);
    rebalance(tr, from);
}

/// Take the first node out and return it (0: the tree is empty).
#[inline]
pub(crate) fn pop<T: Tree>(tr: &T) -> Paddr {
    let t = tr.first().get();
    if t == 0 {
        return 0;
    }
    // The first node has no left child: its right subtree, if any, takes
    // its place, and the next is the leftmost there, or its parent.
    let n = tr.node(t);
    let (r, up) = (n.right.get(), n.up.get());
    replace_child(tr, up, t, r);
    let mut f = if r != 0 { r } else { up };
    if r != 0 {
        while tr.node(f).left.get() != 0 {
            f = tr.node(f).left.get();
        }
    }
    tr.first().set(f);
    if f == 0 {
        tr.last().set(0);
    }
    n.right.set(0);
    n.up.set(0);
    if up != 0 {
        rebalance(tr, up);
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
        if (l != 0 && tr.key(l) > tr.key(t)) || (r != 0 && tr.key(t) > tr.key(r)) {
            return Err("an ordered tree is out of order");
        }
        let (hl, hr) = (walk(tr, l, t, n, most)?, walk(tr, r, t, n, most)?);
        if hl.abs_diff(hr) > 1 {
            return Err("an ordered tree is out of balance");
        }
        let h = 1 + hl.max(hr);
        if node.meta.get() != h {
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
    // The in-order walk is in order: a child's key against its parent's
    // does not say it for nodes further apart.
    let (mut t, mut last) = (f, 0);
    while t != 0 {
        if tr.key(t) < last {
            return Err("an ordered tree is out of order");
        }
        last = tr.key(t);
        t = next(tr, t);
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
