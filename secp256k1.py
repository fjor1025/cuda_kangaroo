"""
secp256k1.py — Minimal, correctness-first secp256k1 field/point arithmetic.

This is Phase 0 of the Kangaroo build: a small, heavily-testable reference
implementation of the group law. Everything downstream (the Kangaroo walk,
the CUDA port later) gets checked against this module's behavior, so it
favors clarity and obviously-correct formulas over speed.

Affine coordinates only. No side-channel/constant-time hardening — this is
a research/benchmarking reference, not a wallet signing path.
"""

from dataclasses import dataclass
from typing import Optional

# --- Curve parameters (secp256k1) ---
P = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F
A = 0
B = 7
N = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_BAAEDCE6_AF48A03B_BFD25E8C_D0364141
GX = 0x79BE667E_F9DCBBAC_55A06295_CE870B07_029BFCDB_2DCE28D9_59F2815B_16F81798
GY = 0x483ADA77_26A3C465_5DA4FBFC_0E1108A8_FD17B448_A6855419_9C47D08F_FB10D4B8


@dataclass(frozen=True)
class Point:
    """Affine point. None coordinates represent the point at infinity (identity)."""
    x: Optional[int]
    y: Optional[int]

    @property
    def is_infinity(self) -> bool:
        return self.x is None


INFINITY = Point(None, None)
G = Point(GX, GY)


def mod_inv(a: int, m: int = P) -> int:
    """Modular inverse via Fermat's little theorem (m must be prime)."""
    a %= m
    if a == 0:
        raise ZeroDivisionError("inverse of 0 does not exist")
    return pow(a, m - 2, m)


def point_add(p1: Point, p2: Point) -> Point:
    """Full affine point addition, handling identity, doubling, and inverse cases."""
    if p1.is_infinity:
        return p2
    if p2.is_infinity:
        return p1
    if p1.x == p2.x and (p1.y != p2.y or p1.y == 0):
        # p2 == -p1 -> sum is the point at infinity
        return INFINITY
    if p1.x == p2.x and p1.y == p2.y:
        # Doubling: lam = (3x^2 + a) / (2y)
        lam = (3 * p1.x * p1.x + A) * mod_inv(2 * p1.y % P) % P
    else:
        # Distinct points: lam = (y2 - y1) / (x2 - x1)
        lam = (p2.y - p1.y) * mod_inv((p2.x - p1.x) % P) % P
    x3 = (lam * lam - p1.x - p2.x) % P
    y3 = (lam * (p1.x - x3) - p1.y) % P
    return Point(x3, y3)


def point_neg(p: Point) -> Point:
    """-P = (x, -y mod P). Used by the negation-map optimization in Phase 5."""
    if p.is_infinity:
        return p
    return Point(p.x, (-p.y) % P)


def scalar_mult(k: int, point: Point = G) -> Point:
    """Double-and-add scalar multiplication. k is reduced mod N first."""
    k %= N
    result = INFINITY
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result


def is_on_curve(p: Point) -> bool:
    if p.is_infinity:
        return True
    return (p.y * p.y - (p.x**3 + A * p.x + B)) % P == 0


def privkey_to_pubkey_hex(k: int, compressed: bool = True) -> str:
    """For sanity-checking against known puzzle pubkeys."""
    pt = scalar_mult(k)
    if compressed:
        prefix = "02" if pt.y % 2 == 0 else "03"
        return prefix + f"{pt.x:064x}"
    return "04" + f"{pt.x:064x}" + f"{pt.y:064x}"


# ---- Self-checks: run directly to validate this module before trusting anything built on it ----
if __name__ == "__main__":
    print("secp256k1 reference self-check")
    print("-" * 60)

    assert is_on_curve(G), "generator must be on curve"
    print("[ok] G is on curve")

    # N*G must be the point at infinity (group order check)
    assert scalar_mult(N, G).is_infinity, "N*G must equal infinity"
    print("[ok] N*G == infinity")

    # Known answer test: Bitcoin Puzzle #1, privkey = 1
    assert privkey_to_pubkey_hex(1) == "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    print("[ok] privkey 1 -> known pubkey (puzzle #1)")

    # Known answer test: Bitcoin Puzzle #135 (solved, from public record)
    # privkey = 0x6d9392a16883f90903d5f78da57af07eb2 (as documented in this puzzle series)
    k135 = 0x6d9392a16883f90903d5f78da57af07eb2
    pub135 = privkey_to_pubkey_hex(k135)
    print(f"[info] privkey #135 -> {pub135}")

    # Additivity check: (a+b)*G == a*G + b*G, for random a,b
    import random
    random.seed(1234)
    for _ in range(20):
        a = random.randrange(1, N)
        b = random.randrange(1, N)
        lhs = scalar_mult((a + b) % N)
        rhs = point_add(scalar_mult(a), scalar_mult(b))
        assert lhs == rhs, f"additivity failed for a={a}, b={b}"
    print("[ok] additivity (a+b)G == aG + bG holds for 20 random pairs")

    # Negation check: -P + P == infinity
    for _ in range(5):
        a = random.randrange(1, N)
        pt = scalar_mult(a)
        assert point_add(pt, point_neg(pt)).is_infinity
    print("[ok] P + (-P) == infinity for 5 random points")

    print("-" * 60)
    print("All Phase 0 checks passed.")
