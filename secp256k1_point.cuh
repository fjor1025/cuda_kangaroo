// secp256k1_point.cuh — Device-side secp256k1 point arithmetic (Phase 4b),
// built on the field arithmetic verified on real hardware in Phase 4a
// (secp256k1_field.cuh -- confirmed PASS on all 3200 test vectors on the
// HiveOS rig, RTX 3080 / sm_86 / CUDA 11.8).
//
// This mirrors point_add() and scalar_mult() in secp256k1.py exactly,
// formula-for-formula. Since the field operations underneath are already
// hardware-verified, the main risk here is a translation error in the
// point-addition FORMULA or its branch conditions (identity, doubling,
// P + (-P) == infinity), not numerical overflow -- there's no new
// carry-propagation risk introduced at this layer.

#pragma once
#include "secp256k1_field.cuh"

struct ECPoint {
    u256 x;
    u256 y;
    bool infinity;
};

__device__ __forceinline__ u256 u256_from_u64(uint64_t v) {
    u256 r; r.limb[0] = v; r.limb[1] = 0; r.limb[2] = 0; r.limb[3] = 0;
    return r;
}

__device__ __forceinline__ bool u256_equal(const u256& a, const u256& b) {
    for (int i = 0; i < NUM_LIMBS; i++) if (a.limb[i] != b.limb[i]) return false;
    return true;
}

__device__ __forceinline__ bool u256_is_zero(const u256& a) {
    for (int i = 0; i < NUM_LIMBS; i++) if (a.limb[i] != 0) return false;
    return true;
}

__device__ __forceinline__ ECPoint make_infinity() {
    ECPoint inf;
    inf.x = u256_from_u64(0);
    inf.y = u256_from_u64(0);
    inf.infinity = true;
    return inf;
}

// Full affine point addition. Mirrors point_add() in secp256k1.py's
// branch structure exactly:
//   1. either operand is infinity -> return the other
//   2. same x, and (different y OR y==0) -> P2 == -P1, sum is infinity
//   3. same x, same y -> doubling formula (lam = 3x^2 / 2y)
//   4. distinct points -> standard chord formula (lam = (y2-y1)/(x2-x1))
__device__ __forceinline__ ECPoint point_add(const ECPoint& p1, const ECPoint& p2) {
    if (p1.infinity) return p2;
    if (p2.infinity) return p1;

    bool same_x = u256_equal(p1.x, p2.x);
    bool same_y = u256_equal(p1.y, p2.y);

    if (same_x && (!same_y || u256_is_zero(p1.y))) {
        return make_infinity();
    }

    u256 lam;
    if (same_x && same_y) {
        // Doubling: lam = 3*x^2 / (2*y)   (curve parameter A=0 for secp256k1)
        u256 three = u256_from_u64(3);
        u256 two = u256_from_u64(2);
        u256 x_sq = mul_mod(p1.x, p1.x);
        u256 numerator = mul_mod(three, x_sq);
        u256 denominator = mul_mod(two, p1.y);
        lam = mul_mod(numerator, inv_mod(denominator));
    } else {
        // Distinct points: lam = (y2-y1) / (x2-x1)
        u256 numerator = sub_mod(p2.y, p1.y);
        u256 denominator = sub_mod(p2.x, p1.x);
        lam = mul_mod(numerator, inv_mod(denominator));
    }

    u256 lam_sq = mul_mod(lam, lam);
    u256 x3 = sub_mod(sub_mod(lam_sq, p1.x), p2.x);
    u256 y3 = sub_mod(mul_mod(lam, sub_mod(p1.x, x3)), p1.y);

    ECPoint result;
    result.x = x3;
    result.y = y3;
    result.infinity = false;
    return result;
}

// Double-and-add scalar multiplication, mirroring scalar_mult() in
// secp256k1.py: processes the scalar from its least-significant bit
// upward, exactly the same bit order as pow_mod() in the field layer.
__device__ __forceinline__ ECPoint scalar_mult(const u256& k, const ECPoint& point) {
    ECPoint result = make_infinity();
    ECPoint addend = point;
    for (int limb_i = 0; limb_i < NUM_LIMBS; limb_i++) {
        uint64_t e = k.limb[limb_i];
        for (int bit = 0; bit < 64; bit++) {
            if (e & 1ULL) {
                result = point_add(result, addend);
            }
            addend = point_add(addend, addend);
            e >>= 1;
        }
    }
    return result;
}

__device__ __forceinline__ ECPoint secp256k1_generator() {
    ECPoint g;
    // Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    g.x.limb[0] = 0x59F2815B16F81798ULL;
    g.x.limb[1] = 0x029BFCDB2DCE28D9ULL;
    g.x.limb[2] = 0x55A06295CE870B07ULL;
    g.x.limb[3] = 0x79BE667EF9DCBBACULL;
    // Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    g.y.limb[0] = 0x9C47D08FFB10D4B8ULL;
    g.y.limb[1] = 0xFD17B448A6855419ULL;
    g.y.limb[2] = 0x5DA4FBFC0E1108A8ULL;
    g.y.limb[3] = 0x483ADA7726A3C465ULL;
    g.infinity = false;
    return g;
}
