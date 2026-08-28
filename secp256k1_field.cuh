// secp256k1_field.cuh — Device-side secp256k1 field arithmetic for sm_86
// (RTX 3080), CUDA 11.8.
//
// IMPORTANT CONTEXT: this was written and algorithm-verified in a sandbox
// with no CUDA toolkit and no GPU -- I could not compile or run this
// myself. The underlying algorithm (4x64-bit limb representation,
// schoolbook multiply, secp256k1's fold-based fast reduction) was
// verified against thousands of random test cases in Python first (see
// ../limb_sim.py), checked against secp256k1.py's trusted bigint
// arithmetic. What's NOT verified yet is the CUDA transcription itself --
// syntax correctness, carry-propagation edge cases in this specific
// implementation, and whether it actually compiles for sm_86. That
// verification has to happen on the actual rig; test_field_arithmetic.cu
// in this directory runs the same test vectors (test_vectors.h,
// generated from the same trusted Python reference) on the GPU and
// reports PASS/FAIL exactly like the Python self-tests did.
//
// secp256k1 prime: p = 2^256 - 2^32 - 977
// Fast reduction identity: 2^256 ≡ 2^32 + 977 (mod p)
// For a 512-bit product P = P_hi*2^256 + P_lo:  P mod p ≡ P_lo + P_hi*(2^32+977)

#pragma once
#include <cstdint>
#include <cstdio>

#define NUM_LIMBS 4
#define FOLD_CONST 0x1000003D1ULL  // 2^32 + 977

struct u256 {
    uint64_t limb[NUM_LIMBS];  // little-endian: limb[0] is least significant
};

// secp256k1 prime p, as little-endian limbs.
__device__ __forceinline__ u256 secp256k1_p() {
    u256 p;
    p.limb[0] = 0xFFFFFFFEFFFFFC2FULL;
    p.limb[1] = 0xFFFFFFFFFFFFFFFFULL;
    p.limb[2] = 0xFFFFFFFFFFFFFFFFULL;
    p.limb[3] = 0xFFFFFFFFFFFFFFFFULL;
    return p;
}

// ---- Comparison ----

__device__ __forceinline__ bool ge_u256(const u256& a, const u256& b) {
    for (int i = NUM_LIMBS - 1; i >= 0; i--) {
        if (a.limb[i] != b.limb[i]) return a.limb[i] > b.limb[i];
    }
    return true;  // equal counts as >=
}

// ---- Addition / subtraction on raw (unreduced) 256-bit values ----
// Returns the carry-out (0 or 1) so callers needing 257-bit precision
// (e.g. the fold-reduction loop) don't lose it.

__device__ __forceinline__ uint64_t add_u256_raw(u256& result, const u256& a, const u256& b) {
    uint64_t carry = 0;
    for (int i = 0; i < NUM_LIMBS; i++) {
        uint64_t s1 = a.limb[i] + b.limb[i];
        uint64_t c1 = (s1 < a.limb[i]) ? 1 : 0;
        uint64_t s2 = s1 + carry;
        uint64_t c2 = (s2 < s1) ? 1 : 0;
        result.limb[i] = s2;
        carry = c1 + c2;  // c1, c2 in {0,1}; c1+c2 in {0,1,2}, but c1==1
                            // implies s1 wrapped to a value < a.limb[i] <=
                            // UINT64_MAX-1 in the general case... to stay
                            // safe or account for both bits of carry, we
                            // let carry accumulate as an integer (max 2)
                            // rather than assuming it's a single bit; the
                            // next iteration's a.limb[i]+b.limb[i] + carry
                            // arithmetic below in sub_u256_raw and the
                            // multiply accumulator handles carry > 1
                            // correctly since it's just added as a uint64.
    }
    return carry;
}

__device__ __forceinline__ uint64_t sub_u256_raw(u256& result, const u256& a, const u256& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < NUM_LIMBS; i++) {
        uint64_t s1 = a.limb[i] - b.limb[i];
        uint64_t b1 = (a.limb[i] < b.limb[i]) ? 1 : 0;
        uint64_t s2 = s1 - borrow;
        uint64_t b2 = (s1 < borrow) ? 1 : 0;
        result.limb[i] = s2;
        borrow = b1 + b2;
    }
    return borrow;
}

// ---- Modular add / sub (inputs and output in [0, p)) ----

__device__ __forceinline__ u256 add_mod(const u256& a, const u256& b) {
    // a, b < p, but a+b can still exceed 2^256 (whenever both are large),
    // in which case the carry bit from add_u256_raw must NOT be silently
    // dropped -- an earlier version did exactly that, which is wrong for
    // a large fraction of inputs, not just a rare edge case. Fold the
    // dropped carry back in using the same 2^256 ≡ FOLD_CONST (mod p)
    // identity the multiplication reduction uses.
    u256 result;
    u256 p = secp256k1_p();
    uint64_t carry = add_u256_raw(result, a, b);
    if (carry) {
        u256 fold; fold.limb[0] = FOLD_CONST; fold.limb[1] = 0; fold.limb[2] = 0; fold.limb[3] = 0;
        uint64_t carry2 = add_u256_raw(result, result, fold);
        if (carry2) {
            // Only possible if result was already within FOLD_CONST of
            // 2^256-1 before this add; handled the same way for rigor,
            // though this second level of overflow is not expected to
            // trigger given FOLD_CONST's small (~2^41) magnitude.
            add_u256_raw(result, result, fold);
        }
    }
    while (ge_u256(result, p)) {
        sub_u256_raw(result, result, p);
    }
    return result;
}

__device__ __forceinline__ u256 sub_mod(const u256& a, const u256& b) {
    // When a < b, compute p - (b - a) directly rather than (a + p) - b.
    // The latter computes an intermediate value that frequently exceeds
    // 256 bits (a+p overflows for almost all a, since p is so close to
    // 2^256), silently dropping the carry -- a bug found by manual
    // tracing, not compilation, since no compiler was available here.
    // Reordering to p - (b-a) never produces an intermediate outside
    // [0, p), so no overflow is possible in either subtraction.
    u256 result;
    u256 p = secp256k1_p();
    if (ge_u256(a, b)) {
        sub_u256_raw(result, a, b);
    } else {
        u256 diff;
        sub_u256_raw(diff, b, a);   // b - a; valid since b > a here, no borrow
        sub_u256_raw(result, p, diff);  // p - (b-a); valid since (b-a) < p
    }
    return result;
}

// ---- Schoolbook 256x256 -> 512-bit multiplication ----
// Produces 8 result limbs. Uses __umul64hi (high 64 bits of a 64x64->128
// product) plus the natural truncation of `a*b` (low 64 bits) -- avoids
// depending on unsigned __int128 support, which is less consistently
// documented across CUDA toolkit versions than this intrinsic.

__device__ __forceinline__ void add_carry_chain(uint64_t* result, int n, int idx, uint64_t value) {
    // Add `value` into result[idx], propagating carry upward. Callers
    // guarantee idx+carry-propagation never runs past index n-1 for
    // this specific use (a 4x4-limb multiply's max value is exactly
    // 512 bits = 8 limbs, so no overflow beyond index 7 is possible).
    while (value != 0 && idx < n) {
        uint64_t sum = result[idx] + value;
        uint64_t carry = (sum < result[idx]) ? 1 : 0;
        result[idx] = sum;
        value = carry;
        idx++;
    }
}

__device__ __forceinline__ void mul_u256_raw512(uint64_t* result8, const u256& a, const u256& b) {
    for (int i = 0; i < 8; i++) result8[i] = 0;
    for (int i = 0; i < NUM_LIMBS; i++) {
        for (int j = 0; j < NUM_LIMBS; j++) {
            uint64_t lo = a.limb[i] * b.limb[j];        // low 64 bits (natural wraparound)
            uint64_t hi = __umul64hi(a.limb[i], b.limb[j]);  // high 64 bits
            add_carry_chain(result8, 8, i + j, lo);
            add_carry_chain(result8, 8, i + j + 1, hi);
        }
    }
}

// ---- Fast reduction mod p, using 2^256 ≡ 2^32+977 (mod p) ----
// Mirrors fast_reduce_secp256k1() in limb_sim.py exactly: fold the high
// half into the low half (repeat if still >= 2^256), then a bounded
// number of conditional subtractions of p.

__device__ __forceinline__ u256 reduce_512_to_mod_p(uint64_t* value8) {
    u256 result;
    // First fold: hi = value8[4..7] (the top 256 bits), lo = value8[0..3].
    // hi * FOLD_CONST can be up to ~256+40=296 bits, so we compute it
    // into a small local buffer and add it back into lo with full carry
    // propagation, looping until the high limbs are all zero.
    for (int i = 0; i < NUM_LIMBS; i++) result.limb[i] = value8[i];
    uint64_t hi[NUM_LIMBS] = {value8[4], value8[5], value8[6], value8[7]};

    // The fold can require more than one pass if the first fold's result
    // is still >= 2^256 (rare, but possible when hi itself is near its
    // maximum). Loop until hi is all zero.
    while (hi[0] || hi[1] || hi[2] || hi[3]) {
        // Compute hi * FOLD_CONST (a 256-bit x 64-bit -> up to 320-bit
        // product), add into result (256-bit), track any new overflow
        // into a fresh `hi` for another pass.
        uint64_t product[NUM_LIMBS + 1] = {0, 0, 0, 0, 0};
        for (int i = 0; i < NUM_LIMBS; i++) {
            uint64_t lo = hi[i] * FOLD_CONST;
            uint64_t carry_hi = __umul64hi(hi[i], FOLD_CONST);
            add_carry_chain(product, NUM_LIMBS + 1, i, lo);
            add_carry_chain(product, NUM_LIMBS + 1, i + 1, carry_hi);
        }
        uint64_t overflow = 0;
        {
            u256 lo256; for (int i = 0; i < NUM_LIMBS; i++) lo256.limb[i] = product[i];
            uint64_t carry = add_u256_raw(result, result, lo256);
            overflow = carry + product[NUM_LIMBS];  // carry out of the 256-bit add, plus any 5th product limb
        }
        hi[0] = overflow; hi[1] = 0; hi[2] = 0; hi[3] = 0;
    }

    // Final conditional subtraction(s) to bring result into [0, p).
    u256 p = secp256k1_p();
    while (ge_u256(result, p)) {
        sub_u256_raw(result, result, p);
    }
    return result;
}

__device__ __forceinline__ u256 mul_mod(const u256& a, const u256& b) {
    uint64_t product[8];
    mul_u256_raw512(product, a, b);
    return reduce_512_to_mod_p(product);
}

// ---- Modular inverse via Fermat's little theorem: a^(p-2) mod p ----

__device__ __forceinline__ u256 pow_mod(const u256& base_in, const u256& exponent) {
    u256 result;
    result.limb[0] = 1; result.limb[1] = 0; result.limb[2] = 0; result.limb[3] = 0;
    u256 base = base_in;
    for (int limb_i = 0; limb_i < NUM_LIMBS; limb_i++) {
        uint64_t e = exponent.limb[limb_i];
        for (int bit = 0; bit < 64; bit++) {
            if (e & 1ULL) {
                result = mul_mod(result, base);
            }
            base = mul_mod(base, base);
            e >>= 1;
        }
    }
    return result;
}

__device__ __forceinline__ u256 inv_mod(const u256& a) {
    // exponent = p - 2
    u256 p = secp256k1_p();
    u256 two; two.limb[0] = 2; two.limb[1] = 0; two.limb[2] = 0; two.limb[3] = 0;
    u256 exponent;
    sub_u256_raw(exponent, p, two);
    return pow_mod(a, exponent);
}
