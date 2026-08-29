// kangaroo_walk.cuh — Phase 4c: the actual kangaroo-walk kernel, built on
// field arithmetic (4a) and point arithmetic (4b), both hardware-verified
// on the HiveOS rig already. This is new, NOT yet hardware-verified code.
//
// Deliberately scoped as a SINGLE-THREAD walk first (one thread runs the
// full tame+wild loop sequentially), matching kangaroo_cpu.py's solve()
// structure exactly. This proves the stepping/DP-matching logic is
// correct in isolation, before the separate (larger) problem of running
// many kangaroos in parallel with a shared, contended DP store across
// threads -- which has its own synchronization concerns and is future
// work, not this phase.

#pragma once
#include "secp256k1_point.cuh"

// ---- Distinguished-point key: matches kangaroo_cpu.dp_key() exactly --
// returns the FULL x-coordinate (not truncated) when the point qualifies.
// Deliberately NOT using the compact 8-byte truncated format from the
// Phase 3 checkpoint code -- that was a different, later design choice
// for the CPU/checkpoint use case. Here the goal is exact behavioral
// parity with kangaroo_walk_sim.py's reference simulation, which itself
// uses the full x-coordinate, so truncating here would risk a new,
// untested source of divergence between the verified Python behavior and
// this kernel.

__device__ __forceinline__ bool dp_key(const ECPoint& p, int dp_bits, u256* out_key) {
    if (p.infinity) return false;
    uint64_t mask = (dp_bits >= 64) ? ~0ULL : ((1ULL << dp_bits) - 1);
    if ((p.x.limb[0] & mask) == 0) {
        *out_key = p.x;
        return true;
    }
    return false;
}

// ---- jump_index: matches kangaroo_cpu.JumpTable.jump_index() exactly.
// Verified (see conversation/generate_walk_test_vectors.py's derivation)
// that (x * CONST) mod 2^64 depends only on x's low 64 bits, i.e. exactly
// limb[0] in this little-endian representation -- so no need to involve
// the other 3 limbs at all.

__device__ __forceinline__ int jump_index(const ECPoint& p, int table_size) {
    if (p.infinity) return 0;
    uint64_t h = p.x.limb[0] * 0x9E3779B97F4A7C15ULL;  // wraps mod 2^64 naturally
    return (int)((h >> 48) % (uint64_t)table_size);
}

// ---- Open-addressing DP hash table, mirroring
// kangaroo_walk_sim.OpenAddressingDPTable exactly (verified there against
// the dict-based CPU baseline before this was written). Capacity MUST be
// a power of two -- see the comment on DP_TABLE_CAPACITY below for why.

#define DP_TABLE_CAPACITY 65536  // must stay a power of two; see hash_index()

struct DPEntry {
    u256 key;
    uint64_t dist;     // NOTE: truncates to 64 bits, same limitation as
                        // Phase 3's checkpoint DIST_MAX. Fine for these
                        // toy-scale test ranges (distances stay far under
                        // 2^64), but would need widening to a full u256
                        // before this is used for real puzzle-scale
                        // ranges where distances can exceed 64 bits.
    uint8_t herd;      // 0 = tame, 1 = wild
    uint8_t occupied;
};

__device__ __forceinline__ uint64_t dp_hash_index(const u256& key) {
    // key % DP_TABLE_CAPACITY, exploiting that DP_TABLE_CAPACITY is a
    // power of two: reduction mod 2^k depends only on the low k bits,
    // which live entirely in limb[0] for any k <= 64. Verified against
    // Python's exact `key % capacity` for the same reason the jump_index
    // simplification above was verified.
    return key.limb[0] & (DP_TABLE_CAPACITY - 1);
}

// Returns true if a DIFFERENT herd already holds this key (a genuine
// collision candidate); false otherwise (freshly inserted, or already
// held by the SAME herd, matching the Python reference's "don't
// overwrite" behavior). On collision, writes the other entry's distance
// and herd to the output parameters.
__device__ __forceinline__ bool dp_table_insert_or_check(
        DPEntry* table, const u256& key, uint64_t dist, uint8_t herd,
        uint64_t* out_other_dist, uint8_t* out_other_herd) {
    uint64_t idx = dp_hash_index(key);
    for (uint64_t probe = 0; probe < DP_TABLE_CAPACITY; probe++) {
        uint64_t slot = (idx + probe) & (DP_TABLE_CAPACITY - 1);
        if (!table[slot].occupied) {
            table[slot].key = key;
            table[slot].dist = dist;
            table[slot].herd = herd;
            table[slot].occupied = 1;
            return false;
        }
        if (u256_equal(table[slot].key, key)) {
            if (table[slot].herd != herd) {
                *out_other_dist = table[slot].dist;
                *out_other_herd = table[slot].herd;
                return true;
            }
            return false;
        }
    }
    // Table full -- shouldn't happen for the test cases this is scoped
    // to (max_jumps chosen with headroom relative to DP_TABLE_CAPACITY).
    return false;
}

// ---- secp256k1_n(): the curve ORDER N, distinct from the field prime P
// (secp256k1_p() in secp256k1_field.cuh). Distances in the kangaroo walk
// are scalars and must eventually be reduced mod N, never mod P -- an
// earlier version of this file used add_mod/sub_mod (mod P) for distance
// tracking, which happened to not manifest as a wrong answer at the tiny
// test scale this is verified against (distances never approached either
// modulus), but was conceptually wrong and would break at real
// puzzle-scale ranges. Fixed: distances now accumulate as plain
// (non-modular) u256 values during the walk -- matching kangaroo_cpu.py,
// where distances are just unbounded Python integers -- and only the
// FINAL key is reduced, correctly, mod N.

__device__ __forceinline__ u256 secp256k1_n() {
    u256 n;
    n.limb[0] = 0xBFD25E8CD0364141ULL;
    n.limb[1] = 0xBAAEDCE6AF48A03BULL;
    n.limb[2] = 0xFFFFFFFFFFFFFFFEULL;
    n.limb[3] = 0xFFFFFFFFFFFFFFFFULL;
    return n;
}

// (a - b) mod N, for the two orderings needed by the collision formula.
// Assumes a, b < N (true for any distance value realistic on a 256-bit
// curve, since expected kangaroo distances are O(sqrt(range)) <<< N).
__device__ __forceinline__ u256 sub_mod_n(const u256& a, const u256& b) {
    u256 n = secp256k1_n();
    u256 result;
    if (ge_u256(a, b)) {
        sub_u256_raw(result, a, b);
    } else {
        u256 diff;
        sub_u256_raw(diff, b, a);
        sub_u256_raw(result, n, diff);
    }
    return result;
}

__device__ __forceinline__ bool point_equal_pub(const ECPoint& a, const ECPoint& b) {
    if (a.infinity != b.infinity) return false;
    if (a.infinity) return true;
    return u256_equal(a.x, b.x) && u256_equal(a.y, b.y);
}

// ---- The walk itself: single-thread tame+wild loop, mirroring
// kangaroo_walk_sim.solve_sim() exactly. Returns true and sets *out_key
// if a valid collision was found and verified (scalar_mult(k)==pubkey);
// false if max_jumps was exhausted first.

__device__ bool kangaroo_solve(
        const ECPoint& pubkey, const u256& a, const u256& b, int dp_bits,
        const int* exponents, int table_size, uint64_t max_jumps,
        DPEntry* dp_table, u256* out_key, uint64_t* out_jumps) {

    // Build the jump table's points from the (host-supplied, already
    // Python-verified) exponents. table_size is small (32), so 32
    // scalar_mult calls here is a one-time, cheap setup cost.
    ECPoint g = secp256k1_generator();
    ECPoint jump_points[32];  // JUMP_TABLE_SIZE from walk_test_vectors.h; kept as a
                               // literal 32 here since this file has no dependency
                               // on that generated header (keeps this file reusable
                               // across different test-vector sets).
    for (int i = 0; i < table_size; i++) {
        u256 exp_scalar = u256_from_u64(1ULL << exponents[i]);
        jump_points[i] = scalar_mult(exp_scalar, g);
    }

    // tame starts at the midpoint of [a,b); wild starts at pubkey.
    // range_size = b - a; tame_start = a + range_size/2. These are plain
    // integer computations on the puzzle bounds (not modular field
    // arithmetic), so use the raw (non-modular) add/sub -- correct as
    // long as a, b stay within 256 bits, true for any realistic range.
    u256 range_size;
    sub_u256_raw(range_size, b, a);
    u256 half_range;
    uint64_t carry_bit = 0;
    for (int i = NUM_LIMBS - 1; i >= 0; i--) {
        uint64_t new_carry = range_size.limb[i] & 1ULL;
        half_range.limb[i] = (range_size.limb[i] >> 1) | (carry_bit << 63);
        carry_bit = new_carry;
    }
    u256 tame_start;
    add_u256_raw(tame_start, a, half_range);

    ECPoint tame_point = scalar_mult(tame_start, g);
    u256 tame_dist = tame_start;

    ECPoint wild_point = pubkey;
    u256 wild_dist = u256_from_u64(0);

    uint64_t jumps = 0;
    while (jumps < max_jumps) {
        // tame step (herd = 0)
        {
            int idx = jump_index(tame_point, table_size);
            tame_point = point_add(tame_point, jump_points[idx]);
            u256 jump_amt = u256_from_u64(1ULL << exponents[idx]);
            // Plain accumulation, NOT mod P -- distances are scalars,
            // not field elements, and this is the walk's running total,
            // not yet ready for any modular reduction.
            u256 new_tame_dist;
            add_u256_raw(new_tame_dist, tame_dist, jump_amt);
            tame_dist = new_tame_dist;
            jumps++;

            u256 key;
            if (dp_key(tame_point, dp_bits, &key)) {
                uint64_t other_dist; uint8_t other_herd;
                if (dp_table_insert_or_check(dp_table, key, tame_dist.limb[0], 0,
                                              &other_dist, &other_herd)) {
                    if (other_herd == 1) {
                        u256 other_dist_u256 = u256_from_u64(other_dist);
                        u256 k = sub_mod_n(tame_dist, other_dist_u256);
                        ECPoint check = scalar_mult(k, g);
                        if (point_equal_pub(check, pubkey)) {
                            *out_key = k;
                            *out_jumps = jumps;
                            return true;
                        }
                    }
                }
            }
        }

        // wild step (herd = 1)
        {
            int idx = jump_index(wild_point, table_size);
            wild_point = point_add(wild_point, jump_points[idx]);
            u256 jump_amt = u256_from_u64(1ULL << exponents[idx]);
            u256 new_wild_dist;
            add_u256_raw(new_wild_dist, wild_dist, jump_amt);
            wild_dist = new_wild_dist;
            jumps++;

            u256 key;
            if (dp_key(wild_point, dp_bits, &key)) {
                uint64_t other_dist; uint8_t other_herd;
                if (dp_table_insert_or_check(dp_table, key, wild_dist.limb[0], 1,
                                              &other_dist, &other_herd)) {
                    if (other_herd == 0) {
                        u256 other_dist_u256 = u256_from_u64(other_dist);
                        u256 k = sub_mod_n(other_dist_u256, wild_dist);
                        ECPoint check = scalar_mult(k, g);
                        if (point_equal_pub(check, pubkey)) {
                            *out_key = k;
                            *out_jumps = jumps;
                            return true;
                        }
                    }
                }
            }
        }
    }
    *out_jumps = jumps;
    return false;
}
