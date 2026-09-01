// kangaroo_walk_parallel.cuh — Phase 4d: many kangaroos per GPU, sharing
// one DP table accessed concurrently by thousands of threads.
//
// Two genuinely new problems versus the single-thread walk (kangaroo_walk.cuh):
//
// 1. DISTINCT STARTING POINTS. The walk's next step is a pure function of
//    the current point (jump_index only looks at x), so two kangaroos of
//    the same herd starting at the identical point take the IDENTICAL
//    path forever. Verified in Python first (multi_kangaroo_sim.py): each
//    kangaroo i in a herd starts at the herd's usual base position plus a
//    distinct offset i*mean_jump, with its distance counter initialized
//    to include that offset -- this keeps the collision math IDENTICAL to
//    the single-pair case (a tame point is always dist*G, a wild point is
//    always pubkey + dist*G), just with nonzero starting distances.
//
// 2. CONCURRENT DP TABLE ACCESS. Thousands of threads can now try to
//    read/write the same hash table simultaneously. A naive translation
//    of the single-thread open-addressing table (kangaroo_walk.cuh's
//    dp_table_insert_or_check) would have a race: one thread could see
//    another thread's "occupied" flag before that thread has finished
//    writing key/dist/herd, and read torn/garbage data.
//
//    Fix: use a 3-state "occupied" field instead of a boolean, updated
//    with atomicCAS:
//      0 = empty
//      2 = CLAIMED, payload being written (transient)
//      1 = READY, payload safe to read
//    A thread claims a slot via atomicCAS(0 -> 2), writes its payload,
//    then publishes via atomicExch(2 -> 1) *after* a __threadfence() to
//    guarantee the payload writes are visible to other threads before the
//    state flip is. Any thread that sees state==2 treats it as "not
//    available yet" and moves to the next probe slot -- this can rarely
//    cause a thread to walk past a slot that would have matched moments
//    later, but that only costs a little extra search time, never
//    correctness (every candidate key is independently verified via
//    scalar_mult(k)==pubkey before being accepted, exactly as in every
//    earlier phase).

#pragma once
#include "kangaroo_walk.cuh"

#define DP_STATE_EMPTY   0u
#define DP_STATE_CLAIMED 2u
#define DP_STATE_READY   1u

struct ParallelDPEntry {
    u256 key;
    uint64_t dist;      // same 64-bit truncation caveat as the single-thread version
    unsigned int herd;   // 0 = tame, 1 = wild
    unsigned int state;  // DP_STATE_* -- must be a CUDA-atomics-native width (32-bit)
};

__device__ __forceinline__ uint64_t dp_hash_index_parallel(const u256& key, uint64_t capacity) {
    // capacity must be a power of two (see kangaroo_walk.cuh's dp_hash_index
    // for why this makes "key % capacity" reduce to a cheap mask on limb[0]).
    return key.limb[0] & (capacity - 1);
}

// Returns true if a DIFFERENT herd's entry was found at this key (a
// genuine collision candidate). On collision, writes the other entry's
// distance and herd to the output parameters. This function may
// OCCASIONALLY miss a collision that a sequential version would have
// caught (if it races past a CLAIMED-but-not-yet-READY slot) -- by
// design, as explained above: safe, not silently wrong, just very
// rarely slightly less efficient.
__device__ __forceinline__ bool dp_table_insert_or_check_parallel(
        ParallelDPEntry* table, uint64_t capacity, const u256& key, uint64_t dist,
        unsigned int herd, uint64_t* out_other_dist, unsigned int* out_other_herd) {
    uint64_t idx = dp_hash_index_parallel(key, capacity);
    for (uint64_t probe = 0; probe < capacity; probe++) {
        uint64_t slot = (idx + probe) & (capacity - 1);

        unsigned int prior_state = atomicCAS(&table[slot].state, DP_STATE_EMPTY, DP_STATE_CLAIMED);

        if (prior_state == DP_STATE_EMPTY) {
            // We won the claim on this slot -- write the payload, THEN
            // publish. The fence guarantees every thread that
            // subsequently observes state==READY also sees these writes
            // (not a stale/partial view of them).
            table[slot].key = key;
            table[slot].dist = dist;
            table[slot].herd = herd;
            __threadfence();
            atomicExch(&table[slot].state, DP_STATE_READY);
            return false;  // fresh insert, not a collision (yet)
        }

        if (prior_state == DP_STATE_CLAIMED) {
            // Someone else is mid-write on this exact slot right now.
            // Don't wait/spin (could deadlock-adjacent stall the whole
            // warp under heavy contention) -- treat as "not a match, not
            // available", and try the next probe slot instead.
            continue;
        }

        // prior_state == DP_STATE_READY: safe to read this slot's payload.
        if (u256_equal(table[slot].key, key)) {
            if (table[slot].herd != herd) {
                *out_other_dist = table[slot].dist;
                *out_other_herd = table[slot].herd;
                return true;
            }
            return false;  // same herd already holds this key, nothing new
        }
        // Different key occupying this slot (a hash collision, not a
        // kangaroo collision) -- keep probing.
    }
    return false;  // table full; shouldn't happen with reasonable sizing
}

// ---- One-time jump table setup (NOT per-thread!) ----
//
// An earlier version of this file had EVERY thread independently
// recompute the 32-entry jump table via scalar_mult inside the main
// kernel -- with thousands of threads, that's thousands of redundant
// computations of the exact same 32 points. Fixed: a separate, tiny
// kernel computes the table once into global memory that every thread
// in the main kernel just reads. (Can't use __constant__ memory for
// this despite it being a natural fit for "every thread reads the same
// small broadcast data" -- __constant__ memory can only be written from
// host code via cudaMemcpyToSymbol, not from a device kernel, and the
// jump points have to be computed via device-side scalar_mult since
// there's no host-side EC arithmetic in this codebase. Plain global
// memory reads that are uniform across threads still get broadcast
// efficiently through the GPU's cache hierarchy on Ampere.)

__global__ void build_jump_table_kernel(const int* exponents, int table_size,
                                          ECPoint* jump_points_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_size) return;
    ECPoint g = secp256k1_generator();
    u256 exp_scalar = u256_from_u64(1ULL << exponents[i]);
    jump_points_out[i] = scalar_mult(exp_scalar, g);
}

// ---- The parallel walk kernel ----
//
// Thread layout: threads [0, num_tame) are tame kangaroos, threads
// [num_tame, num_tame+num_wild) are wild. Each computes its own distinct
// starting offset directly from its index -- no host-precomputed offset
// table needed.
//
// Early-exit coordination: `*global_found` is a single shared flag
// (initialized to 0 by the host before launch). Any thread that verifies
// a real solution atomically claims it (CAS 0->1) so exactly one thread
// writes the result, and every thread checks this flag periodically to
// stop promptly once any thread (not necessarily itself) has solved it --
// otherwise thousands of threads would keep burning cycles after the
// answer is already known.

__global__ void parallel_kangaroo_kernel(
        ECPoint pubkey, u256 a, u256 b, int dp_bits,
        const int* exponents, const ECPoint* jump_points, int table_size,
        uint64_t mean_jump, uint64_t max_jumps_per_thread,
        int num_tame, int num_wild,
        ParallelDPEntry* dp_table, uint64_t dp_capacity,
        int* global_found, uint64_t* out_key_limbs, uint64_t* out_total_jumps) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_kangaroos = num_tame + num_wild;
    if (tid >= total_kangaroos) return;

    bool is_tame = (tid < num_tame);
    int herd_local_index = is_tame ? tid : (tid - num_tame);

    ECPoint g = secp256k1_generator();
    // jump_points and mean_jump are now precomputed ONCE (see
    // build_jump_table_kernel above / host-side sum) and passed in -- no
    // per-thread scalar_mult calls or redundant summation needed, unlike
    // an earlier version of this file.

    u256 range_size;
    sub_u256_raw(range_size, b, a);
    u256 half_range;
    {
        uint64_t carry_bit = 0;
        for (int i = NUM_LIMBS - 1; i >= 0; i--) {
            uint64_t new_carry = range_size.limb[i] & 1ULL;
            half_range.limb[i] = (range_size.limb[i] >> 1) | (carry_bit << 63);
            carry_bit = new_carry;
        }
    }
    u256 tame_base;
    add_u256_raw(tame_base, a, half_range);

    u256 dist;
    ECPoint point;
    if (is_tame) {
        uint64_t offset = (uint64_t)herd_local_index * mean_jump;
        add_u256_raw(dist, tame_base, u256_from_u64(offset));
        point = scalar_mult(dist, g);
    } else {
        uint64_t offset = (uint64_t)herd_local_index * mean_jump;
        dist = u256_from_u64(offset);
        if (offset == 0) {
            point = pubkey;
        } else {
            ECPoint offset_point = scalar_mult(dist, g);
            point = point_add(pubkey, offset_point);
        }
    }

    unsigned int my_herd = is_tame ? 0u : 1u;
    uint64_t jumps = 0;

    while (jumps < max_jumps_per_thread) {
        // Cooperative early exit: check every iteration. This is a plain
        // (non-atomic) read of a flag another thread may be writing
        // concurrently -- intentionally so; we only need to notice
        // "someone solved it" within a bounded number of extra
        // iterations, not instantly, so the relaxed read is fine and
        // much cheaper than an atomic load every step.
        if (*global_found) break;

        int idx = jump_index(point, table_size);
        ECPoint candidate = point_add(point, jump_points[idx]);
        u256 jump_amt = u256_from_u64(1ULL << exponents[idx]);
        u256 new_dist;
        add_u256_raw(new_dist, dist, jump_amt);
        point = candidate;
        dist = new_dist;
        jumps++;

        u256 key;
        if (dp_key(point, dp_bits, &key)) {
            uint64_t other_dist; unsigned int other_herd;
            if (dp_table_insert_or_check_parallel(dp_table, dp_capacity, key, dist.limb[0],
                                                    my_herd, &other_dist, &other_herd)) {
                if (other_herd != my_herd) {
                    u256 other_dist_u256 = u256_from_u64(other_dist);
                    u256 k = is_tame ? sub_mod_n(dist, other_dist_u256)
                                     : sub_mod_n(other_dist_u256, dist);
                    ECPoint check = scalar_mult(k, g);
                    if (point_equal_pub(check, pubkey)) {
                        if (atomicCAS(global_found, 0, 1) == 0) {
                            for (int i = 0; i < 4; i++) out_key_limbs[i] = k.limb[i];
                        }
                        break;
                    }
                }
            }
        }
    }

    atomicAdd((unsigned long long*)out_total_jumps, (unsigned long long)jumps);
}
