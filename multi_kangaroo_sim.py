"""
multi_kangaroo_sim.py — Phase 4d, step 1: verify the MULTI-kangaroo offset
scheme in Python before adding CUDA's concurrency concerns on top.

The core new idea versus the single-pair walk (kangaroo_walk_sim.py): with
M tame and M wild kangaroos sharing one DP table, every kangaroo needs a
DISTINCT starting point. The walk's next step is a pure function of the
current point (jump_index only looks at point.x), so if two kangaroos of
the same herd start at the identical point, they take the IDENTICAL path
forever -- completely wasted parallelism, not just inefficiency.

Scheme used here (standard multi-kangaroo practice): kangaroo i in a herd
starts at the herd's usual base position plus a distinct offset
`i * mean_jump`, with its distance counter initialized to include that
offset from the start. This keeps the collision math IDENTICAL to the
single-pair case -- a tame kangaroo's point is always (its tracked
distance)*G, and a wild kangaroo's point is always pubkey + (its tracked
distance)*G, exactly as before, just with nonzero starting distances now
for every kangaroo except the very first of each herd.

This is deliberately still single-threaded/sequential in Python (round-
robin stepping through all 2M kangaroos) -- concurrency correctness is a
CUDA-specific concern (atomic DP table access) tackled separately, after
this offset scheme itself is confirmed to solve correctly.
"""

import math
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secp256k1 import scalar_mult, point_add, N
from kangaroo_cpu import JumpTable, dp_key
from kangaroo_walk_sim import OpenAddressingDPTable


def solve_multi(pubkey, a, b, dp_bits, num_tame, num_wild, max_rounds, dp_table_capacity):
    """Round-robin steps num_tame tame kangaroos and num_wild wild
    kangaroos, each with a distinct starting offset, sharing one DP
    table. Returns (recovered_key_or_None, total_jumps)."""
    range_size = b - a
    table = JumpTable.build(range_size)
    dp_table = OpenAddressingDPTable(dp_table_capacity)

    tame_base = a + range_size // 2
    mean_jump = int(table.mean_jump)

    # Each kangaroo: [point, dist]
    tames = []
    for i in range(num_tame):
        start = tame_base + i * mean_jump
        tames.append([scalar_mult(start), start])

    wilds = []
    for j in range(num_wild):
        offset = j * mean_jump
        wilds.append([point_add(pubkey, scalar_mult(offset)) if offset else pubkey, offset])

    jumps = 0
    for round_i in range(max_rounds):
        for i in range(num_tame):
            point, dist = tames[i]
            idx = table.jump_index(point)
            point = point_add(point, table.points[idx])
            dist += 1 << table.exponents[idx]
            tames[i] = [point, dist]
            jumps += 1

            key = dp_key(point, dp_bits)
            if key is not None:
                collision, other_dist, other_herd = dp_table.insert_or_check(key, dist, 0)
                if collision and other_herd == 1:
                    k = (dist - other_dist) % N
                    if scalar_mult(k) == pubkey:
                        return k, jumps

        for j in range(num_wild):
            point, dist = wilds[j]
            idx = table.jump_index(point)
            point = point_add(point, table.points[idx])
            dist += 1 << table.exponents[idx]
            wilds[j] = [point, dist]
            jumps += 1

            key = dp_key(point, dp_bits)
            if key is not None:
                collision, other_dist, other_herd = dp_table.insert_or_check(key, dist, 1)
                if collision and other_herd == 0:
                    k = (other_dist - dist) % N
                    if scalar_mult(k) == pubkey:
                        return k, jumps

    return None, jumps


if __name__ == "__main__":
    import kangaroo_cpu as baseline

    print("Verifying the multi-kangaroo offset scheme against the trusted")
    print("single-pair baseline")
    print("-" * 70)

    test_cases = [(16, 4), (18, 4), (20, 5), (22, 5)]
    rng = random.Random(42)
    all_passed = True

    for bits, dp_bits in test_cases:
        a, b = 1 << (bits - 1), 1 << bits
        k_true = rng.randrange(a, b)
        Q = scalar_mult(k_true)

        r_base = baseline.solve(Q, a, b, dp_bits=dp_bits, max_jumps=2_000_000, seed=99)

        # 8 tame + 8 wild kangaroos, round-robin. max_rounds chosen
        # generously; total jumps = rounds * (num_tame+num_wild).
        k_multi, jumps_multi = solve_multi(Q, a, b, dp_bits=dp_bits,
                                            num_tame=8, num_wild=8,
                                            max_rounds=200_000,
                                            dp_table_capacity=1 << 16)

        ok = (k_multi == k_true) and (r_base.private_key == k_true)
        all_passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] range=2^{bits} true_k={k_true} "
              f"baseline_jumps={r_base.total_jumps} "
              f"multi_recovered={k_multi} multi_jumps={jumps_multi} "
              f"(speedup_vs_baseline={r_base.total_jumps/max(jumps_multi,1):.2f}x fewer *rounds* "
              f"-- see note below)")

    print("-" * 70)
    print("NOTE: 'multi_jumps' counts total individual kangaroo steps across")
    print("all 16 kangaroos, same unit as baseline's single-pair total_jumps.")
    print("In a REAL parallel GPU run these all happen concurrently, so the")
    print("relevant comparison isn't total-steps but WALL-CLOCK time -- with")
    print("16 kangaroos instead of 2, wall-clock should improve roughly")
    print("proportionally to kangaroo count (birthday-paradox scaling),")
    print("not appear here as a 'jumps' reduction in this sequential simulation.")
    if all_passed:
        print("All cases passed: the offset scheme produces correct results.")
    else:
        print("FAILURES ABOVE -- do not proceed to CUDA until these pass.")
