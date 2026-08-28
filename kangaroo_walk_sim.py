"""
kangaroo_walk_sim.py — Python simulation of the EXACT algorithm that will
be transcribed into the CUDA kangaroo-walk kernel (Phase 4c), verified
against the already-trusted kangaroo_cpu.py baseline before touching CUDA.

Why simulate first: same reasoning as limb_sim.py for Phase 4a. The CUDA
version won't use a Python dict for the DP store (no such thing exists on
GPU) -- it'll use a fixed-capacity open-addressing hash table, which is a
genuinely different data structure with its own failure modes (table full,
probe sequence bugs, etc.) that's worth verifying separately from the
"does the walk itself recover the right key" question kangaroo_cpu.py's
existing self-test already answers.

This simulation mirrors the planned CUDA kernel's structure:
  - JumpTable built exactly like kangaroo_cpu.JumpTable.build()
  - jump_index via the same multiplicative hash
  - DP detection via the same trailing-zero-bits check, but using an
    explicit fixed-capacity open-addressing table (linear probing)
    instead of a dict, since that's what has to run on a GPU thread
  - Single tame + single wild kangaroo, stepped sequentially in one
    "thread" (matching the planned first CUDA test: prove the stepping
    logic is correct before scaling to many parallel kangaroos, which is
    a separate later phase with its own synchronization concerns)
"""

import math
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secp256k1 import scalar_mult, point_add, N
from kangaroo_cpu import JumpTable, dp_key


class OpenAddressingDPTable:
    """Fixed-capacity hash table with linear probing, mirroring what the
    CUDA kernel will use (thread-local/global array, no dynamic
    allocation). Stores (key -> (dist, herd)); "herd" is 0 for tame, 1
    for wild, matching the planned CUDA representation."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.occupied = [False] * capacity
        self.keys = [0] * capacity
        self.dists = [0] * capacity
        self.herds = [0] * capacity

    def insert_or_check(self, key, dist, herd):
        """Returns (collision_found, other_dist, other_herd) if a DIFFERENT
        herd already holds this key; otherwise inserts (if not already
        present) and returns (False, None, None)."""
        idx = key % self.capacity
        for probe in range(self.capacity):
            slot = (idx + probe) % self.capacity
            if not self.occupied[slot]:
                self.occupied[slot] = True
                self.keys[slot] = key
                self.dists[slot] = dist
                self.herds[slot] = herd
                return False, None, None
            if self.keys[slot] == key:
                if self.herds[slot] != herd:
                    return True, self.dists[slot], self.herds[slot]
                return False, None, None  # same herd already has this key
        raise RuntimeError("DP table full -- capacity too small for this test")


def solve_sim(pubkey, a, b, dp_bits, max_jumps, dp_table_capacity):
    range_size = b - a
    table = JumpTable.build(range_size)
    dp_table = OpenAddressingDPTable(dp_table_capacity)

    tame_start = a + range_size // 2
    tame_point = scalar_mult(tame_start)
    tame_dist = tame_start

    wild_point = pubkey
    wild_dist = 0

    jumps = 0
    while jumps < max_jumps:
        # tame step (herd = 0)
        idx = table.jump_index(tame_point)
        tame_point = point_add(tame_point, table.points[idx])
        tame_dist += 1 << table.exponents[idx]
        jumps += 1

        key = dp_key(tame_point, dp_bits)
        if key is not None:
            collision, other_dist, other_herd = dp_table.insert_or_check(key, tame_dist, 0)
            if collision and other_herd == 1:
                k = (tame_dist - other_dist) % N
                if scalar_mult(k) == pubkey:
                    return k, jumps

        # wild step (herd = 1)
        idx = table.jump_index(wild_point)
        wild_point = point_add(wild_point, table.points[idx])
        wild_dist += 1 << table.exponents[idx]
        jumps += 1

        key = dp_key(wild_point, dp_bits)
        if key is not None:
            collision, other_dist, other_herd = dp_table.insert_or_check(key, wild_dist, 1)
            if collision and other_herd == 0:
                k = (other_dist - wild_dist) % N
                if scalar_mult(k) == pubkey:
                    return k, jumps

    return None, jumps


if __name__ == "__main__":
    import kangaroo_cpu as baseline

    print("Verifying the CUDA-bound walk algorithm (with explicit open-")
    print("addressing DP table) against the trusted kangaroo_cpu.py baseline")
    print("-" * 70)

    test_cases = [(16, 4), (18, 4), (20, 5), (22, 5)]
    rng = random.Random(42)
    all_passed = True

    for bits, dp_bits in test_cases:
        a, b = 1 << (bits - 1), 1 << bits
        k_true = rng.randrange(a, b)
        Q = scalar_mult(k_true)

        # Baseline (dict-based, already trusted)
        r_base = baseline.solve(Q, a, b, dp_bits=dp_bits, max_jumps=2_000_000, seed=99)

        # Simulated CUDA-bound algorithm (open-addressing table)
        k_sim, jumps_sim = solve_sim(Q, a, b, dp_bits=dp_bits, max_jumps=2_000_000,
                                      dp_table_capacity=65536)

        ok = (k_sim == k_true) and (r_base.private_key == k_true)
        all_passed &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] range=2^{bits} true_k={k_true} "
              f"baseline_recovered={r_base.private_key} (jumps={r_base.total_jumps}) "
              f"sim_recovered={k_sim} (jumps={jumps_sim})")

    print("-" * 70)
    if all_passed:
        print("All cases passed: the open-addressing DP table produces the same")
        print("result as the dict-based baseline. Safe to transcribe into CUDA.")
    else:
        print("FAILURES ABOVE -- do not transcribe to CUDA until these pass.")
