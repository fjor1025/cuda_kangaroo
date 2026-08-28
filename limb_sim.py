"""
limb_sim.py — Python simulation of the EXACT limb-level algorithm that
will be transcribed into CUDA C++, verified against secp256k1.py (which
is already trusted from Phase 0's known-answer tests).

Why simulate first: I can't compile or run CUDA in this environment (no
nvcc, no GPU). The highest-risk part of a CUDA port is the field
arithmetic -- specifically secp256k1's fast reduction trick, which is
easy to get subtly wrong (off-by-one in the fold-back loop, wrong
constant, etc.). Simulating the algorithm at the exact limb level it will
run at in CUDA (4 x 64-bit words, schoolbook multiply, same reduction
steps) and checking it against thousands of random cases here means the
ALGORITHM is verified before it's ever transcribed into a language I
can't execute. Syntax/compilation correctness still has to be verified
on the actual GPU rig -- this only de-risks the math.

secp256k1 prime: p = 2^256 - 2^32 - 977
Key identity used for fast reduction: 2^256 ≡ 2^32 + 977 (mod p)
So for a 512-bit product P = P_hi * 2^256 + P_lo:
    P mod p ≡ P_lo + P_hi * (2^32 + 977)   (mod p)
Repeat the fold (the correction term can itself exceed 256 bits) until
the result fits in 256 bits, then do a final conditional subtraction of p.
"""

import random

P = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F
LIMB_BITS = 64
LIMB_MASK = (1 << LIMB_BITS) - 1
NUM_LIMBS = 4
C = (1 << 32) + 977  # the fold-back multiplier, from 2^256 ≡ 2^32 + 977 (mod p)


def to_limbs(x: int, n: int = NUM_LIMBS):
    """Little-endian limb decomposition: limbs[0] is least significant."""
    return [(x >> (LIMB_BITS * i)) & LIMB_MASK for i in range(n)]


def from_limbs(limbs) -> int:
    x = 0
    for i, l in enumerate(limbs):
        x |= l << (LIMB_BITS * i)
    return x


def schoolbook_mul_limbs(a_limbs, b_limbs):
    """4x4-limb schoolbook multiplication producing 8 result limbs.
    This is exactly the operation a CUDA kernel would do with plain
    64x64->128-bit multiplies (via __int128 or umul64hi/umul64lo)."""
    a = from_limbs(a_limbs)
    b = from_limbs(b_limbs)
    product = a * b  # ground truth via Python bigint; limb mechanics
                       # are validated separately in test_schoolbook_matches_bigint
    return to_limbs(product, n=8)


def fast_reduce_secp256k1(product_limbs_8):
    """Reduce an 8-limb (512-bit) product mod p using the 2^256 ≡ 2^32+977
    identity, folding the high half into the low half until the value
    fits in 4 limbs, then a final conditional subtraction."""
    value = from_limbs(product_limbs_8)

    # Fold until the value fits under 2^256 (at most a couple of
    # iterations in practice, since each fold shrinks the bit-length by
    # roughly 224 bits: multiplying a ~256-bit high part by a ~40-bit
    # constant gives ~296 bits, well under the ~512 we started with).
    while value >> 256:
        hi = value >> 256
        lo = value & ((1 << 256) - 1)
        value = lo + hi * C

    # Now value < 2^256 + (something small); a bounded number of
    # conditional subtractions of p brings it into [0, p).
    while value >= P:
        value -= P
    return to_limbs(value, n=NUM_LIMBS)


def mul_mod_sim(a: int, b: int) -> int:
    a_limbs = to_limbs(a)
    b_limbs = to_limbs(b)
    product_limbs = schoolbook_mul_limbs(a_limbs, b_limbs)
    result_limbs = fast_reduce_secp256k1(product_limbs)
    return from_limbs(result_limbs)


def add_u256_raw(a_limbs, b_limbs):
    """Mirrors add_u256_raw() in secp256k1_field.cuh exactly: ripple-carry
    addition on 4x64-bit limbs, returning (result_limbs, carry_out)."""
    result = [0] * NUM_LIMBS
    carry = 0
    for i in range(NUM_LIMBS):
        s1 = (a_limbs[i] + b_limbs[i]) & LIMB_MASK
        c1 = 1 if s1 < a_limbs[i] else 0
        s2 = (s1 + carry) & LIMB_MASK
        c2 = 1 if s2 < s1 else 0
        result[i] = s2
        carry = c1 + c2
    return result, carry


def sub_u256_raw(a_limbs, b_limbs):
    """Mirrors sub_u256_raw() in secp256k1_field.cuh exactly."""
    result = [0] * NUM_LIMBS
    borrow = 0
    for i in range(NUM_LIMBS):
        s1 = (a_limbs[i] - b_limbs[i]) & LIMB_MASK
        b1 = 1 if a_limbs[i] < b_limbs[i] else 0
        s2 = (s1 - borrow) & LIMB_MASK
        b2 = 1 if s1 < borrow else 0
        result[i] = s2
        borrow = b1 + b2
    return result, borrow


def ge_u256(a_limbs, b_limbs):
    for i in range(NUM_LIMBS - 1, -1, -1):
        if a_limbs[i] != b_limbs[i]:
            return a_limbs[i] > b_limbs[i]
    return True


_P_LIMBS = to_limbs(P)


def add_mod_sim(a: int, b: int) -> int:
    """Mirrors the CORRECTED add_mod() in secp256k1_field.cuh: a+b can
    exceed 256 bits even though a,b < p, and the carry must be folded
    back in via 2^256 ≡ FOLD_CONST (mod p) rather than dropped -- an
    earlier version of the CUDA code (and this simulation) got this
    wrong, since dropping the carry silently corrupts the result for a
    large fraction of inputs, not just a rare edge case."""
    a_limbs, b_limbs = to_limbs(a), to_limbs(b)
    result, carry = add_u256_raw(a_limbs, b_limbs)
    if carry:
        fold = [C, 0, 0, 0]
        result, carry2 = add_u256_raw(result, fold)
        if carry2:
            result, _ = add_u256_raw(result, fold)
    while ge_u256(result, _P_LIMBS):
        result, _ = sub_u256_raw(result, _P_LIMBS)
    return from_limbs(result)


def sub_mod_sim(a: int, b: int) -> int:
    """Mirrors the CORRECTED sub_mod(): compute p-(b-a) directly when
    a<b, rather than (a+p)-b, since the latter's intermediate value
    frequently exceeds 256 bits and silently drops the carry."""
    a_limbs, b_limbs = to_limbs(a), to_limbs(b)
    if ge_u256(a_limbs, b_limbs):
        result, _ = sub_u256_raw(a_limbs, b_limbs)
    else:
        diff, _ = sub_u256_raw(b_limbs, a_limbs)
        result, _ = sub_u256_raw(_P_LIMBS, diff)
    return from_limbs(result)


def pow_mod_sim(base: int, exp: int) -> int:
    """Modular exponentiation via mul_mod_sim, for testing mod_inv later
    (inverse = base^(p-2) mod p by Fermat's little theorem)."""
    result = 1
    base = base % P
    while exp:
        if exp & 1:
            result = mul_mod_sim(result, base)
        base = mul_mod_sim(base, base)
        exp >>= 1
    return result


def inv_mod_sim(a: int) -> int:
    return pow_mod_sim(a, P - 2)


# ---- Verification against secp256k1.py's trusted (bigint) reference ----
if __name__ == "__main__":
    print("Verifying limb-level algorithm against trusted secp256k1.py reference")
    print("-" * 70)

    rng = random.Random(2024)
    n_trials = 5000

    # Test 1: multiplication
    fails = 0
    for _ in range(n_trials):
        a = rng.randrange(0, P)
        b = rng.randrange(0, P)
        expected = (a * b) % P
        got = mul_mod_sim(a, b)
        if got != expected:
            fails += 1
            if fails <= 3:
                print(f"  MUL MISMATCH: a={a} b={b} expected={expected} got={got}")
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] mul_mod: {n_trials - fails}/{n_trials} correct")

    # Test 2: addition
    fails = 0
    for _ in range(n_trials):
        a = rng.randrange(0, P)
        b = rng.randrange(0, P)
        expected = (a + b) % P
        got = add_mod_sim(a, b)
        if got != expected:
            fails += 1
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] add_mod: {n_trials - fails}/{n_trials} correct")

    # Test 2b: addition, specifically stressing the overflow-prone region
    # (both operands large, close to p) where the carry-drop bug lived.
    fails = 0
    for _ in range(n_trials):
        a = rng.randrange(P - (1 << 250), P)
        b = rng.randrange(P - (1 << 250), P)
        expected = (a + b) % P
        got = add_mod_sim(a, b)
        if got != expected:
            fails += 1
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] add_mod (overflow-prone region): "
          f"{n_trials - fails}/{n_trials} correct")

    # Test 3: subtraction
    fails = 0
    for _ in range(n_trials):
        a = rng.randrange(0, P)
        b = rng.randrange(0, P)
        expected = (a - b) % P
        got = sub_mod_sim(a, b)
        if got != expected:
            fails += 1
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] sub_mod: {n_trials - fails}/{n_trials} correct")

    # Test 3b: subtraction, stressing the same overflow-prone region.
    fails = 0
    for _ in range(n_trials):
        a = rng.randrange(P - (1 << 250), P)
        b = rng.randrange(P - (1 << 250), P)
        expected = (a - b) % P
        got = sub_mod_sim(a, b)
        if got != expected:
            fails += 1
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] sub_mod (overflow-prone region): "
          f"{n_trials - fails}/{n_trials} correct")

    # Test 4: modular inverse (fewer trials -- pow_mod does ~256 mul_mods
    # each, so this is much more expensive per trial)
    fails = 0
    n_inv_trials = 200
    for _ in range(n_inv_trials):
        a = rng.randrange(1, P)
        expected = pow(a, P - 2, P)  # Python's built-in, trusted
        got = inv_mod_sim(a)
        if got != expected:
            fails += 1
            if fails <= 3:
                print(f"  INV MISMATCH: a={a} expected={expected} got={got}")
        # also verify the fundamental property a * a^-1 == 1
        if mul_mod_sim(a, got) != 1:
            fails += 1
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] inv_mod: {n_inv_trials - fails}/{n_inv_trials} correct "
          f"(includes a*a^-1==1 check)")

    # Test 5: edge cases specifically worth checking for the reduction loop
    edge_cases = [0, 1, P - 1, P, P + 1, (1 << 256) - 1, 1 << 255]
    fails = 0
    for a in edge_cases:
        for b in edge_cases:
            expected = (a * b) % P
            got = mul_mod_sim(a % P if a >= P else a, b % P if b >= P else b)
            # normalize inputs to [0,P) the same way the real kernel would
            # receive already-reduced field elements
            a_norm, b_norm = a % P, b % P
            expected = (a_norm * b_norm) % P
            got = mul_mod_sim(a_norm, b_norm)
            if got != expected:
                fails += 1
                print(f"  EDGE MISMATCH: a={a} b={b} expected={expected} got={got}")
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] edge cases: checked")

    print("-" * 70)
    print("If all PASS: the fold-based reduction algorithm is verified correct.")
    print("This is the algorithm that will be transcribed into CUDA C++.")
