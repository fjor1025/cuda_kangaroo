# Phase 4a: secp256k1 Field Arithmetic (CUDA) — Build & Test Instructions

## Why this step is different from everything before it

Phases 0-3 were built and verified entirely in a sandbox where I could run
every test myself. **This sandbox has no CUDA toolkit and no GPU** — I
could not compile or run any of this code. Your rig is now genuinely
load-bearing for verification, not just for eventual performance.

To reduce risk given that constraint:
1. The underlying algorithm (4×64-bit limb representation, schoolbook
   multiplication, secp256k1's fold-based fast reduction) was verified
   in Python (`limb_sim.py`) against thousands of random cases, checked
   against the already-trusted `secp256k1.py` from Phase 0.
2. I then manually traced through the CUDA transcription line-by-line
   looking for carry-propagation bugs, and found **two real ones** this
   way (see below) — fixed and re-verified in Python before touching the
   CUDA file again.
3. What's *not* yet verified is the CUDA-specific syntax and whether it
   actually compiles for sm_86. That's what running this on your rig
   will tell us.

## Two bugs caught before you ever see a compiler error

Worth knowing about, since they're the kind of bug that would have
silently produced wrong answers rather than crashing:

- **`sub_mod`**: when `a < b`, computing `(a + p) - b` as an intermediate
  step overflows 256 bits for almost all values of `a` (since `p` is so
  close to 2^256) — and the overflow was being silently dropped. Fixed
  by reordering to `p - (b - a)`, which never produces an out-of-range
  intermediate value.
- **`add_mod`**: `a + b` can also exceed 256 bits whenever both operands
  are reasonably large, and that carry bit was likewise being dropped
  instead of folded back in via the `2^256 ≡ 2^32+977 (mod p)` identity.

Both are now fixed and covered by dedicated stress tests in `limb_sim.py`
that specifically target the overflow-prone region (both operands close
to `p`), not just uniformly random inputs that might rarely hit it.

## A note if your editor struggles opening this folder

The generated header files (`test_vectors.h`, `point_test_vectors.h`) are
large (900KB / 119KB) files consisting of thousands of array-literal
lines -- not meant for a human or a code editor's language server to
parse. If VSCode (or any editor) tries to index, syntax-highlight, or
diff-preview these, it can hang or crash. **This is an editor/tooling
issue, unrelated to whether the code compiles or is correct.**

To avoid it: these two files are **not included** in this package and
are listed in `.gitignore`. Regenerate them locally after cloning (this
also keeps the git repo clean, since they're fully deterministic --
fixed random seeds, byte-identical output every time):

```bash
python3 generate_test_vectors.py
python3 generate_point_test_vectors.py
```

Do this once, before compiling. `secp256k1.py` is included in this
package specifically so these scripts are self-contained and don't need
anything outside this folder.

## Files in this directory- `limb_sim.py` — Python verification of the algorithm (run this first,
  though you've likely already seen it pass in the earlier conversation).
- `secp256k1_field.cuh` — the actual CUDA device code: `add_mod`,
  `sub_mod`, `mul_mod`, `inv_mod` on a `u256` type (4×64-bit limbs).
- `generate_test_vectors.py` — generates `test_vectors.h` from the
  trusted Python reference (already run; `test_vectors.h` is included).
- `test_vectors.h` — 2000 mul, 500 add, 500 sub, 200 inv known-answer
  test vectors.
- `test_field_arithmetic.cu` — the test harness. Runs each operation on
  the GPU against the test vectors and reports PASS/FAIL, same style as
  the Python self-tests throughout this project.

## Build

```bash
cd cuda/
nvcc -O3 -arch=sm_86 -o test_field_arithmetic test_field_arithmetic.cu
```

`-arch=sm_86` targets your RTX 3080s specifically (Ampere).

## Run

```bash
./test_field_arithmetic
```

## What to send back

**If it doesn't compile**: the exact `nvcc` error output. Compiler errors
here are expected to be plausible — this is untested CUDA syntax — and
should be straightforward to fix once I can see what nvcc actually
objects to.

**If it compiles but any test group shows FAIL**: the full output,
including the "first failure at index N" — that tells us which specific
input triggered it, which I can then reproduce and debug in the Python
simulation (fast, no GPU needed) before touching the CUDA again.

**If all four groups PASS**: field arithmetic is verified correct on your
actual hardware, and we move to Phase 4b — point arithmetic (`add`,
`double`) built on top of these verified field operations.

**Status: CONFIRMED.** This step already ran clean on the rig:
```
[PASS] mul_mod: 2000/2000 correct
[PASS] add_mod: 500/500 correct
[PASS] sub_mod: 500/500 correct
[PASS] inv_mod: 200/200 correct
```
All field arithmetic test groups passed on first compile, no errors.

## Phase 4b: Point Arithmetic

Builds directly on Phase 4a's now hardware-verified field arithmetic.
Adds `secp256k1_point.cuh` (point_add, scalar_mult, the generator point)
and a test harness the same way.

**New files:**
- `secp256k1_point.cuh` — `ECPoint` struct, `point_add` (mirrors
  secp256k1.py's branch structure exactly: infinity identity, doubling,
  P+(-P)==infinity, and the general chord formula), `scalar_mult`
  (double-and-add), and the generator point G (constants verified by
  independent hand-decomposition, cross-checked programmatically against
  secp256k1.py -- see conversation history).
- `generate_point_test_vectors.py` — generates `point_test_vectors.h`
  from the trusted Python reference: 156 point_add cases (6 targeted
  edge cases -- doubling, P+(-P), both infinity-identity directions,
  infinity+infinity, G+G -- plus 150 random) and 85 scalar_mult cases
  (k=0, k=1, k=2, k=N-1, the actual solved puzzle #135 private key as an
  independent known-answer check, plus 80 random).
- `point_test_vectors.h` — the generated vectors (included, already run).
- `test_point_arithmetic.cu` — the test harness.

**Build and run:**
```bash
nvcc -O3 -arch=sm_86 -o test_point_arithmetic test_point_arithmetic.cu
./test_point_arithmetic
```

**Expected output on success:**
```
secp256k1 point arithmetic self-test (Phase 4b)
Built on field arithmetic already verified on this hardware (Phase 4a)
----------------------------------------------------------------------
[PASS] point_add: 156/156 correct
[PASS] scalar_mult: 85/85 correct
----------------------------------------------------------------------
All point arithmetic test groups PASSED.
```

Same instructions as Phase 4a apply if anything fails: send back the
full output (especially the "first failure at index N") and I'll
reproduce it in the Python reference to debug.

## Phase 4c: The Kangaroo Walk Itself

Built on the now hardware-verified field (4a) and point (4b) arithmetic.
This is the actual jump table, distinguished-point detection, and
tame/wild stepping logic -- genuinely new, unverified-on-hardware code.

**Scope**: deliberately a single-thread walk (one thread runs the full
tame+wild loop sequentially), matching `kangaroo_cpu.py`'s `solve()`
structure exactly. This proves the stepping/DP-matching logic in
isolation before the separate, larger problem of running many kangaroos
in parallel with a shared, contended DP store across threads (future
work, not this phase).

**New files:**
- `kangaroo_walk.cuh` — the walk kernel: jump table construction (from
  Python-precomputed exponents, avoiding re-deriving that math in C++),
  distinguished-point key (matches `kangaroo_cpu.dp_key` exactly, full
  x-coordinate not truncated), an open-addressing DP hash table (verified
  first in `kangaroo_walk_sim.py` against the dict-based CPU baseline),
  and the walk loop itself.
- `kangaroo_walk_sim.py` — Python simulation of the exact algorithm
  (including the open-addressing table mechanics), verified against
  `kangaroo_cpu.py`'s dict-based baseline before any CUDA was written.
  All 4 test cases matched the baseline's jump counts *exactly*.
- `generate_walk_test_vectors.py` — generates `walk_test_vectors.h`:
  jump-table exponents, puzzle params, and expected recovered keys, all
  computed via the trusted Python reference.
- `walk_test_vectors.h` — the generated test cases (included, already run).
- `test_kangaroo_walk.cu` — the test harness.

**A bug worth knowing about**: an early version used `add_mod`/`sub_mod`
(reduction mod **P**, the field prime) to track kangaroo distances.
Distances are scalars and must be reduced mod **N** (the curve order,
a different value from P) — using the wrong modulus happened to not
produce a wrong answer at this toy test scale (distances never got
anywhere near either modulus), but was conceptually incorrect and would
have broken at real puzzle scale. Fixed: distances now accumulate as
plain (non-modular) values during the walk, exactly like the unbounded
Python integers in `kangaroo_cpu.py`, with proper mod-N reduction only
at the final key computation.

**Build and run:**
```bash
python3 generate_walk_test_vectors.py   # regenerate walk_test_vectors.h
nvcc -O3 -arch=sm_86 -o test_kangaroo_walk test_kangaroo_walk.cu
./test_kangaroo_walk
```

**Expected output on success:**
```
Kangaroo walk self-test (Phase 4c)
----------------------------------------------------------------------
[PASS] test 0: found=1 key_matches=1
[PASS] test 1: found=1 key_matches=1
[PASS] test 2: found=1 key_matches=1
[PASS] test 3: found=1 key_matches=1
----------------------------------------------------------------------
All 4 kangaroo walk test cases PASSED -- recovered the exact
correct private key in every case, matching the Python reference.
```

**Status: CONFIRMED on the rig, with a real bug found and fixed along the
way.** The first run compiled with a warning (`a __device__ variable
"WALK_TESTS" cannot be directly read in a host function`) and failed
every test's key comparison (`found=1 key_matches=0`). Root cause: the
test harness's `main()` was directly indexing the `__device__` (GPU-only)
`WALK_TESTS` array from host code -- illegal, and silently read garbage
instead of the real expected keys. The kernel itself was almost
certainly finding the correct key the whole time (it verifies
`scalar_mult(k)==pubkey` internally before ever returning `found=true`).
Fixed with `cudaMemcpyFromSymbol` to properly copy the test data to host
memory first. After the fix: clean compile, **zero warnings**, all 4
tests pass.

## Scaling Up: Real Hardware Timing Data

Extended the test set with larger bit-widths (26, 28, 30, 32 -- capped
there because generating the Python reference itself became impractically
slow beyond 2^32, taking several minutes just to verify one case) and
added timing instrumentation (`cudaEvent` based) to report jumps/sec for
each test case.

**Important scope note**: this single-thread scaling exercise gets real
hardware throughput numbers and validates correctness at larger ranges,
but it does **not** meaningfully approach puzzle #90-135 scale by itself.
The gap is enormous:

| Range | Expected steps (~sqrt(range)) |
|---|---|
| 2^32 (this test's max) | ~65,536 |
| Puzzle #90 | ~35 trillion |
| Puzzle #135 | ~1.5×10^20 |

Even at an optimistic 10 million steps/sec on a single thread, puzzle #90
alone would take over a month. This is the concrete, measured reason
Phase 4d (parallelizing across thousands of CUDA threads per GPU) and
multi-GPU scaling are necessary next steps, not optional polish -- the
real jumps/sec number from this test run is what lets us calculate
exactly how much parallelism is actually needed.

**Rebuild and rerun** (regenerate the vectors first -- the new larger
test cases aren't in your current `walk_test_vectors.h`):
```bash
python3 generate_walk_test_vectors.py   # now takes a few minutes (bits=32 verification is slow in pure Python)
nvcc -O3 -arch=sm_86 -o test_kangaroo_walk test_kangaroo_walk.cu
./test_kangaroo_walk
```

Expected output now includes timing per test:
```
[PASS] test 0: found=1 key_matches=1 jumps=177 time=0.12ms (1475000 jumps/sec)
...
[PASS] test 7: found=1 key_matches=1 jumps=1488796 time=XXXms (XXX jumps/sec)
```
The jumps/sec figures are the real number to send back -- that's what
turns "we need Phase 4d" from a general statement into a concrete plan
(how many parallel threads, roughly how long a real puzzle attempt would
take at various thread counts, etc.).
