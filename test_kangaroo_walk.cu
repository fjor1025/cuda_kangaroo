// test_kangaroo_walk.cu — Phase 4c self-test: runs the single-thread
// kangaroo walk against test cases generated from the trusted Python
// reference (walk_test_vectors.h, from generate_walk_test_vectors.py).
//
// Build:
//   nvcc -O3 -arch=sm_86 -o test_kangaroo_walk test_kangaroo_walk.cu
// Run:
//   ./test_kangaroo_walk

#include <cstdio>
#include "kangaroo_walk.cuh"
#include "walk_test_vectors.h"

__global__ void run_walk_test(int test_idx, int* result_found, uint64_t* result_key_limbs,
                                uint64_t* result_jumps, DPEntry* dp_table) {
    // Single-thread kernel: only thread 0 does anything. One kangaroo
    // walk per kernel launch, matching the Phase 4c scope (prove the
    // stepping/DP logic is correct before scaling to many parallel
    // kangaroos, which is separate future work).
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    WalkTestCase tc = WALK_TESTS[test_idx];

    u256 a, b, pubkey_x, pubkey_y;
    for (int i = 0; i < 4; i++) {
        a.limb[i] = tc.a[i];
        b.limb[i] = tc.b[i];
        pubkey_x.limb[i] = tc.pubkey_x[i];
        pubkey_y.limb[i] = tc.pubkey_y[i];
    }
    ECPoint pubkey; pubkey.x = pubkey_x; pubkey.y = pubkey_y; pubkey.infinity = false;

    // DP table clearing moved OUT of this kernel and into a host-side
    // cudaMemset before launch (see main()) -- a single GPU thread
    // looping over 65,536 entries one at a time is a pathologically slow
    // pattern (no other threads to hide memory latency behind), and was
    // found to be adding a roughly CONSTANT ~9.7s of overhead to every
    // single test case regardless of actual work, completely swamping
    // the real walk-logic timing. cudaMemset uses the GPU's dedicated
    // bulk memory-clear path instead, which is orders of magnitude
    // faster for this.

    u256 out_key;
    uint64_t out_jumps;
    bool found = kangaroo_solve(pubkey, a, b, tc.dp_bits, tc.exponents, JUMP_TABLE_SIZE,
                                 tc.max_jumps, dp_table, &out_key, &out_jumps);

    *result_found = found ? 1 : 0;
    *result_jumps = out_jumps;
    if (found) {
        for (int i = 0; i < 4; i++) result_key_limbs[i] = out_key.limb[i];
    }
}

int check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR during %s: %s\n", what, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main() {
    printf("Kangaroo walk self-test (Phase 4c)\n");
    printf("Built on field (4a) and point (4b) arithmetic, both already\n");
    printf("verified on this hardware. This tests NEW code: the walk loop,\n");
    printf("DP hash table, and collision detection.\n");
    printf("----------------------------------------------------------------------\n");

    DPEntry* d_dp_table;
    cudaMalloc(&d_dp_table, DP_TABLE_CAPACITY * sizeof(DPEntry));

    int* d_found;
    uint64_t* d_key_limbs;
    uint64_t* d_jumps;
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_key_limbs, 4 * sizeof(uint64_t));
    cudaMalloc(&d_jumps, sizeof(uint64_t));

    // WALK_TESTS is a __device__ (GPU-only) array -- it cannot be read
    // directly from host code (the compiler warns about exactly this:
    // "a __device__ variable cannot be directly read in a host
    // function"). An earlier version did `WALK_TESTS[t]` directly in
    // main(), which silently read garbage/undefined memory rather than
    // erroring, making every test's expected-key comparison meaningless
    // -- the kernel was very likely finding the CORRECT key the whole
    // time (it verifies scalar_mult(k)==pubkey internally before ever
    // returning found=true), but main() was comparing against garbage.
    // Fix: explicitly copy the array to host memory first.
    WalkTestCase h_walk_tests[N_WALK_TESTS];
    cudaMemcpyFromSymbol(h_walk_tests, WALK_TESTS, N_WALK_TESTS * sizeof(WalkTestCase));

    int failures = 0;

    for (int t = 0; t < N_WALK_TESTS; t++) {
        // Clear the DP table BEFORE starting the timer -- this is setup,
        // not walk-logic work, and cudaMemset's bulk clear is fast
        // enough (microseconds, not the ~9.7s the old single-thread
        // device-side loop took) that it wouldn't meaningfully skew
        // results even if included, but keeping it outside the timed
        // region is the more honest measurement.
        cudaMemset(d_dp_table, 0, DP_TABLE_CAPACITY * sizeof(DPEntry));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        run_walk_test<<<1, 1>>>(t, d_found, d_key_limbs, d_jumps, d_dp_table);
        cudaEventRecord(stop);

        if (check_cuda(cudaGetLastError(), "kernel launch")) { failures++; continue; }
        if (check_cuda(cudaDeviceSynchronize(), "kernel execution")) { failures++; continue; }

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        int h_found;
        uint64_t h_key_limbs[4];
        uint64_t h_jumps;
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_key_limbs, d_key_limbs, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_jumps, d_jumps, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        WalkTestCase tc = h_walk_tests[t];
        bool key_matches = h_found &&
            h_key_limbs[0] == tc.expected_key[0] && h_key_limbs[1] == tc.expected_key[1] &&
            h_key_limbs[2] == tc.expected_key[2] && h_key_limbs[3] == tc.expected_key[3];

        double jumps_per_sec = (elapsed_ms > 0) ? (h_jumps / (elapsed_ms / 1000.0)) : 0.0;

        printf("[%s] test %d: found=%d key_matches=%d jumps=%llu time=%.2fms (%.0f jumps/sec)\n",
               key_matches ? "PASS" : "FAIL", t, h_found, key_matches,
               (unsigned long long)h_jumps, elapsed_ms, jumps_per_sec);
        if (!key_matches) failures++;
    }

    cudaFree(d_dp_table);
    cudaFree(d_found);
    cudaFree(d_key_limbs);
    cudaFree(d_jumps);

    printf("----------------------------------------------------------------------\n");
    if (failures == 0) {
        printf("All %d kangaroo walk test cases PASSED -- recovered the exact\n", N_WALK_TESTS);
        printf("correct private key in every case, matching the Python reference.\n");
    } else {
        printf("%d/%d test case(s) FAILED.\n", failures, N_WALK_TESTS);
    }
    return failures == 0 ? 0 : 1;
}
