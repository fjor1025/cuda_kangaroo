// test_kangaroo_parallel.cu — Phase 4d self-test: runs the parallel
// (many-kangaroo, shared atomic DP table) kernel against the same test
// cases used for the single-thread version (walk_test_vectors.h),
// checking both correctness (exact key recovery) and aggregate
// throughput.
//
// Build:
//   nvcc -O3 -arch=sm_86 -o test_kangaroo_parallel test_kangaroo_parallel.cu
// Run:
//   ./test_kangaroo_parallel [num_tame] [num_wild]
// (defaults: 128 tame + 128 wild = 256 total kangaroos, if not specified)

#include <cstdio>
#include <cstdlib>
#include "kangaroo_walk_parallel.cuh"
#include "walk_test_vectors.h"

int check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR during %s: %s\n", what, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    // See test_kangaroo_walk.cu for why this matters: CUDA's default
    // sync policy can spin-poll a CPU core at 100% for the whole kernel
    // duration instead of sleeping the host thread, which is a real
    // concern for a rig whose CPU isn't built for sustained load. This
    // is even more relevant here, since the parallel kernel runs for
    // longer stretches than the single-thread version.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    int num_tame = (argc > 1) ? atoi(argv[1]) : 128;
    int num_wild = (argc > 2) ? atoi(argv[2]) : 128;
    int total_kangaroos = num_tame + num_wild;

    printf("Parallel kangaroo walk self-test (Phase 4d)\n");
    printf("Built on the single-thread walk (4c), field (4a) and point (4b)\n");
    printf("arithmetic, all already verified on this hardware.\n");
    printf("Kangaroos: %d tame + %d wild = %d total\n", num_tame, num_wild, total_kangaroos);
    printf("----------------------------------------------------------------------\n");

    WalkTestCase h_walk_tests[N_WALK_TESTS];
    cudaMemcpyFromSymbol(h_walk_tests, WALK_TESTS, N_WALK_TESTS * sizeof(WalkTestCase));

    int* d_exponents;
    ECPoint* d_jump_points;
    ParallelDPEntry* d_dp_table;
    int* d_global_found;
    uint64_t* d_key_limbs;
    uint64_t* d_total_jumps;

    uint64_t dp_capacity = 1 << 20;  // generous: 1M slots, well beyond what
                                      // any single test case here should need
    cudaMalloc(&d_exponents, JUMP_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_jump_points, JUMP_TABLE_SIZE * sizeof(ECPoint));
    cudaMalloc(&d_dp_table, dp_capacity * sizeof(ParallelDPEntry));
    cudaMalloc(&d_global_found, sizeof(int));
    cudaMalloc(&d_key_limbs, 4 * sizeof(uint64_t));
    cudaMalloc(&d_total_jumps, sizeof(uint64_t));

    int failures = 0;
    int threads_per_block = 256;
    int blocks = (total_kangaroos + threads_per_block - 1) / threads_per_block;

    for (int t = 0; t < N_WALK_TESTS; t++) {
        WalkTestCase tc = h_walk_tests[t];

        u256 a, b, pubkey_x, pubkey_y;
        for (int i = 0; i < 4; i++) {
            a.limb[i] = tc.a[i];
            b.limb[i] = tc.b[i];
            pubkey_x.limb[i] = tc.pubkey_x[i];
            pubkey_y.limb[i] = tc.pubkey_y[i];
        }
        ECPoint pubkey; pubkey.x = pubkey_x; pubkey.y = pubkey_y; pubkey.infinity = false;

        cudaMemcpy(d_exponents, tc.exponents, JUMP_TABLE_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        uint64_t mean_jump_sum = 0;
        for (int i = 0; i < JUMP_TABLE_SIZE; i++) mean_jump_sum += (1ULL << tc.exponents[i]);
        uint64_t mean_jump = mean_jump_sum / JUMP_TABLE_SIZE;

        cudaMemset(d_dp_table, 0, dp_capacity * sizeof(ParallelDPEntry));
        cudaMemset(d_global_found, 0, sizeof(int));
        cudaMemset(d_total_jumps, 0, sizeof(uint64_t));

        // One-time jump table build (see kangaroo_walk_parallel.cuh for
        // why this is a separate kernel rather than per-thread work).
        build_jump_table_kernel<<<1, JUMP_TABLE_SIZE>>>(d_exponents, JUMP_TABLE_SIZE, d_jump_points);
        if (check_cuda(cudaGetLastError(), "build_jump_table_kernel launch")) { failures++; continue; }
        if (check_cuda(cudaDeviceSynchronize(), "build_jump_table_kernel execution")) { failures++; continue; }

        // Per-thread jump budget: same generous headroom multiplier the
        // single-thread test used, but divided across however many
        // kangaroos are now sharing the work.
        uint64_t max_jumps_per_thread = (tc.max_jumps / total_kangaroos) + 1000;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        parallel_kangaroo_kernel<<<blocks, threads_per_block>>>(
            pubkey, a, b, tc.dp_bits, d_exponents, d_jump_points, JUMP_TABLE_SIZE,
            mean_jump, max_jumps_per_thread, num_tame, num_wild,
            d_dp_table, dp_capacity, d_global_found, d_key_limbs, d_total_jumps);
        cudaEventRecord(stop);

        if (check_cuda(cudaGetLastError(), "parallel_kangaroo_kernel launch")) { failures++; continue; }
        if (check_cuda(cudaDeviceSynchronize(), "parallel_kangaroo_kernel execution")) { failures++; continue; }

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        int h_found;
        uint64_t h_key_limbs[4];
        uint64_t h_total_jumps;
        cudaMemcpy(&h_found, d_global_found, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_key_limbs, d_key_limbs, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_total_jumps, d_total_jumps, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        bool key_matches = h_found &&
            h_key_limbs[0] == tc.expected_key[0] && h_key_limbs[1] == tc.expected_key[1] &&
            h_key_limbs[2] == tc.expected_key[2] && h_key_limbs[3] == tc.expected_key[3];

        double jumps_per_sec = (elapsed_ms > 0) ? (h_total_jumps / (elapsed_ms / 1000.0)) : 0.0;

        printf("[%s] test %d: found=%d key_matches=%d total_jumps=%llu time=%.2fms (%.0f aggregate jumps/sec)\n",
               key_matches ? "PASS" : "FAIL", t, h_found, key_matches,
               (unsigned long long)h_total_jumps, elapsed_ms, jumps_per_sec);
        if (!key_matches) failures++;
    }

    cudaFree(d_exponents);
    cudaFree(d_jump_points);
    cudaFree(d_dp_table);
    cudaFree(d_global_found);
    cudaFree(d_key_limbs);
    cudaFree(d_total_jumps);

    printf("----------------------------------------------------------------------\n");
    if (failures == 0) {
        printf("All %d parallel kangaroo test cases PASSED.\n", N_WALK_TESTS);
    } else {
        printf("%d/%d test case(s) FAILED.\n", failures, N_WALK_TESTS);
    }
    return failures == 0 ? 0 : 1;
}
