// test_field_arithmetic.cu — Phase 4a self-test: runs the field
// arithmetic in secp256k1_field.cuh against test vectors generated from
// the TRUSTED Python reference (test_vectors.h, from generate_test_vectors.py,
// which itself was checked against secp256k1.py's known-answer tests).
//
// This is the step that actually closes the verification loop I could
// not close myself (no CUDA toolkit / GPU in the environment this was
// written in). Compile and run this on the HiveOS rig; a clean PASS on
// every group means the field arithmetic transcription is correct.
//
// Build:
//   nvcc -O3 -arch=sm_86 -o test_field_arithmetic test_field_arithmetic.cu
// Run:
//   ./test_field_arithmetic

#include <cstdio>
#include "secp256k1_field.cuh"
#include "test_vectors.h"

__device__ bool u256_equal(const u256& a, const u256& b) {
    for (int i = 0; i < NUM_LIMBS; i++) {
        if (a.limb[i] != b.limb[i]) return false;
    }
    return true;
}

__device__ u256 load_u256(uint64_t src[4]) {
    u256 r;
    for (int i = 0; i < 4; i++) r.limb[i] = src[i];
    return r;
}

__global__ void test_mul_kernel(int* results) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_MUL_VECTORS) return;
    u256 a = load_u256(MUL_A[i]);
    u256 b = load_u256(MUL_B[i]);
    u256 expected = load_u256(MUL_EXPECTED[i]);
    u256 got = mul_mod(a, b);
    results[i] = u256_equal(got, expected) ? 1 : 0;
}

__global__ void test_add_kernel(int* results) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_ADD_VECTORS) return;
    u256 a = load_u256(ADD_A[i]);
    u256 b = load_u256(ADD_B[i]);
    u256 expected = load_u256(ADD_EXPECTED[i]);
    u256 got = add_mod(a, b);
    results[i] = u256_equal(got, expected) ? 1 : 0;
}

__global__ void test_sub_kernel(int* results) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_SUB_VECTORS) return;
    u256 a = load_u256(SUB_A[i]);
    u256 b = load_u256(SUB_B[i]);
    u256 expected = load_u256(SUB_EXPECTED[i]);
    u256 got = sub_mod(a, b);
    results[i] = u256_equal(got, expected) ? 1 : 0;
}

__global__ void test_inv_kernel(int* results) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_INV_VECTORS) return;
    u256 a = load_u256(INV_A[i]);
    u256 expected = load_u256(INV_EXPECTED[i]);
    u256 got = inv_mod(a);
    results[i] = u256_equal(got, expected) ? 1 : 0;

    // Also verify the fundamental property a * inv(a) == 1, independent
    // of whether it matches the precomputed expected value -- catches
    // the case where the test vector itself might be wrong, not just
    // the kernel.
    u256 one; one.limb[0] = 1; one.limb[1] = 0; one.limb[2] = 0; one.limb[3] = 0;
    u256 product = mul_mod(a, got);
    if (!u256_equal(product, one)) {
        results[i] = 0;
    }
}

int check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR during %s: %s\n", what, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// Shared result-collection logic. Kernel launches are kept separate and
// explicit in main() (rather than passed in as function pointers) --
// simpler and lower-risk to get right without being able to compile-test
// this, at the cost of a little repetition in main().
int collect_results(const char* name, int n, int* d_results) {
    if (check_cuda(cudaGetLastError(), "kernel launch")) return -1;
    if (check_cuda(cudaDeviceSynchronize(), "kernel execution")) return -1;

    int* h_results = new int[n];
    cudaMemcpy(h_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost);

    int pass_count = 0;
    int first_fail = -1;
    for (int i = 0; i < n; i++) {
        if (h_results[i]) pass_count++;
        else if (first_fail < 0) first_fail = i;
    }

    printf("[%s] %s: %d/%d correct", pass_count == n ? "PASS" : "FAIL", name, pass_count, n);
    if (first_fail >= 0) printf(" (first failure at index %d)", first_fail);
    printf("\n");

    delete[] h_results;
    return (pass_count == n) ? 0 : 1;
}

int main() {
    printf("secp256k1 field arithmetic self-test (Phase 4a)\n");
    printf("Test vectors generated from the trusted Python reference (secp256k1.py)\n");
    printf("----------------------------------------------------------------------\n");

    int failures = 0;
    int threads = 256;
    int* d_results;

    // mul_mod
    cudaMalloc(&d_results, N_MUL_VECTORS * sizeof(int));
    test_mul_kernel<<<(N_MUL_VECTORS + threads - 1) / threads, threads>>>(d_results);
    failures += (collect_results("mul_mod", N_MUL_VECTORS, d_results) != 0);
    cudaFree(d_results);

    // add_mod
    cudaMalloc(&d_results, N_ADD_VECTORS * sizeof(int));
    test_add_kernel<<<(N_ADD_VECTORS + threads - 1) / threads, threads>>>(d_results);
    failures += (collect_results("add_mod", N_ADD_VECTORS, d_results) != 0);
    cudaFree(d_results);

    // sub_mod
    cudaMalloc(&d_results, N_SUB_VECTORS * sizeof(int));
    test_sub_kernel<<<(N_SUB_VECTORS + threads - 1) / threads, threads>>>(d_results);
    failures += (collect_results("sub_mod", N_SUB_VECTORS, d_results) != 0);
    cudaFree(d_results);

    // inv_mod
    cudaMalloc(&d_results, N_INV_VECTORS * sizeof(int));
    test_inv_kernel<<<(N_INV_VECTORS + threads - 1) / threads, threads>>>(d_results);
    failures += (collect_results("inv_mod", N_INV_VECTORS, d_results) != 0);
    cudaFree(d_results);

    printf("----------------------------------------------------------------------\n");
    if (failures == 0) {
        printf("All field arithmetic test groups PASSED.\n");
        printf("The CUDA transcription matches the Python reference exactly.\n");
    } else {
        printf("%d test group(s) FAILED. Do not proceed to point arithmetic\n", failures);
        printf("until these pass -- everything else builds on this being correct.\n");
    }
    return failures == 0 ? 0 : 1;
}
