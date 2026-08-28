// test_point_arithmetic.cu — Phase 4b self-test: point_add and
// scalar_mult against test vectors from the trusted Python reference.
//
// Build:
//   nvcc -O3 -arch=sm_86 -o test_point_arithmetic test_point_arithmetic.cu
// Run:
//   ./test_point_arithmetic

#include <cstdio>
#include "secp256k1_point.cuh"
#include "point_test_vectors.h"

__device__ ECPoint load_point(uint64_t x_src[4], uint64_t y_src[4], int inf) {
    ECPoint p;
    for (int i = 0; i < 4; i++) { p.x.limb[i] = x_src[i]; p.y.limb[i] = y_src[i]; }
    p.infinity = (inf != 0);
    return p;
}

__device__ bool point_equal(const ECPoint& a, const ECPoint& b) {
    if (a.infinity != b.infinity) return false;
    if (a.infinity) return true;  // both infinity, coordinates don't matter
    return u256_equal(a.x, b.x) && u256_equal(a.y, b.y);
}

__global__ void test_add_kernel(int* results) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_ADD_TESTS) return;
    ECPoint p1 = load_point(ADDT_P1_X[i], ADDT_P1_Y[i], ADDT_P1_INF[i]);
    ECPoint p2 = load_point(ADDT_P2_X[i], ADDT_P2_Y[i], ADDT_P2_INF[i]);
    ECPoint expected = load_point(ADDT_EXPECTED_X[i], ADDT_EXPECTED_Y[i], ADDT_EXPECTED_INF[i]);
    ECPoint got = point_add(p1, p2);
    results[i] = point_equal(got, expected) ? 1 : 0;
}

__global__ void test_scalar_kernel(int* results) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_SCALAR_TESTS) return;
    u256 k; for (int j = 0; j < 4; j++) k.limb[j] = SCALARS[i][j];
    ECPoint expected = load_point(SCALAR_EXPECTED_X[i], SCALAR_EXPECTED_Y[i], SCALAR_EXPECTED_INF[i]);
    ECPoint g = secp256k1_generator();
    ECPoint got = scalar_mult(k, g);
    results[i] = point_equal(got, expected) ? 1 : 0;
}

int check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR during %s: %s\n", what, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

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
    printf("secp256k1 point arithmetic self-test (Phase 4b)\n");
    printf("Built on field arithmetic already verified on this hardware (Phase 4a)\n");
    printf("----------------------------------------------------------------------\n");

    int failures = 0;
    int threads = 256;
    int* d_results;

    cudaMalloc(&d_results, N_ADD_TESTS * sizeof(int));
    test_add_kernel<<<(N_ADD_TESTS + threads - 1) / threads, threads>>>(d_results);
    failures += (collect_results("point_add", N_ADD_TESTS, d_results) != 0);
    cudaFree(d_results);

    cudaMalloc(&d_results, N_SCALAR_TESTS * sizeof(int));
    test_scalar_kernel<<<(N_SCALAR_TESTS + threads - 1) / threads, threads>>>(d_results);
    failures += (collect_results("scalar_mult", N_SCALAR_TESTS, d_results) != 0);
    cudaFree(d_results);

    printf("----------------------------------------------------------------------\n");
    if (failures == 0) {
        printf("All point arithmetic test groups PASSED.\n");
        printf("Includes: doubling, P+(-P)==infinity, infinity identity cases,\n");
        printf("k=0/1/N-1, and the actual solved puzzle #135 private key as an\n");
        printf("independent known-answer check.\n");
    } else {
        printf("%d test group(s) FAILED. Do not proceed to the kangaroo walk\n", failures);
        printf("kernel until these pass -- point arithmetic is the foundation\n");
        printf("everything else in the walk builds on.\n");
    }
    return failures == 0 ? 0 : 1;
}
