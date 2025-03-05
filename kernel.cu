#include <iostream>
#include <cuda_runtime.h>
#include <stdgpu/vector.cuh>

static void printErrors() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << std::endl;
	}
}

__global__ void kernel(int* v, int* n, int cap) {
	atomicAdd(n, 1);
	if (*n < cap) {
		v[*n] = 42;
	}
}

int main() {
	// dynamic array.
	int* v;
	int* n;
	int cap = 2;

	cudaMallocManaged(reinterpret_cast<void **>(&n), sizeof(int));
	cudaMallocManaged(reinterpret_cast<void **>(&v), sizeof(int) * cap);

	kernel<<<1, 1>>>(v, n, cap);
	cudaDeviceSynchronize();
	printErrors();

	std::cout << v[1] << std::endl;

	cudaFree(v);
	return 0;
}