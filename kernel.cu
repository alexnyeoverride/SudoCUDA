#include <iostream>
#include <cuda_runtime.h>
#include "array.cuh"


// TODO: CUDA sudoku solver algorithm:  parallel backtracking.  Not actual backtracking tho.
//		If a thread runs into a dead end, push its first guess to `invalid`.
//		Dynamic parallelism looks just like recursion on the CPU.
//		Use a 2D array to store the sudoku board's known values.
//		All threads share an invalid value data structure.
//		Use a stack to store the guesses.  Use a 3D array to store the invalid values.


__global__ void kernel(DeviceArray<int> v) {
	v.push(threadIdx.x);
	if (false)
	kernel<<<1, 1>>>(v);
}


int main() {
	auto v = HostArray<int>(10);

	kernel<<<1, 20>>>(v);

	cudaDeviceSynchronize();
	if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << std::endl;
	}

	for (int i = 0; i < v.getCount(); i++) {
		std::cout << v[i] << std::endl;
	}

	return 0;
}