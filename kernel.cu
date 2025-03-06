#include <iostream>
#include <cuda_runtime.h>
#include "array.cuh"


__global__ void kernel(AgnosticArray<int> v) {
	v.push(threadIdx.x);
}


int main() {
	auto v = HostArray<int>(10);

	kernel<<<1, 20>>>(v);

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << std::endl;
	}
	std::cout << v[0] << std::endl;
	std::cout << v[1] << std::endl;
	std::cout << v[2] << std::endl;
	std::cout << v[3] << std::endl;

	return 0;
}