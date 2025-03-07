#include <iostream>
#include <cuda_runtime.h>
#include "array.cuh"


__global__ void kernel(DeviceArray<int> v) {
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

	for (int i = 0; i < v.getCount(); i++) {
		std::cout << v[i] << std::endl;
	}

	return 0;
}