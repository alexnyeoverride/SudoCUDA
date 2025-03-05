#include <iostream>
#include <cuda_runtime.h>


template <typename T>
struct DeviceArray {
	int capacity;
	int* count;
	T* data;

	class AgnosticArray;

	__host__ __device__ static DeviceArray create(const int capacity) {
		int* count;
		T* data;
		// TODO: conditionally compile for host or device, with cudaMallocManaged on the host and ??? on the device.
		// This way the device will be able to dynamically allocate memory.
		cudaMallocManaged(&data, sizeof(T) * capacity);
		cudaMallocManaged(&count, sizeof(int));
		*count = 0;
		return { capacity, count, data };
	}

	__host__ void destroy() {
		cudaFree(data);
		cudaFree(count);
	}

	__host__ __device__ void push(T value) {
#ifdef __CUDA_ARCH__
		const auto index = atomicAdd(count, 1);
#else
		const auto index = *count;
		*count += 1;
#endif
		if (index < capacity) {
			data[index] = value;
		}
	}

	__host__ __device__ T& operator[](int index) {
		return data[index];
	}

	__host__ __device__ T& operator[](int index) const {
		return data[index];
	}

	__host__ __device__ int getCount() const {
		return *count;
	}

	__host__ __device__ int getCapacity() const {
		return capacity;
	}

	__host__ operator AgnosticArray() const noexcept;
};



template <typename T>
class AgnosticArray {
	DeviceArray<T> array;

public:
	__host__ explicit AgnosticArray(int capacity)
		: array(DeviceArray<T>::create(capacity)) {}

	__host__ explicit AgnosticArray(DeviceArray<T> array)
		: array(array) {}

	__host__ ~AgnosticArray() {
		array.destroy();
	}

	AgnosticArray(const AgnosticArray&) = delete;
	AgnosticArray& operator=(const AgnosticArray&) = delete;

	__host__ AgnosticArray(AgnosticArray&& other) noexcept
		: array(std::move(other.array)) {}

	__host__ AgnosticArray& operator=(AgnosticArray&& other) noexcept {
		if (this != &other) {
			array.destroy();
			array = std::move(other.array);
		}
		return *this;
	}

	// Implicit conversion to DynamicArray<T> for lending to kernels.
	__host__ operator DeviceArray<T>() const noexcept { // NOLINT(*-explicit-constructor)
		return array;
	}

	__host__ T& operator[](int index) { return array[index]; }
	__host__ const T& operator[](int index) const { return array[index]; }
	__host__ int getCount() const { return array.getCount(); }
	__host__ int getCapacity() const { return array.getCapacity(); }
};


template <typename T>
__host__ DeviceArray<T>::operator typename AgnosticArray() noexcept {
	return AgnosticArray<T>(*this);
}


__global__ void kernel(DeviceArray<int> v) {
	v.push(threadIdx.x);
}


int main() {
	auto v = AgnosticArray<int>(10);

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