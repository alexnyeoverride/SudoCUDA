#pragma once

#include <cuda_runtime.h>


template <typename T>
struct DeviceArray {
	int capacity;
	int* count;
	T* data;

	__host__ __device__ static DeviceArray create(const int capacity) {
		int* count;
		T* data;

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

		if (index >= capacity) {
			*count = capacity;
		} else {
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
};


template <typename T>
class HostArray {
	DeviceArray<T> array;

public:
	__host__ explicit HostArray(int capacity)
		: array(DeviceArray<T>::create(capacity)) {}

	__host__ explicit HostArray(DeviceArray<T> array)
		: array(array) {}

	__host__ ~HostArray() {
		array.destroy();
	}

	HostArray(const HostArray&) = delete;
	HostArray& operator=(const HostArray&) = delete;

	__host__ HostArray(HostArray&& other) noexcept
		: array(std::move(other.array)) {}

	__host__ HostArray& operator=(HostArray&& other) noexcept {
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