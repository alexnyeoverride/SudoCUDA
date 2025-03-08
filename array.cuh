#pragma once

#include <cassert>
#include <cuda_runtime.h>

enum class Error : int {
    None,
    Overflow,
	Underflow,
};
template <typename T>
struct Result {
    Error error;
	T value;
};


template <typename T>
struct DeviceArray {
	int capacity;
	int* count;
	T* data;

	__host__ static DeviceArray create(const int capacity) {
		int* count;
		T* data;

		const auto maybeError = cudaMallocManaged(&data, sizeof(T) * capacity);
		cudaMallocManaged(&count, sizeof(int));

		if (maybeError != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(maybeError) << std::endl;
        }

		*count = 0;
		return { capacity, count, data };
	}

	__host__ void destroy() {
		cudaFree(data);
		cudaFree(count);
	}

	// TODO: return an error type instead of using assert.
	__host__ __device__ Result<T> push(T value) {
#ifdef __CUDA_ARCH__
		const auto index = atomicAdd(count, 1);
#else
		const auto index = *count;
		*count += 1; // TODO: atomic addition on the CPU.  include <atomic> and call std::atomic_fetch_add.
#endif

		if (index >= capacity) {
			*count = capacity;
			return { Error::Overflow };
		} else {
			data[index] = value;
			return { Error::None };
		}
	}

	__host__ __device__ Result<T> pop() {
		Result<T> R;
#ifdef __CUDA_ARCH__
		const auto index = atomicSub(count, 1);
#else
		const auto index = *count;
		*count -= 1; // TODO: atomic subtraction on the CPU.  include <atomic> and call std::atomic_fetch_sub.
#endif
		if (index <= 0) {
            *count = 0;
			return {Error::Underflow};
        } else {
        	return {
        		Error::None,
        		data[index - 1]
        	};
        }
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

	// Implicit conversion to DynamicArray<T> for lending to kernels.
	__host__ operator DeviceArray<T>() const noexcept { // NOLINT(*-explicit-constructor)
		return array;
	}

	__host__ int getCount() const { return array.getCount(); }
	__host__ int getCapacity() const { return array.getCapacity(); }
	__host__ void push(T value) { array.push(value); }
};