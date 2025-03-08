#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include "array.cuh"
#include <cstdint>

#define _ (-1)
using Board = std::array<std::array<int8_t, 9>, 9>;

struct Work {
	Board board;
	int8_t guessRow;
	int8_t guessColumn;
};


// hard
/*
constexpr Board puzzle = {
	8, _, _, _, _, _, _, _, _,
	_, _, 3, 6, _, _, _, _, _,
	_, 7, _, _, 9, _, 2, _, _,
	_, 5, _, _, _, 7, _, _, _,
	_, _, _, _, 4, 5, 7, _, _,
	_, _, _, 1, _, _, _, 3, _,
	_, _, 1, _, _, _, _, 6, 8,
	_, _, 8, 5, _, _, _, 1, _,
	_, 9, _, _, _, _, 4, _, _
};
*/
// easy
constexpr Board puzzle = {
    2, 5, 6, 8, 3, 7, 1, 4, 9,
	7, 1, 9, 4, _, _, 8, 3, 6,
	8, 4, 3, 6, 1, 9, 2, 5, 7,
	4, 6, 7, 1, 5, 8, 9, 2, 3,
	3, 9, _, 7, 6, 4, 5, 1, 8,
	5, 8, 1, 3, 9, 2, 6, 7, 4,
	1, 7, 8, 2, 4, 6, 3, 9, 5,
	6, 3, 5, 9, 7, 1, 4, 8, 2,
	9, 2, 4, 5, 8, 3, 7, 6, _
};

__device__ int64_t xxhash(int64_t x) {
	x *= 2654435761;
	x = (x << 13) | (x >> (32 - 13));
	x *= 2654435761;
	x = (x << 17) | (x >> (32 - 17));
	x *= 2654435761;
	x = (x << 5) | (x >> (32 - 5));
	return x;
}

__device__ int64_t hashBoard(const Board& board) {
	auto runningHash = 0;
	for (int8_t i = 0; i < 9; i++) {
		for (int8_t j = 0; j < 9; j++) {
			runningHash = xxhash(board[i][j] ^ runningHash);
		}
	}
	return runningHash;
}

__device__ void makeGuesses(
	Board* workingBoard,
	DeviceArray<Work> workStack,
	int* numWorkingThreads
) {
	for (int8_t row = 0; row < 9; ++row) {
		for (int8_t col = 0; col < 9; ++col) {
			if ((*workingBoard)[row][col] == _) {
				for (int8_t guess = 1; guess <= 9; ++guess) {
					Board newBoard = *workingBoard;
					newBoard[row][col] = guess;
					Result<Work> result;

					auto numTries = 0;
					do {
						result = workStack.push({
							newBoard,
							row,
							col
						});

						numTries++;
						if (numTries > 100) {
							printf("Retry limit reached.  Exiting.\n");
							break;
						}
					} while (result.error == Error::Overflow);

					atomicSub(numWorkingThreads, 1);
				}
			}
		}
	}
}

__device__ bool applyConstraints(
	const Work* work
) {
	const int8_t column = work->guessColumn;
	const int8_t row = work->guessRow;
	const auto workingBoard = work->board;

	// Apply constraint to row
	for (int8_t i = 0; i < 9; i++) {
		if (i == column) {
			continue;
		}
		if (workingBoard[row][i] == workingBoard[row][column]) {
			return false;
		}
	}
	// Apply constraint to column
	for (int8_t i = 0; i < 9; i++) {
		if (i == row) {
			continue;
		}
		if (workingBoard[i][column] == workingBoard[row][column]) {
			return false;
		}
	}
	// Apply constraint to box
	const int8_t boxStartI = row / 3 * 3;
	const int8_t boxStartJ = column / 3 * 3;
	for (int8_t i = boxStartI; i < boxStartI + 3; i++) {
		for (int8_t j = boxStartJ; j < boxStartJ + 3; j++) {
			if (row == i && column == j) {
				continue;
			}
			if (workingBoard[i][j] == workingBoard[row][column]) {
				return false;
			}
		}
	}

	return true;
}

__device__ bool complete(const Board* board) {
	for (int8_t i = 0; i < 9; i++) {
		for (int8_t j = 0; j < 9; j++) {
			if ((*board)[i][j] == _) {
				return false;
			}
		}
	}
	return true;
}

__host__ __device__ void printBoard(const Board* board) {
	for (int8_t i = 0; i < 9; i++) {
		for (int8_t j = 0; j < 9; j++) {
			if ((*board)[i][j] == _)
                printf("_ ");
            else
			printf("%d ", (*board)[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

// TODO: 64 bits is probably not enough.
__device__ int64_t guessPathEquivalenceClassBloomFilter;

__global__ void solve(
	Board* knownValues,
	DeviceArray<Work> workStack,
	int* numWorkingThreads
) {
	auto numCycles = 0;

	while (true) {
		numCycles++;
		if (numCycles > 10000000) {
			printf("Solution iteration limit reached.  Exiting.\n");
            break;
        }

		if (complete(knownValues)) {
			break;
		}

		auto [error, work] = workStack.pop();
		if (error == Error::Underflow) {
			continue; // Wait for work.
		}

		if (!applyConstraints(&work)) {
			continue; // This guess is invalid.  Look for new work.
		}

		if (complete(&work.board)) {
			*knownValues = work.board;
			break;
		}


		/* TODO
		const auto boardHash = hashBoard(work.board);
		if ((guessPathEquivalenceClassBloomFilter & boardHash) == boardHash) {
			printf("Duplicate guess path found.  Skipping.\n");
			continue; // We've made this same set of guesses in another order before.
		} else {
			atomicOr(&guessPathEquivalenceClassBloomFilter, boardHash);
		}
		*/

		atomicAdd(numWorkingThreads, 1);
		makeGuesses(&work.board, workStack, numWorkingThreads);
	}
}

int main() {
	Board* knownValues;
	if (
		cudaMallocManaged(&knownValues, sizeof(Board)) !=
		cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}
	*knownValues = puzzle;

	// TODO: what capacity makes sense?  How do I measure the impact of threads' waiting to insert work?
	auto workStack = HostArray<Work>(1024*1024*100);

	int* numWorkingThreads;
	if (
		cudaMallocManaged(&numWorkingThreads, sizeof(int)) !=
		cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}
	*numWorkingThreads = 0;


	workStack.push({ *knownValues, 0, 0 });

	cudaDeviceSynchronize();
	if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		printf("makeGuesses failed");
		std::cerr << cudaGetErrorString(err) << std::endl;
	}

	printf("Initial work stack size: %d\n", workStack.getCount());

	solve<<<1, 32>>>(knownValues, workStack, numWorkingThreads);
	cudaDeviceSynchronize();

	if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		printf("solve failed");
		std::cerr << cudaGetErrorString(err) << std::endl;
	}

	std::cout << "Solution:\n";
	printBoard(knownValues);

	cudaFree(knownValues);
	cudaFree(numWorkingThreads);
	return 0;
}