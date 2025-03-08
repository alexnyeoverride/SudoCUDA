#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include "array.cuh"
#include <cstdint>

#define _ (-1)
using Board = std::array<std::array<int8_t, 9>, 9>;

// Each unit of work is a board with all values except one known, plus a guess.
struct Work {
	Board board;
	int8_t guessRow;
	int8_t guessColumn;
};


// TODO: push all guesses to the stack.  (at each push, check for an error, and inner-loop-retry)
__device__ __host__ void makeGuesses(
	Board* workingBoard,
	DeviceArray<Work> workStealingStack
) {
	for (int8_t row = 0; row < 9; ++row) {
		for (int8_t col = 0; col < 9; ++col) {
			if ((*workingBoard)[row][col] == _) {
				for (int8_t guess = 1; guess <= 9; ++guess) {
					Board newBoard = *workingBoard;
					newBoard[row][col] = guess;

					/*
					const auto result = workStealingStack.push({
						newBoard,
						row,
						col
					});
					// TODO: threads could deadlock here.  Maybe split threads into solvers and guessers / poppers and pushers.
					if (result.error == Error::Overflow) {
						// TODO: inner-loop-retry.
					}
					*/

					Result<Work> result;
					do {
						result = workStealingStack.push({
							newBoard,
							row,
							col
						});
					} while (result.error == Error::Underflow);
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

// CUDA parallel work-stealing Sudoku solver.

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

// TODO: optimization.  After determining what guesses to make, push n-1 of them to the stack,
//		and just work on one without pushing then immediately popping it.
__global__ void solve(
	Board* knownValues,
	DeviceArray<Work> workStealingStack
) {
	// TODO: a way to break the loop for unsolvable puzzles.  If all threads are waiting for work, and the stack is empty.
	//		Maybe add a numWaitingThreads argument.

	while (true) {
		if (complete(knownValues)) {
			break;
		}

		auto [error, work] = workStealingStack.pop();
		if (error == Error::Underflow) {
			continue; // Wait for work. // TODO: unless puzzle is unsolvable.
		}

		if (!applyConstraints(&work)) {
			continue; // This guess is invalid.  Look for new work.
		}

		if (complete(&work.board)) {
			*knownValues = work.board;
		}

		// TODO: work path permutation optimization would go here.
		//		If the work guess path is a permutation of a prior guess path, skip it.
		//		Advanced optimization, need a hash table of ordered position-value pair set keys.

		makeGuesses(knownValues, workStealingStack);

		printf("%d\n", workStealingStack.getCount()); // TODO: debugging
	}
}


int main() {
	Board* knownValues;
	cudaMallocManaged(&knownValues, sizeof(Board));

	// TODO: what capacity makes sense?  How do I measure the impact of threads' waiting to insert work?
	auto workStealingStack = HostArray<Work>(1024*1024*100);

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
	*knownValues = puzzle;

	// Initialize workStealingStack to the first layer of guesses for all empty cells.
	makeGuesses(knownValues, workStealingStack);

	solve<<<1, 32>>>(knownValues, workStealingStack);
	if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << std::endl;
	}
}