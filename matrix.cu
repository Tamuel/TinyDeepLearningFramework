#include "matrix.h"
#include <fstream>
#include <string>
#include <math.h>
#include "srcMeasure.h"

#define BLOCK_DIM 32
#define ZERO 0.0000000001
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

void checkCudaError() {
	if (DEBUG_M) {
		cudaError_t e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("cuda failure '%s' at %s:%d:\n",
				cudaGetErrorString(e),
				__FILE__, __LINE__);
			exit(1);
		}
	}
}

int Matrix::numberOfMatrices = 0;
bool Matrix::mInitThread = false;
bool Matrix::mThreadRun[N_THREAD] = { false };
bool Matrix::mThreadStart[N_THREAD] = { false };
bool Matrix::mThreadWait[N_THREAD] = { false };
bool Matrix::mThreadFinish[N_THREAD] = { false };
mutex Matrix::mThreadsReadyMtx[N_THREAD];
mutex Matrix::mNumberLock;
int Matrix::runThreads = 0;
condition_variable Matrix::mThreadsReadyCV;
vector<thread> Matrix::mThreads;
threadMode Matrix::mMode;
Matrix* Matrix::threadM1;
Matrix* Matrix::threadM2;
Matrix Matrix::threadResult;
Matrix* Matrix::threadDestination;
dtype Matrix::threadScal;
bool Matrix::threadBool;
bool* Matrix::gpuBool;
void* Matrix::threadArg1;
void* Matrix::threadArg2;
cudaStream_t Matrix::cudaStream[N_STREAM];
int Matrix::currentStream;

bool Matrix::gpuIsReady = false;

void Matrix::readyForGpuCalc() {
	if (!gpuIsReady) {
		gpuIsReady = true;
		cudaSetDevice(1);
		cudaMalloc((void**)&gpuBool, sizeof(bool));
		for (int i = 0; i < N_STREAM; i++) {
			cudaStreamCreateWithFlags(&cudaStream[i], cudaStreamNonBlocking);
		}
		currentStream = 0;
		checkCudaError();
	}
}

void Matrix::initThreads() {
	if (mInitThread == false) {
		mInitThread = true;
		for (int i = 0; i < N_THREAD; i++) {
			mThreadRun[i] = true;
			mThreadStart[i] = false;
			mThreadWait[i] = false;
			mThreadFinish[i] = false;
			mThreads.push_back(thread(&threadFunction, i));
			mThreads[i].detach();
		}
	}
}

void Matrix::stopThreads() {
	mInitThread = false;
	for (int i = 0; i < N_THREAD; i++) {
		mThreadRun[i] = false;
		mThreadWait[i] = false;
		mThreadFinish[i] = false;
		mThreads.clear();
	}
}

void Matrix::threadFunction(int threadId) {
	if (CPU) {
		if (DEBUG_M) cout << "Initialize thread" << threadId << endl;
		while (mThreadRun[threadId]) {
			if (mThreadStart[threadId]) {
				mNumberLock.lock();
				runThreads++;
				mNumberLock.unlock();

				if (DEBUG_M) cout << "Start thread" << threadId << endl;
				dimension divisionDim;
				int start, end, range;
				if (threadM1->row() > threadM1->col() && mMode != COL_SWAP || mMode == ROW_SWAP) {
					divisionDim = ROW;
					range = (threadM1->row() / N_THREAD + 1);
					start = threadId * range >= threadM1->row() ? threadM1->row() : threadId * range;
					end = (threadId + 1) * range >= threadM1->row() ? threadM1->row() : (threadId + 1) * range;
				}
				else {
					divisionDim = COL;
					if (mMode == MULT) {
						range = (threadM2->col() / N_THREAD + 1);
						start = threadId * range >= threadM2->col() ? threadM2->col() : threadId * range;
						end = (threadId + 1) * range >= threadM2->col() ? threadM2->col() : (threadId + 1) * range;
					}
					else {
						range = (threadM1->col() / N_THREAD + 1);
						start = threadId * range >= threadM1->col() ? threadM1->col() : threadId * range;
						end = (threadId + 1) * range >= threadM1->col() ? threadM1->col() : (threadId + 1) * range;
					}
				}
				int arg1, arg2;

				// Calculation
				switch (mMode) {
				case SUM:
				case SUM_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] + (*threadM2)[i][j];
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] + (*threadM2)[i][j];
							}
						}
					}
					break;

				case SCAL_SUM:
				case SCAL_SUM_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] + threadScal;
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] + threadScal;
							}
						}
					}
					break;

				case SUB:
				case SUB_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] - (*threadM2)[i][j];
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] - (*threadM2)[i][j];
							}
						}
					}
					break;

				case SCAL_SUB:
				case SCAL_SUB_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] - threadScal;
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] - threadScal;
							}
						}
					}
					break;

				case MULT:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM2->col(); j++) {
								(*threadDestination)[i][j] = 0;
								for (int k = 0; k < threadM1->col(); k++) {
									(*threadDestination)[i][j] += (*threadM1)[i][k] * (*threadM2)[k][j];
								}
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = 0;
								for (int k = 0; k < threadM1->col(); k++) {
									(*threadDestination)[i][j] += (*threadM1)[i][k] * (*threadM2)[k][j];
								}
							}
						}
					}
					break;

				case ELT_MULT:
				case ELT_MULT_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] * (*threadM2)[i][j];
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] * (*threadM2)[i][j];
							}
						}
					}
					break;

				case SCAL_MULT:
				case SCAL_MULT_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] * threadScal;
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] * threadScal;
							}
						}
					}
					break;

				case ELT_DIVIDE:
				case ELT_DIVIDE_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] / (*threadM2)[i][j];
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] / (*threadM2)[i][j];
							}
						}
					}
					break;

				case SCAL_DIVIDE:
				case SCAL_DIVIDE_ASSIGN:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] / threadScal;
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j] / threadScal;
							}
						}
					}
					break;

				case SET:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = threadScal;
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = threadScal;
							}
						}
					}
					break;

				case COPY:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j];
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = (*threadM1)[i][j];
							}
						}
					}
					break;

				case SCAL_POW:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[i][j] = pow((*threadM1)[i][j], threadScal);
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[i][j] = pow((*threadM1)[i][j], threadScal);
							}
						}
					}
					break;

				case TRANS:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[j][i] = (*threadM1)[i][j];
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[j][i] = (*threadM1)[i][j];
							}
						}
					}
					break;

				case ROW_SWAP:
					arg1 = *(int*)threadArg1;
					arg2 = *(int*)threadArg2;
					dtype temp;
					for (int i = start; i < end; i++) {
						temp = (*threadDestination)[i][arg1];
						(*threadDestination)[i][arg1] = (*threadDestination)[i][arg2];
						(*threadDestination)[i][arg2] = temp;
					}
					break;

				case COL_SWAP:
					arg1 = *(int*)threadArg1;
					arg2 = *(int*)threadArg2;
					for (int j = start; j < end; j++) {
						temp = (*threadDestination)[arg1][j];
						(*threadDestination)[arg1][j] = (*threadDestination)[arg2][j];
						(*threadDestination)[arg2][j] = temp;
					}
					break;

				case SIGMOID:
					if (divisionDim == ROW) {
						for (int i = start; i < end; i++) {
							for (int j = 0; j < threadM1->col(); j++) {
								(*threadDestination)[j][i] = 1 / (1 + exp(-(*threadDestination)[i][j]));
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row(); i++) {
							for (int j = start; j < end; j++) {
								(*threadDestination)[j][i] = 1 / (1 + exp(-(*threadDestination)[i][j]));
							}
						}
					}
					break;

				case NOT_EQUAL:
					threadBool = false;
					if (divisionDim == ROW) {
						for (int i = start; i < end && threadBool == false; i++) {
							for (int j = 0; j < threadM1->col() && threadBool == false; j++) {
								if ((*threadM1)[i][j] != (*threadM2)[i][j])
									threadBool = true;
							}
						}
					}
					else {
						for (int i = 0; i < threadM1->row() && threadBool == false; i++) {
							for (int j = start; j < end && threadBool == false; j++) {
								if ((*threadM1)[i][j] != (*threadM2)[i][j])
									threadBool = true;
							}
						}
					}
					break;
				}

				// Wait the other threads
				if (DEBUG_M) cout << "Wait thread" << threadId << endl;
				mThreadWait[threadId] = true;
				bool wait = true;
				while (wait) {
					wait = false;
					for (int i = 0; i < N_THREAD; i++)
						if (mThreadWait[i] == false) {
							wait = true;
							break;
						}
				}
				mThreadStart[threadId] = false;
				mThreadFinish[threadId] = true;
				if (DEBUG_M) cout << "Finish thread" << threadId << endl;
			}
		}

		if (DEBUG_M) cout << "End matrix thread" << threadId << endl;
	}
}

// Matrix multiplication with tiling
__global__ void matMultKernel(dtype* tOut, dtype* t1, dtype* t2,
	const int t1_row, const int t1_col, const int t2_row, const int t2_col) {
	// Use shared memory for speeding up
	__shared__ dtype s_m[BLOCK_DIM][BLOCK_DIM];
	__shared__ dtype s_n[BLOCK_DIM][BLOCK_DIM];

	register int thread_row = threadIdx.y, thread_col = threadIdx.x;
	register int block_row = blockIdx.y,  block_col = blockIdx.x;

	register int row = block_row * blockDim.y + thread_row;
	register int col = block_col * blockDim.x + thread_col;

	register int t1_c, t2_r;
	register dtype pValue = 0;

	if (row >= t1_row && col >= t2_col)
		return;

	for (register int m = 0; m < t1_col / BLOCK_DIM + 1; m++) {
		t1_c = m * BLOCK_DIM + thread_col;
		t2_r = m * BLOCK_DIM + thread_row;
		s_m[thread_row][thread_col] = row < t1_row && t1_c < t1_col ? t1[row * t1_col + t1_c] : 0;
		s_n[thread_row][thread_col] = t2_r < t2_row && col < t2_col ? t2[t2_r * t2_col + col] : 0;
	
		__syncthreads();

#pragma unroll
		for (int k = 0; k < BLOCK_DIM; k++)
			pValue += s_m[thread_row][k] * s_n[k][thread_col];

		__syncthreads();
	}

	if (row >= t1_row || col >= t2_col) 
		return;

	tOut[row * t2_col + col] = pValue;
}

// Matrix multiplication with tiling with transpose
__global__ void matMultTrans1Kernel(dtype* tOut, dtype* t1, dtype* t2,
	const int t1_row, const int t1_col, const int t2_row, const int t2_col) {
	// Use shared memory for speeding up
	__shared__ dtype s_m[BLOCK_DIM][BLOCK_DIM];
	__shared__ dtype s_n[BLOCK_DIM][BLOCK_DIM];

	register int thread_row = threadIdx.y, thread_col = threadIdx.x;
	register int block_row = blockIdx.y, block_col = blockIdx.x;

	register int row = block_row * blockDim.y + thread_row;
	register int col = block_col * blockDim.x + thread_col;

	register int t1_r, t2_r;
	register dtype pValue = 0;

	if (row >= t1_row && col >= t2_col)
		return;

	for (register int m = 0; m < t1_row / BLOCK_DIM + 1; ++m) {
		t1_r = m * BLOCK_DIM + thread_col;
		t2_r = m * BLOCK_DIM + thread_row;
		s_m[thread_row][thread_col] = t1_r < t1_row && row < t1_col ? t1[t1_r * t1_col + row] : 0;
		s_n[thread_row][thread_col] = t2_r < t2_row && col < t2_col ? t2[t2_r * t2_col + col] : 0;
		__syncthreads();
#pragma unroll
		for (int k = 0; k < BLOCK_DIM; k++)
			pValue += s_m[thread_row][k] * s_n[k][thread_col];

		__syncthreads();
	}

	if (row >= t1_col || col >= t2_col)
		return;

	tOut[row * t2_col + col] = pValue;
}

// Matrix multiplication with tiling with transpose
__global__ void matMultTrans2Kernel(dtype* tOut, dtype* t1, dtype* t2,
	const int t1_row, const int t1_col, const int t2_row, const int t2_col) {
	// Use shared memory for speeding up
	__shared__ dtype s_m[BLOCK_DIM][BLOCK_DIM];
	__shared__ dtype s_n[BLOCK_DIM][BLOCK_DIM];

	register int thread_row = threadIdx.y, thread_col = threadIdx.x;
	register int block_row = blockIdx.y, block_col = blockIdx.x;

	register int row = block_row * blockDim.y + thread_row;
	register int col = block_col * blockDim.x + thread_col;

	register int t1_c, t2_c;
	register dtype pValue = 0;

	if (row >= t1_row && col >= t2_col)
		return;

	for (register int m = 0; m < t1_col / BLOCK_DIM + 1; ++m) {
		t1_c = m * BLOCK_DIM + thread_col;
		t2_c = m * BLOCK_DIM + thread_row;
		s_m[thread_row][thread_col] = row < t1_row && t1_c < t1_col ? t1[row * t1_col + t1_c] : 0;
		s_n[thread_row][thread_col] = col < t2_row && t2_c < t2_col ? t2[col * t2_col + t2_c] : 0;
		__syncthreads();
#pragma unroll
		for (int k = 0; k < BLOCK_DIM; k++)
			pValue += s_m[thread_row][k] * s_n[k][thread_col];

		__syncthreads();
	}

	if (row >= t1_row || col >= t2_row)
		return;

	tOut[row * t2_row + col] = pValue;
}

// Matrix multiplication with tiling with transpose
__global__ void matMultTrans12Kernel(dtype* tOut, dtype* t1, dtype* t2,
	const int t1_row, const int t1_col, const int t2_row, const int t2_col) {
	// Use shared memory for speeding up
	__shared__ dtype s_m[BLOCK_DIM][BLOCK_DIM];
	__shared__ dtype s_n[BLOCK_DIM][BLOCK_DIM];

	register int thread_row = threadIdx.y, thread_col = threadIdx.x;
	register int block_row = blockIdx.y, block_col = blockIdx.x;

	register int row = block_row * blockDim.y + thread_row;
	register int col = block_col * blockDim.x + thread_col;

	register int t1_r, t2_c;
	register dtype pValue = 0;

	if (row >= t1_row && col >= t2_col)
		return;

	for (register int m = 0; m < t1_row / BLOCK_DIM + 1; ++m) {
		t1_r = m * BLOCK_DIM + thread_col; 
		t2_c = m * BLOCK_DIM + thread_row;
		s_m[thread_row][thread_col] = t1_r < t1_row && row < t1_col ? t1[t1_r * t1_col + row] : 0;
		s_n[thread_row][thread_col] = col < t2_row && t2_c < t2_col ? t2[col * t2_col + t2_c] : 0;
		__syncthreads();
#pragma unroll
		for (int k = 0; k < BLOCK_DIM; k++)
			pValue += s_m[thread_row][k] * s_n[k][thread_col];

		__syncthreads();
	}

	if (row >= t1_col || col >= t2_row)
		return;

	tOut[row * t2_row + col] = pValue;
}

__global__ void scalarMultKernel(dtype* t_out, dtype* t1, dtype scalar, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = scalar * t1[i];
}

__global__ void scalarMultAssignKernel(dtype* t_out, dtype scalar, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] *= scalar;
}

__global__ void addKernel(dtype* t_out, dtype* t1, dtype* t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] + t2[i];
}

__global__ void addAssignKernel(dtype* t_out, dtype* t1, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] += t1[i];
}

__global__ void scalAddKernel(dtype* t_out, dtype* t1, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] + t2;
}

__global__ void scalAddAssignKernel(dtype* t_out, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] += t2;
}

__global__ void subKernel(dtype* t_out, dtype* t1, dtype* t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] - t2[i];
}

__global__ void subAssignKernel(dtype* t_out, dtype* t1, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] -= t1[i];
}

__global__ void scalSubKernel(dtype* t_out, dtype* t1, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] - t2;
}

__global__ void scalSubAssignKernel(dtype* t_out, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] -= t2;
}

__global__ void multiplyKernel(dtype* t_out, dtype* t1, dtype* t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] * t2[i];
}

__global__ void multiplyAssignKernel(dtype* t_out, dtype* t1, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] *= t1[i];
}

__global__ void divideKernel(dtype* t_out, dtype* t1, dtype* t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] / t2[i];
}

__global__ void divideAssignKernel(dtype* t_out, dtype* t1, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] /= t1[i];
}

__global__ void scalDivideKernel(dtype* t_out, dtype* t1, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1[i] / t2;
}

__global__ void scalDivideAssignKernel(dtype* t_out, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] /= t2;
}

__global__ void setKernel(dtype* t_out, dtype* t1, int tOutRow, int tOutCol, int t1Row, int t1Col, int rowOffset, int colOffset) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowOffset == 0 && colOffset == 0) {
		if (i < tOutRow * tOutCol)
			t_out[i] = t1[i];
	}
	else {
		int r = i / tOutCol;
		int c = i % tOutCol;
		t_out[i] = t1[i];
	}
}

__global__ void scalSetKernel(dtype* t_out, dtype t1, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = t1;
}

__global__ void scalPowKernel(dtype* t_out, dtype* t1, dtype t2, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = pow(t1[i], t2);
}

__global__ void transKernel(dtype* t_out, dtype* t1, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[(int)(i / row) + (i % row) * col] = t1[i];
}

__global__ void swapKernel(dtype* t_out, bool swapRow, int row, int col, int arg1, int arg2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (swapRow) {
		if (i < row) {
			dtype temp = t_out[arg1 * col + i];
			t_out[arg1 * col + i] = t_out[arg2 * col + i];
			t_out[arg2 * col + i] = temp;
		}
	}
	else {
		if (i < col) {
			dtype temp = t_out[i * col + arg1];
			t_out[i * col + arg1] = t_out[i * col + arg2];
			t_out[i * col + arg2] = temp;
		}
	}
}

__global__ void sigmoidKernel(dtype* t_out, int row, int col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		t_out[i] = 1.0 / (1.0 + exp(-t_out[i]));
}

__global__ void notEqualKernel(dtype* t1, dtype* t2, int row, int col, bool* result) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < row * col)
		if (t1[i] != t2[i])
			*result = true;
}

__global__ void reset(bool* result) {
	*result = false;
}


void Matrix::startThreads() {
	// Start Threads
	for (int i = 0; i < N_THREAD; i++)
		mThreadStart[i] = true;

	// Wait for calculation finish
	bool wait = true;
	while (wait) {
		wait = false;
		for (int i = 0; i < N_THREAD; i++)
			if (mThreadFinish[i] == false) {
				wait = true;
				break;
			}
	}
	for (int i = 0; i < N_THREAD; i++) {
		mThreadFinish[i] = false;
		mThreadWait[i] = false;
	}

	if (DEBUG_M) cout << "End calculation" << endl;
}

void Matrix::startGPUCalc() {
	allocDevData();
	mGpuIsChanged = true;
	if (threadDestination != NULL) {
		threadDestination->allocDevData();
		threadDestination->mGpuIsChanged = true;
	}

	if (threadM1 != NULL) {
		threadM1->allocDevData();
		threadM1->mGpuIsChanged = true;
	}

	if (threadM2 != NULL) {
		threadM2->allocDevData();
		threadM2->mGpuIsChanged = true;
	}

	bool transM1 = threadArg1 == NULL ? false : *(bool*)threadArg1;
	bool transM2 = threadArg2 == NULL ? false : *(bool*)threadArg2;
	dim3 dimBlock;
	dim3 dimGrid;
	switch (mMode) {
	case SUM:
	case SUM_ASSIGN:
	case SCAL_SUM:
	case SCAL_SUM_ASSIGN:
	case SUB:
	case SUB_ASSIGN:
	case SCAL_SUB:
	case SCAL_SUB_ASSIGN:
	case ELT_MULT:
	case ELT_MULT_ASSIGN:
	case SCAL_MULT:
	case SCAL_MULT_ASSIGN:
	case ELT_DIVIDE:
	case ELT_DIVIDE_ASSIGN:
	case SCAL_DIVIDE:
	case SCAL_DIVIDE_ASSIGN:
	case SET:
	case COPY:
	case SCAL_POW:
	case TRANS:
	case SIGMOID:
		dimBlock.x = BLOCK_DIM * BLOCK_DIM;
		dimBlock.y = 1;
		dimGrid.x = threadDestination->size() / dimBlock.x + 1;
		dimGrid.y = 1;
		break;

	case NOT_EQUAL:
		dimBlock.x = BLOCK_DIM * BLOCK_DIM;
		dimBlock.y = 1;
		dimGrid.x = threadM1->size() / dimBlock.x + 1;
		dimGrid.y = 1;
		break;

	case MULT:
		dimBlock.x = BLOCK_DIM;
		dimBlock.y = BLOCK_DIM;
		dimGrid.x = ((transM2 ? threadM2->row() : threadM2->col()) + BLOCK_DIM - 1) / dimBlock.x;
		dimGrid.y = ((transM1 ? threadM1->col() : threadM1->row()) + BLOCK_DIM - 1) / dimBlock.y;
		break;

	case ROW_SWAP:
		dimBlock.x = BLOCK_DIM * BLOCK_DIM;
		dimBlock.y = 1;
		dimGrid.x = threadDestination->row() / dimBlock.x + 1;
		dimGrid.y = 1;
		break;

	case COL_SWAP:
		dimBlock.x = BLOCK_DIM * BLOCK_DIM;
		dimBlock.y = 1;
		dimGrid.x = threadDestination->col() / dimBlock.x + 1;
		dimGrid.y = 1;
		break;
	}

	float alpha = 1;
	float beta = 0;

	// Calculation
	switch (mMode) {
	case SUM:
		addKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		break;

	case SUM_ASSIGN:
		addAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		break;

	case SCAL_SUM:
		scalAddKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		break;

	case SCAL_SUM_ASSIGN:
		scalAddAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		break;

	case SUB:
		subKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		break;

	case SUB_ASSIGN:
		subAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		break;

	case SCAL_SUB:
		scalSubKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		break;

	case SCAL_SUB_ASSIGN:
		scalSubAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		break;

	case MULT:
		if (!transM1 && !transM2) {
			matMultKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData,
				threadM1->row(), threadM1->col(), threadM2->row(), threadM2->col());
		}
		else if (transM1 && !transM2) {
			matMultTrans1Kernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData,
				threadM1->row(), threadM1->col(), threadM2->row(), threadM2->col());
		}
		else if (!transM1 && transM2) {
			matMultTrans2Kernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData,
				threadM1->row(), threadM1->col(), threadM2->row(), threadM2->col());
		}
		else if (transM1 && transM2) {
			matMultTrans12Kernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData,
				threadM1->row(), threadM1->col(), threadM2->row(), threadM2->col());
		}
		break;

	case ELT_MULT:
		multiplyKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case ELT_MULT_ASSIGN:
		multiplyAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case SCAL_MULT:
		scalarMultKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case SCAL_MULT_ASSIGN:
		scalarMultAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case ELT_DIVIDE:
		divideKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case ELT_DIVIDE_ASSIGN:
		divideAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM2->mDevData, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case SCAL_DIVIDE:
		scalDivideKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case SCAL_DIVIDE_ASSIGN:
		scalDivideAssignKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case SET:
		scalSetKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case COPY:
		setKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadDestination->row(), threadDestination->col(), 0, 0, 0, 0);
		checkCudaError();
		break;

	case SCAL_POW:
		scalPowKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadScal, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case TRANS:
		transKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, threadM1->mDevData, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case ROW_SWAP:
		swapKernel << <dimGrid, dimBlock>> > (threadDestination->mDevData, true, threadDestination->row(), threadDestination->col(), *(int*)threadArg1, *(int*)threadArg2);
		checkCudaError();
		break;

	case COL_SWAP:
		swapKernel << < dimGrid, dimBlock>> >(threadDestination->mDevData, false, threadDestination->row(), threadDestination->col(), *(int*)threadArg1, *(int*)threadArg2);
		checkCudaError();
		break;

	case SIGMOID:
		sigmoidKernel << < dimGrid, dimBlock>> >(threadDestination->mDevData, threadDestination->row(), threadDestination->col());
		checkCudaError();
		break;

	case NOT_EQUAL:
		reset << < 1, 1 >> > (gpuBool);
		notEqualKernel << < dimGrid, dimBlock>> >(threadM1->mDevData, threadM2->mDevData, threadM1->row(), threadM2->col(), gpuBool);
		cudaMemcpy(&threadBool, gpuBool, sizeof(bool), cudaMemcpyDeviceToHost);
		checkCudaError();
		break;
	}

	checkCudaError();
}

void Matrix::calculation(threadMode aMode, Matrix* m1, Matrix* m2, Matrix* m_des, void* arg1, void* arg2) {
	threadM1 = m1;
	threadM2 = m2;
	threadDestination = m_des;
	mMode = aMode;

	threadArg1 = arg1;
	threadArg2 = arg2;

	if (m_des == &threadResult) {
		if (mMode == MULT) {
			bool transM1 = *(bool*)arg1;
			bool transM2 = *(bool*)arg2;
			int rowM1 = transM1 ? m1->col() : m1->row();
			int colM2 = transM2 ? m2->row() : m2->col();

			if (rowM1 != threadResult.row() || colM2 != threadResult.col())
				threadResult.reshape(rowM1, colM2);
		}
		else {
			if (m1->row() != threadResult.row() || m1->col() != threadResult.col())
				threadResult.reshape(m1->row(), m1->col());
		}
	}

	if (CPU) startThreads();
	else if (GPU) startGPUCalc();
}

void Matrix::calculation(threadMode aMode, Matrix* m1, dtype scal, Matrix* m_des, void* arg1, void* arg2) {
	threadM1 = m1;
	threadM2 = NULL;
	threadScal = scal;
	threadDestination = m_des;
	mMode = aMode;

	if (m_des == &threadResult)
		threadResult.reshape(m1->row(), m1->col());

	threadArg1 = arg1;
	threadArg2 = arg2;

	if (CPU) startThreads();
	else if (GPU) startGPUCalc();
}

void Matrix::allocHostData() {
	mData = new dtype[mRow * mCol];
	memset(mData, 0, sizeof(dtype) * mRow * mCol);
	mCpuIsChanged = true;
	mHostAllocated = true;
}

void Matrix::freeHostData() {
	if (mHostAllocated) {
		delete[] mData;
		mCpuIsChanged = false;
		mHostAllocated = false;
	}
}

void Matrix::freeDevData() {
	if (GPU && mDevAllocated) {
		cudaFree(mDevData);
		checkCudaError();
		mGpuIsChanged = false;
		mDevAllocated = false;
	}
}

void Matrix::operator=(Matrix& other) {
	if(other.row() != row() || other.col() != col())
		reshape(other.mRow, other.mCol);

	calculation(COPY, &other, nullptr, this);
}

Matrix Matrix::matMult(Matrix* a, Matrix* b, Matrix* des, bool transA, bool transB) {
	int aRow = transA ? a->col() : a->row();
	int aCol = transA ? a->row() : a->col();
	int bRow = transB ? b->col() : b->row();
	int bCol = transB ? b->row() : b->col();
	if (aCol != bRow) {
		cout << "Cannot multiply matrix" << endl;
		exit(EXIT_FAILURE);
	}

	calculation(MULT, a, b, des == NULL ? &threadResult : des, (void*)&transA, (void*)&transB);

	return threadResult;
}

Matrix Matrix::matMult(dtype a, Matrix* b, Matrix* des) {

	calculation(SCAL_MULT, b, a, des == NULL ? &threadResult : des);

	return threadResult;
}

Matrix Matrix::matTranspose(Matrix* a) {
	Matrix result(*a);

	result.reshape(a->col(), a->row());

	calculation(TRANS, a, nullptr, &result);

	return result;
}

Matrix Matrix::matMean(Matrix* a, dimension d) {
	int row, col;
	switch (d) {
	case ROW:
		row = 1;
		col = a->col();
		break;

	case COL:
		row = a->row();
		col = 1;
		break;
	}

	Matrix mean(row, col);

	switch (d) {
	case ROW:
		for (int i = 0; i < a->col(); i++) {
			for (int j = 0; j < a->row(); j++)
				mean[0][i] += (*a)[j][i];
			mean[0][i] /= a->row();
		}
		break;

	case COL:
		for (int i = 0; i < a->row(); i++) {
			for (int j = 0; j < a->col(); j++)
				mean[i][0] += (*a)[i][j];
			mean[i][0] /= a->col();
		}
		break;
	}

	return mean;
}

Matrix Matrix::matCov(Matrix* a, dimension d) {
	Matrix mean = matMean(a, ROW);
	Matrix D(a->row(), a->col());
	for (int i = 0; i < a->row(); i++) {
		for (int j = 0; j < a->col(); j++) {
			D[i][j] = (*a)[i][j] - mean[0][j];
		}
	}
	Matrix cov = matMult(1 / dtype(a->row() - 1), &matMult(&matTranspose(&D), &D));

	return cov;
}

Matrix Matrix::matAdd(Matrix* a, Matrix* b) {
	if (a->row() != b->row() || a->col() != b->col()) {
		cout << "Cannot add matrix" << endl;
		exit(EXIT_FAILURE);
	}

	calculation(SUM, a, b);

	return threadResult;
}

Matrix Matrix::matSub(Matrix* a, Matrix* b) {
	if (a->row() != b->row() || a->col() != b->col()) {
		cout << "Cannot subtract matrix" << endl;
		exit(EXIT_FAILURE);
	}

	calculation(SUB, a, b);

	return threadResult;
}

dtype Matrix::matSum(Matrix* a) {
	dtype result = 0;

	for (int i = 0; i < a->row(); i++)
		for (int j = 0; j < a->col(); j++)
			result += (*a)[i][j];

	return result;
}

dtype Matrix::matDet(Matrix* a) {
	if (a->row() != a->col()) {
		cout << "Cannot get determinant" << endl;
		exit(EXIT_FAILURE);
	}

	dtype result = 0;

	if (a->row() == 2 && a->row() == a->col())
		return ((*a)[0][0] * (*a)[1][1] - (*a)[0][1] * (*a)[1][0]);

	for (int i = 0; i < a->col(); i++) {
		Matrix temp(a->row() - 1, a->col() - 1);
		int skip = 0;
		for (int j = 0; j < temp.row(); j++) {
			for (int k = 0; k < temp.col(); k++) {
				if (k < i) skip = 0;
				else skip = 1;
				temp[j][k] = (*a)[j + 1][k + skip];
			}
		}
		result += powf(-1, i) * (*a)[0][i] * matDet(&temp);
	}

	return result;
}

Matrix Matrix::matInverse(Matrix* a) {
	if (a->row() != a->col() || a->row() < 2) {
		cout << "Cannot get inverse" << endl;
		exit(EXIT_FAILURE);
	}

	Matrix result(a->row(), a->col());

	if (a->row() == 2) {
		result[0][0] = (*a)[1][1];
		result[1][1] = (*a)[0][0];
		result[0][1] = -(*a)[0][1];
		result[1][0] = -(*a)[1][0];
		return matMult(matDet(a), &result);
	}

	for (int i = 0; i < a->row(); i++) {
		for (int j = 0; j < a->col(); j++) {
			Matrix temp(a->row() - 1, a->col() - 1);
			for (int k = 0; k < temp.row(); k++)
				for (int l = 0; l < temp.col(); l++)
					temp[k][l] = (*a)[k + (k < i ? 0 : 1)][l + (l < j ? 0 : 1)];

			result[i][j] = matDet(&temp);
		}
	}

	result = matTranspose(&result);

	for (int i = 0; i < result.row(); i++)
		for (int j = 0; j < result.col(); j++)
			result[i][j] *= powf(-1, i + j);


	result = matMult(1 / matDet(a), &result);

	return result;
}

dtype zeroCheck(dtype value) {
	return (value > -ZERO && value < ZERO) ? 0 : value;
}

Matrix Matrix::matInverse2(Matrix* a) {
	if (a->row() != a->col()) {
		cout << "Cannot get inverse" << endl;
		exit(EXIT_FAILURE);
	}
	Matrix concat(a->row(), a->col() * 2);
	Matrix result(a->row(), a->col());

	for (int i = 0; i < a->row(); i++)
		for (int j = 0; j < a->col(); j++)
			concat[i][j] = (*a)[i][j];

	for (int i = 0; i < a->row(); i++)
		concat[i][i + a->col()] = 1;

	int first = -1;
	for (int i = 0; i < concat.row(); i++)
		if (concat[i][0] != 0) {
			first = i;
			break;
		}

	if (first == -1) {
		cout << "Cannget get inverse, insufficient rank\n";
		exit(EXIT_FAILURE);
	}

	if (first != -1) concat.rowSwap(0, first);

	// Row reduction
	for (int r = 0; r < concat.row() - 1; r++) {
		if (concat[r + 1][r] != 0) {
			for (int r2 = r + 1; r2 < concat.row(); r2++) {
				dtype coe = -concat[r2][r] / concat[r][r];

				for (int c = 0; c < a->col(); c++) {
					concat[r2][c] += zeroCheck(concat[r][c] * coe);
					concat[r2][c + a->col()] += zeroCheck(concat[r][c + a->col()] * coe);
				}
			}
			int swapIndex = r + 1;

			for (int r3 = r + 1; r3 < concat.row(); r3++)
				if (concat[r3][r + 1] != 0 && concat[r3][r + 1] > concat[swapIndex][r + 1]) {
					swapIndex = r3;
					break;
				}

			concat.rowSwap(r + 1, swapIndex);
		}
	}

	int rank = 0;
	for (int i = 0; i < concat.row(); i++)
		if (zeroCheck(concat[i][i]) != 0) rank++;

	if (rank != concat.row()) {
		cout << "Cannot inverse this matrix, insufficient rank\n";
		exit(EXIT_FAILURE);
	}

	// Make diagonal 1
	for (int r = 0; r < concat.row(); r++) {
		if (concat[r][r] != 1) {
			dtype coe = 1.0 / concat[r][r];

			for (int c = 0; c < concat.col(); c++)
				concat[r][c] *= coe;
		}
	}

	// Make to identity matrix
	for (int r = concat.row() - 2; r >= 0; r--) {
		for (int r2 = r + 1; r2 < concat.row(); r2++) {
			if (concat[r][r2] != 0) {
				dtype coe = -concat[r][r2] / concat[r2][r2];

				for (int c = r + 1; c < concat.col(); c++)
					concat[r][c] += concat[r2][c] * coe;
			}
		}
	}

	// Copy to result
	for (int r = 0; r < a->row(); r++) {
		for (int c = 0; c < a->col(); c++) {
			result[r][c] = concat[r][c + a->col()];
		}
	}

	return result;
}

Matrix Matrix::matPseudoInverse(Matrix* a) {
	Matrix result(a->row(), a->row());
	
	result = ((*a).T() * (*a)).I() * (*a).T();

	return result;
}

void Matrix::rowSwap(int index1, int index2) {
	if (index1 == index2)
		return;

	calculation(ROW_SWAP, this, 0.0, this, (void*)&index1, (void*)&index2);
}

void Matrix::colSwap(int index1, int index2) {
	if (index1 == index2)
		return;

	calculation(COL_SWAP, this, 0.0, this, (void*)&index1, (void*)&index2);
}

void Matrix::matAbs() {
	for (int i = 0; i < row(); i++)
		for (int j = 0; j < col(); j++)
			(*this)[i][j] = abs((*this)[i][j]);
}

void Matrix::operator()(int value) {
	calculation(SET, this, value, this);
}

void Matrix::operator()(initializer_list<dtype> values, bool repeat) {
	if (repeat) {
		for (int i = 0; i < row(); i++)
			for (int j = 0; j < col(); j++)
				(*this)[i][j] = *(values.begin() + (i * row() + j) % values.size());
	}
	else {
		for (int i = 0; i < values.size(); i++)
			(*this)[i / col()][i % col()] = *(values.begin() + i);
	}
}

Matrix Matrix::operator+(Matrix& other) {
	return matAdd(this, &other);
}

Matrix Matrix::operator-(Matrix& other) {
	return matSub(this, &other);
}

bool Matrix::operator==(Matrix& other) {
	if (this->row() != other.row() || this->col() != other.col()) {
		cout << "Cannot compare matrix" << endl;
		exit(EXIT_FAILURE);
	}

	calculation(NOT_EQUAL, this, &other, NULL);
	return !threadBool;
}

bool Matrix::operator!=(Matrix& other) {
	if (this->row() != other.row() || this->col() != other.col()) {
		cout << "Cannot compare matrix" << endl;
		exit(EXIT_FAILURE);
	}

	calculation(NOT_EQUAL, this, &other, NULL);
	return threadBool;
}

void Matrix::operator-=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot subtract matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(SUB_ASSIGN, this, &other, this);
}

void Matrix::operator+=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot add matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(SUM_ASSIGN, this, &other, this);
}

void Matrix::operator-=(dtype other) {
	calculation(SCAL_SUB_ASSIGN, this, other, this);
}

void Matrix::operator+=(dtype other) {
	calculation(SCAL_SUM_ASSIGN, this, other, this);
}

void Matrix::operator/=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot divide matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(ELT_DIVIDE_ASSIGN, this, &other, this);
}

void Matrix::operator/=(dtype other) {
	calculation(SCAL_DIVIDE_ASSIGN, this, other, this);
}

// Element wise multiplication
void Matrix::operator*=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot multiply matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(ELT_MULT_ASSIGN, this, &other, this);
}

void Matrix::operator*=(dtype other) {
	calculation(SCAL_MULT_ASSIGN, this, other, this);
}

dtype Matrix::operator+(dtype other) {
	if (this->row() * this->col() == 1)
		return (*this)[0][0] + other;
	else {
		cout << "Cannot add dtype and matrix, matrix size must be 1" << endl;
		exit(EXIT_FAILURE);
	}
}

dtype Matrix::operator-(dtype other) {
	if (this->row() * this->col() == 1)
		return (*this)[0][0] - other;
	else {
		cout << "Cannot add dtype and matrix, matrix size must be 1" << endl;
		exit(EXIT_FAILURE);
	}
}

Matrix Matrix::operator^(dtype coefficient) {
	calculation(SCAL_POW, this, coefficient);
	return threadResult;
}

// Matrix multiplication
Matrix Matrix::operator*(Matrix& other) {
	return matMult(this, &other);
}

Matrix Matrix::operator*(dtype other) {
	return matMult(other, this);
}

Matrix Matrix::operator/(Matrix& other) {
	calculation(ELT_DIVIDE, this, &other);

	return threadResult;
}

Matrix Matrix::operator/(dtype other) {
	calculation(SCAL_DIVIDE, this, other);

	return threadResult;
}

Matrix Matrix::cov(dimension d) {
	return matCov(this, d);
}

Matrix Matrix::mean(dimension d) {
	return matMean(this, d);
}

Matrix Matrix::T() {
	return matTranspose(this);
}

Matrix Matrix::I() {
	matInverse2(this);
	return *this;
}

Matrix Matrix::PI() {
	matPseudoInverse(this);
	return *this;
}

dtype Matrix::det() {
	return matDet(this);
}

dtype Matrix::sum() {
	return matSum(this);
}

Matrix Matrix::rowSum() {
	Matrix result(1, this->col());
	for (int i = 0; i < row(); i++)
		for (int j = 0; j < col(); j++)
			result[0][j] += (*this)[i][j];

	return result;
}

Matrix Matrix::colSum() {
	Matrix result(this->row(), 1);
	for (int i = 0; i < row(); i++)
		for (int j = 0; j < col(); j++)
			result[i][0] += (*this)[i][j];

	return result;
}

Matrix Matrix::repeat(int row, int col) {
	if (row < 1 || col < 1) {
		cout << "Row or Col argument must be larger than 1\n";
		exit(EXIT_FAILURE);
	}
	Matrix result(this->row() * row, this->col() * col);

	for (int i = 0; i < result.row(); i++)
		for (int j = 0; j < result.col(); j++)
			result[i][j] = (*this)[i % this->row()][j % this->col()];

	return result;
}

// Data will be swiped out
void Matrix::reshape(int aRow, int aCol) {
	if (row() != aRow || col() != aCol) {
		freeHostData();
		freeDevData();
		mRow = aRow;
		mCol = aCol;
		allocHostData();
		allocDevData();
	}
}

void Matrix::print() {
	printf("[%d x %d] Matrix\n", mRow, mCol);
	for (int i = 0; i < row(); i++) {
		for (int j = 0; j < col(); j++) {
			printf("%8.3f\t", (*this)[i][j]);
		}
		cout << endl;
	}
}

void Matrix::copyFrom(Matrix& source, int srcRowOffset, int srcColOffset) {
	for (int i = 0; i < mRow; i++)
		for (int j = 0; j < mCol; j++)
			(*this)[i][j] = source[i + srcRowOffset][j + srcColOffset];
}

void Matrix::copyTo(Matrix& destination, int dstRowOffset, int dstColOffset) {
	for (int i = 0; i < mRow; i++)
		for (int j = 0; j < mCol; j++)
			destination[i + dstRowOffset][j + dstColOffset] = (*this)[i][j];
}

Matrix Matrix::split(dimension dimen, int aStart, int aSize) {

	Matrix result;

	switch (dimen)
	{
	case ROW:
		result.reshape(aSize, this->col());
		result.copyFrom(*this, aStart);
		break;

	case COL:
		result.reshape(this->row(), aSize);
		result.copyFrom(*this, 0, aStart);
		break;

	default:
		break;
	}

	return result;
}

Matrix Matrix::concat(dimension dimen, Matrix& other) {
	if ((dimen == ROW && this->col() != other.col()) || (dimen == COL && this->row() != other.row())) {
		cout << "Cannot concatenate two matrices\n";
		exit(EXIT_FAILURE);
	}

	Matrix result;
	
	switch (dimen)
	{
	case ROW:
		result.reshape(this->row() + other.row(), this->col());
		for (int i = 0; i < this->row(); i++)
			for (int j = 0; j < result.col(); j++)
				result[i][j] = (*this)[i][j];

		for (int i = 0; i < other.row(); i++)
			for (int j = 0; j < result.col(); j++) {
				result[this->row() + i][j] = other[i][j];
			}
		break;

	case COL:
		result.reshape(this->row(), this->col() + other.col());
		for (int i = 0; i < result.row(); i++)
			for (int j = 0; j < this->col(); j++)
				result[i][j] = (*this)[i][j];

		for (int i = 0; i < result.row(); i++)
			for (int j = 0; j < other.col(); j++)
				result[i][this->col() + j] = other[i][j];
		break;

	default:
		break;
	}

	return result;
}

void Matrix::sig() {
	calculation(SIGMOID, this, 0.0, this);
}

bool Matrix::saveToFile(string filePath) {

	fstream fileStream(filePath, ios::out | ios::trunc);
	
	if (!fileStream.is_open()) {
		if (DEBUG_M) cout << "Cannot save matrix file [" << filePath << "]\n";
		return false;
	}

	fileStream << row() << "\t" << col() << endl;
	for (int i = 0; i < row(); i++) {
		for (int j = 0; j < col(); j++) {
			fileStream << (*this)[i][j] << "\t";
		}
		fileStream << "\n";
	}

	fileStream.close();

	return true;
}

bool Matrix::loadFromFile(string filePath, bool forceReshape) {
	fstream fileStream(filePath, ios::in);

	if (!fileStream.is_open()) {
		if(DEBUG_M) cout << "Cannot load matrix file [" << filePath << "]\n";
		return false;
	}
	int fileRow, fileCol;
	fileStream >> fileRow >> fileCol;

	if (!forceReshape && (fileRow != row() || fileCol != col())) {
		cout << "Row or Column of matrix is not equal [" << filePath << "]\n";
		exit(EXIT_FAILURE);
	}

	if (forceReshape)
		this->reshape(fileRow, fileCol);

	for (int i = 0; i < row(); i++)
		for (int j = 0; j < col(); j++)
			fileStream >> (*this)[i][j];

	fileStream.close();
	
	return true;
}

void Matrix::allocDevData() {
	if (GPU && mCpuIsChanged) {
		if (!mDevAllocated) {
			dtype* deviceData;
			cudaMalloc((void**)&deviceData, sizeof(dtype) * size());
			checkCudaError();
			cudaMemcpy(deviceData, mData, sizeof(dtype) * size(), cudaMemcpyHostToDevice);
			checkCudaError();
			mDevData = deviceData;
			mDevAllocated = true;
		}
		else {
			cudaMemcpy(mDevData, mData, sizeof(dtype) * size(), cudaMemcpyHostToDevice);
			checkCudaError();
		}
		mCpuIsChanged = false;
	}
}

void Matrix::retrieveDataFromDevice() {
	if (GPU) {
		if (mDevAllocated) {
			if (mGpuIsChanged) {
				cudaMemcpy(mData, mDevData, sizeof(dtype) * size(), cudaMemcpyDeviceToHost);
				checkCudaError();
				mGpuIsChanged = false;
			}
		}
		else {
			cout << "Device data is not allocated\n";
			exit(EXIT_FAILURE);
		}
	}
}