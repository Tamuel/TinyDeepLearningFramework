#pragma once
#include <iostream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <Windows.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#define N_THREAD 1
#define DEBUG_M false
#define GPU true
#define CPU !GPU
#define N_STREAM 2

using namespace std;

typedef double dtype;

enum threadMode {SUM, SUM_ASSIGN, SCAL_SUM, SCAL_SUM_ASSIGN, SUB, SUB_ASSIGN, SCAL_SUB, SCAL_SUB_ASSIGN,
	MULT, ELT_MULT, ELT_MULT_ASSIGN, SCAL_MULT, SCAL_MULT_ASSIGN, ELT_DIVIDE, ELT_DIVIDE_ASSIGN, SCAL_DIVIDE,
	SCAL_DIVIDE_ASSIGN, SET, COPY, SCAL_POW, TRANS, ROW_SWAP, COL_SWAP, SIGMOID, NOT_EQUAL};
enum dimension { ROW, COL };
class Matrix {
private:
	int mRow;
	int mCol;

	// For CPU
	dtype* mData;
	bool mHostAllocated;
	bool mCpuIsChanged;

	// For GPU
	dtype* mDevData;
	bool mDevAllocated;
	bool mGpuIsChanged;

	// For multithreading
	static int numberOfMatrices;
	static bool mInitThread;
	static bool mThreadRun[N_THREAD];
	static bool mThreadStart[N_THREAD];
	static bool mThreadWait[N_THREAD];
	static bool mThreadFinish[N_THREAD];
	static mutex mThreadsReadyMtx[N_THREAD];
	static mutex mNumberLock;
	static int runThreads;
	static condition_variable mThreadsReadyCV;
	static vector<thread> mThreads;
	static threadMode mMode;
	static Matrix* threadM1;
	static Matrix* threadM2;
	static Matrix threadResult;
	static Matrix* threadDestination;
	static dtype threadScal;
	static bool threadBool;
	static bool* gpuBool;
	static void* threadArg1;
	static void* threadArg2;
	static cudaStream_t cudaStream[N_STREAM];
	static int currentStream;

	// For gpu calculation
	static bool gpuIsReady;

	void readyForGpuCalc();

	void initThreads();

	void stopThreads();

	static void threadFunction(int threadId);

	void startGPUCalc();

	void startThreads();

	void calculation(threadMode aMode, Matrix* m1, Matrix* m2, Matrix* m_des = &threadResult, void* arg1 = NULL, void* arg2 = NULL);

	void calculation(threadMode aMode, Matrix* m1, dtype scal, Matrix* m_des = &threadResult, void* arg1 = NULL, void* arg2 = NULL);

	void allocHostData();

	void freeHostData();

	void freeDevData();

public:

	Matrix() {
		numberOfMatrices++;
		mCpuIsChanged = false;
		mGpuIsChanged = false;
		mHostAllocated = false;
		mDevAllocated = false;
		if (CPU) initThreads();
		else if (GPU) readyForGpuCalc();
		mRow = 0;
		mCol = 0;
		mData = NULL;
	}

	Matrix(int aRow, int aCol) {
		numberOfMatrices++;
		mCpuIsChanged = false;
		mGpuIsChanged = false;
		mHostAllocated = false;
		mDevAllocated = false;
		if (CPU) initThreads();
		else if (GPU) readyForGpuCalc();
		mRow = aRow;
		mCol = aCol;
		allocHostData();
		allocDevData();
	}

	Matrix(Matrix& other) {
		numberOfMatrices++;
		mCpuIsChanged = false;
		mGpuIsChanged = false;
		mHostAllocated = false;
		mDevAllocated = false;
		mRow = other.row();
		mCol = other.col();
		allocHostData();
		allocDevData();
		calculation(COPY, &other, nullptr, this);
	}

	~Matrix() {
		numberOfMatrices--;
		if(numberOfMatrices == 0 && CPU) stopThreads();
		freeHostData();
		freeDevData();
	}

	static void changeCudaStream() {
		currentStream++;
		if (currentStream == N_STREAM)
			currentStream = 0;
	}

	__host__ __device__ int row() {
		return mRow;
	}

	__host__ __device__ int col() {
		return mCol;
	}

	__host__ __device__ int size() {
		return mCol * mRow;
	}

	__host__ __device__ dtype* operator[](int i) {
		if (i >= this->row()) {
			cout << "Cannot access matrix[" << i << "]\n";
			exit(EXIT_FAILURE);
		}
		retrieveDataFromDevice();
		mCpuIsChanged = true;
		return &mData[i * this->col()];
	}

	void operator=(Matrix& other);

	Matrix matMult(Matrix* a, Matrix* b, Matrix* des = NULL, bool transA = false, bool transB = false);

	Matrix matMult(dtype a, Matrix* b, Matrix* des = NULL);

	Matrix matTranspose(Matrix* a);

	Matrix matMean(Matrix* a, dimension d);

	Matrix matCov(Matrix* a, dimension d);

	Matrix matAdd(Matrix* a, Matrix* b);

	Matrix matSub(Matrix* a, Matrix* b);

	dtype matSum(Matrix* a);

	dtype matDet(Matrix* a);

	Matrix matInverse(Matrix* a);

	Matrix matInverse2(Matrix* a);

	Matrix matPseudoInverse(Matrix* a);

	void rowSwap(int index1, int index2);

	void colSwap(int index1, int index2);

	void matAbs();

	void operator()(int value);

	void operator()(initializer_list<dtype> values, bool repeat = false);

	Matrix operator+(Matrix& other);

	Matrix operator-(Matrix& other);

	void operator-=(Matrix& other);

	void operator+=(Matrix& other);

	void operator-=(dtype other);

	void operator+=(dtype other);

	// Element wise division
	void operator/=(Matrix& other);

	// Element wise division
	void operator/=(dtype other);

	// Element wise multiplication
	void operator*=(Matrix& other);

	// Element wise multiplication
	void operator*=(dtype other);

	dtype operator+(dtype other);

	dtype operator-(dtype other);

	bool operator==(Matrix& other);

	bool operator!=(Matrix& other);

	// Matrix multiplication
	Matrix operator*(Matrix& other);

	Matrix operator*(dtype other);

	// Element wise division
	Matrix operator/(Matrix& other);

	// Element wise division
	Matrix operator/(dtype other);

	Matrix operator^(dtype coefficient);

	Matrix cov(dimension d);

	Matrix mean(dimension d);

	// Transpose
	Matrix T();

	// Inverse
	Matrix I();

	// Pesudo Inverse
	Matrix PI();

	dtype det();

	// Summation of matrix's all elements
	dtype sum();

	Matrix rowSum();

	Matrix colSum();

	Matrix repeat(int row, int col = 1);

	// Data will be swiped out
	void reshape(int aRow, int aCol);

	void print();

	void copyFrom(Matrix& source, int srcRowOffset = 0, int srcColOffset = 0);

	void copyTo(Matrix& destination, int dstRowOffset = 0, int dstColOffset = 0);

	// Split matrix, specific dimension with [aStart, aEnd)
	Matrix split(dimension dimen, int aStart, int aSize);
	
	// Concatenate this and other matrix
	Matrix concat(dimension dimen, Matrix& other);
	
	// Sigmoid
	void sig();

	bool saveToFile(string filePath);

	bool loadFromFile(string filePath, bool forceReshape = false);

	void allocDevData();

	void retrieveDataFromDevice();
};
