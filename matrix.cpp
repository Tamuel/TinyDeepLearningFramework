#include "matrix.h"
#include <fstream>
#include <string>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <math.h>

#define ZERO 0.0000000001

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
double Matrix::threadScal;

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
	unique_lock<mutex> threadReadyLock(mThreadsReadyMtx[threadId]);
	if(DEBUG_M) cout << "Initialize thread" << threadId << endl;
	while (mThreadRun[threadId]) {
		//if (mThreadStart[threadId])
			//mThreadsReadyCV.wait(threadReadyLock);

		if (mThreadStart[threadId]) {
			mNumberLock.lock();
			runThreads++;
			mNumberLock.unlock();

			if (DEBUG_M) cout << "Start thread" << threadId << endl;
			dimension divisionDim;
			int start, end, range;
			if (threadM1->row() > threadM1->col()) {
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

			// Calculation
			switch (mMode) {
			case SUM:
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

void Matrix::startThreads() {
	// Start Threads
	for (int i = 0; i < N_THREAD; i++)
		mThreadStart[i] = true;

	//while(runThreads != N_THREAD)
	//	mThreadsReadyCV.notify_one();
	//runThreads = 0;

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

void Matrix::calculation(threadMode aMode, Matrix* m1, Matrix* m2, Matrix* m_des) {
	if (DEBUG_M) cout << "Start calculation" << endl;
	threadM1 = m1;
	threadM2 = m2;
	threadDestination = m_des;
	mMode = aMode;

	if (m_des == &threadResult)
		threadResult.reshape(m1->row(), m2->col());

	startThreads();
}

void Matrix::calculation(threadMode aMode, Matrix* m1, double scal, Matrix* m_des) {
	if (DEBUG_M) cout << "Start calculation" << endl;
	threadM1 = m1;
	threadScal = scal;
	threadDestination = m_des;
	mMode = aMode;

	if (m_des == &threadResult)
		threadResult.reshape(m1->row(), m1->col());

	startThreads();
}

void Matrix::allocData() {
	mData = new double[mRow * mCol];
	memset(mData, 0, sizeof(double) * mRow * mCol);

	mAllocated = true;
}

void Matrix::freeData() {
	if (mAllocated) {
		delete[] mData;
	}
	mAllocated = false;
} 

int Matrix::row() {
	return mRow;
}

int Matrix::col() {
	return mCol;
}

double* Matrix::operator[](int i) {
	return &mData[i];
}

void Matrix::operator=(Matrix& other) {
	reshape(other.mRow, other.mCol);
	calculation(COPY, &other, nullptr, this);
}

Matrix Matrix::matMult(Matrix* a, Matrix* b) {
	if (a->col() != b->row()) {
		cout << "Cannot multiply matrix" << endl;
		exit(EXIT_FAILURE);
	}

	calculation(MULT, a, b);

	return threadResult;
}

Matrix Matrix::matMult(double a, Matrix* b) {
	Matrix result(b->row(), b->col());

	calculation(SCAL_MULT, b, a);

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
	Matrix cov = matMult(1 / double(a->row() - 1), &matMult(&matTranspose(&D), &D));

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

double Matrix::matSum(Matrix* a) {
	double result = 0;

	for (int i = 0; i < a->row(); i++)
		for (int j = 0; j < a->col(); j++)
			result += (*a)[i][j];

	return result;
}

double Matrix::matDet(Matrix* a) {
	if (a->row() != a->col()) {
		cout << "Cannot get determinant" << endl;
		exit(EXIT_FAILURE);
	}

	double result = 0;

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

void Matrix::matRowSwap(Matrix* a, int index1, int index2) {
	if (index1 == index2)
		return;

	double temp;
	for (int i = 0; i < a->col(); i++) {
		temp = (*a)[index1][i];
		(*a)[index1][i] = (*a)[index2][i];
		(*a)[index2][i] = temp;
	}
}

void Matrix::matColSwap(Matrix* a, int index1, int index2) {
	if (index1 == index2)
		return;

	double temp;
	for (int i = 0; i < a->row(); i++) {
		temp = (*a)[i][index1];
		(*a)[i][index1] = (*a)[i][index2];
		(*a)[i][index2] = temp;
	}
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

double zeroCheck(double value) {
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

	if (first != -1) matRowSwap(&concat, 0, first);

	// Row reduction
	for (int r = 0; r < concat.row() - 1; r++) {
		if (concat[r + 1][r] != 0) {
			for (int r2 = r + 1; r2 < concat.row(); r2++) {
				double coe = -concat[r2][r] / concat[r][r];

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

			matRowSwap(&concat, r + 1, swapIndex);
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
			double coe = 1.0 / concat[r][r];

			for (int c = 0; c < concat.col(); c++)
				concat[r][c] *= coe;
		}
	}

	// Make to identity matrix
	for (int r = concat.row() - 2; r >= 0; r--) {
		for (int r2 = r + 1; r2 < concat.row(); r2++) {
			if (concat[r][r2] != 0) {
				double coe = -concat[r][r2] / concat[r2][r2];

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

void Matrix::matAbs() {
	for (int i = 0; i < row(); i++)
		for (int j = 0; j < col(); j++)
			(*this)[i][j] = abs((*this)[i][j]);
}

void Matrix::operator()(int value) {
	calculation(SET, this, value, this);
}

void Matrix::operator()(initializer_list<double> values, bool repeat) {
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

void Matrix::operator-=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot subtract matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(SUB, this, &other, this);
}

void Matrix::operator+=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot subtract matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(SUM, this, &other, this);
}

void Matrix::operator-=(double other) {
	calculation(SCAL_SUB, this, other, this);
}

void Matrix::operator+=(double other) {
	calculation(SCAL_SUM, this, other, this);
}

void Matrix::operator/=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot subtract matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(ELT_DIVIDE, this, &other, this);
}

void Matrix::operator/=(double other) {
	calculation(SCAL_DIVIDE, this, other, this);
}

// Element wise multiplication
void Matrix::operator*=(Matrix& other) {
	if (row() != other.row() || col() != other.col()) {
		cout << "Cannot subtract matrix" << endl;
		exit(EXIT_FAILURE);
	}
	calculation(ELT_MULT, this, &other, this);
}

void Matrix::operator*=(double other) {
	calculation(SCAL_MULT, this, other, this);
}

double Matrix::operator+(double other) {
	if (this->row() * this->col() == 1)
		return (*this)[0][0] + other;
	else {
		cout << "Cannot add double and matrix, matrix size must be 1" << endl;
		exit(EXIT_FAILURE);
	}
}

double Matrix::operator-(double other) {
	if (this->row() * this->col() == 1)
		return (*this)[0][0] - other;
	else {
		cout << "Cannot add double and matrix, matrix size must be 1" << endl;
		exit(EXIT_FAILURE);
	}
}

Matrix Matrix::operator^(double coefficient) {
	calculation(SCAL_POW, this, coefficient);
	return threadResult;
}

// Matrix multiplication
Matrix Matrix::operator*(Matrix& other) {
	return matMult(this, &other);
}

Matrix Matrix::operator*(double other) {
	return matMult(other, this);
}

Matrix Matrix::operator/(Matrix& other) {
	calculation(ELT_DIVIDE, this, &other);

	return threadResult;
}

Matrix Matrix::operator/(double other) {
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
	return matInverse2(this);
}

Matrix Matrix::PI() {
	return matPseudoInverse(this);
}

double Matrix::det() {
	return matDet(this);
}

double Matrix::sum() {
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
	freeData();
	mRow = aRow;
	mCol = aCol;
	allocData();
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
			for (int j = 0; j < result.col(); j++)
				result[this->row() + i][j] = other[i][j];
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

void Matrix::saveToFile(string filePath) {
	fstream fileStream(filePath, ios::out | ios::trunc);
	fileStream << row() << "\t" << col() << endl;
	for (int i = 0; i < row(); i++) {
		for (int j = 0; j < col(); j++) {
			fileStream << (*this)[i][j] << "\t";
		}
		fileStream << "\n";
	}
	
	fileStream.close();
}

void Matrix::loadFromFile(string filePath, bool forceReshape) {
	fstream fileStream(filePath, ios::in);

	if (!fileStream.is_open()) {
		cout << "Cannot load matrix file [" << filePath << "]\n";
		exit(EXIT_FAILURE);
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
}