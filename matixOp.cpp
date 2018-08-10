#include "matrixOp.h"
#include <math.h>
#include <random>
#include <time.h>


void sigmoid(Matrix* a) {
	for (int i = 0; i < (*a).row(); i++)
		for (int j = 0; j < (*a).col(); j++)
			(*a)[i][j] = 1.0 / (1.0 + exp(-(*a)[i][j]));
}

void random(Matrix* a, dtype aMin, dtype aMax) {
	if (aMax < aMin) {
		cout << "Max value must larger than Min value\n";
		exit(EXIT_FAILURE);
	}
	random_device rd;
	mt19937_64 gen(rd());
	uniform_real_distribution<> dist(aMin, aMax);

	for (int i = 0; i < a->row(); i++)
		for (int j = 0; j < a->col(); j++)
			(*a)[i][j] = dist(gen);
}

Matrix onehot(Matrix* values, int row, int col) {
	Matrix output(row, col);
	
	for (int i = 0; i < values->col(); i++)
		output[(*values)[0][i]][i] = 1;

	return output;
}


Matrix argMax(Matrix* a, dimension dim) {
	Matrix result;
	switch (dim) {
	case ROW:
		result.reshape(a->row(), 1);

		for (int i = 0; i < a->row(); i++) {
			int y = 0;
			for (int j = 0; j < a->col(); j++) {
				if ((*a)[i][j] >(*a)[i][y]) {
					y = j;
				}
			}
			result[i][0] = y;
		}
		break;

	case COL:
		result.reshape(1, a->col());

		for (int i = 0; i < a->col(); i++) {
			int x = 0;
			for (int j = 0; j < a->row(); j++) {
				if ((*a)[j][i] >(*a)[x][i]) {
					x = j;
				}
			}
			result[0][i] = x;
		}
		break;
	}

	return result;
}

Matrix argMin(Matrix* a) {
	int x = 0, y = 0;
	for (int i = 0; i < a->row(); i++) {
		for (int j = 0; j < a->col(); j++) {
			if ((*a)[i][j] < (*a)[x][y]) {
				x = i;
				y = j;
			}
		}
	}
	Matrix result(1, 2);
	result[0][0] = x;
	result[0][1] = y;

	return result;
}