#pragma once
#include <iostream>
#include "matrix.h"

using namespace std;

void sigmoid(Matrix* a);

void random(Matrix* a, dtype aMin = -1, dtype aMax = 1);

Matrix onehot(Matrix* values, int row, int col);

Matrix argMax(Matrix* a, dimension dim = ROW);

Matrix argMin(Matrix* a);