#ifndef CCPP_DATA_PROVIDER_H
#define CCPP_DATA_PROVIDER_H

#include <iostream>
#include <string>
#include <fstream>
#include <Windows.h>
#include <cstdlib>
#include "matrix.h"
#include "matrixOp.h"

#define N_DATASET_ATTR 28 * 28 + 1

using namespace std;

extern int imgMagicNumber, labelMagicNumber;
extern int numberOfLabels, numberOfImages;
extern int numberOfRows, numberOfColumns;

Matrix mnistDataProvider(string img_path, string label_path, int dataSize);

Matrix twoMoonDataProvider(string path, int dataSize);

void ShowConsoleCursor(bool showFlag);

void gotoxy(int x, int y);

COORD getxy();

void drawMatrix(string name, Matrix* input);


#endif
