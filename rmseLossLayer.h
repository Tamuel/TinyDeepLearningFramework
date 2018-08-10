#pragma once
#include <iostream>
#include "layer.h"

using namespace std;

class RmseLossLayer : public Layer{
private:

public:
	RmseLossLayer();

	Matrix forwardPropagation(Matrix* aInput);

	Matrix backwardPropagation(Matrix* aLoss);
};