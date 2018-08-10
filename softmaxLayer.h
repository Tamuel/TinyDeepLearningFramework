#pragma once
#include <iostream>
#include "layer.h"

using namespace std;

class SoftmaxLayer : public Layer {
private:
	Matrix mOutput;
	Matrix gradient;

public:
	SoftmaxLayer();

	Matrix* forwardPropagation(Matrix* aInput);

	Matrix* backwardPropagation(Matrix* aGradient);

	void updateWeights();

	dtype getLearningRate() {
		return mLearningRate;
	}

	void setLearningRate(dtype aRate) {
		mLearningRate = aRate;
	}

	bool saveWeight(string aPath) {
		return true;
	}

	bool loadWeight(string aPath) {
		return true;
	}
};