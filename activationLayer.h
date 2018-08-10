#pragma once
#include <iostream>
#include "layer.h"

using namespace std;

enum activation {ACT_SIGMOID, ACT_RELU};

class ActivationLayer : public Layer {
private:
	activation mActivation;
	Matrix mOutput;

	Matrix gradient;

public:
	ActivationLayer(activation aActivation);

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