#pragma once
#include <iostream>
#include "matrix.h"
#include "layer.h"

using namespace std;


// Fully Connected Layer
class BiasLayer : public Layer {
private:
	int mDimenstion;

	Matrix mOutput;
	Matrix mInput;

	Matrix mBias;

	Matrix gradient;
	Matrix mCumulatedGradient;

public:
	BiasLayer(int aDimension, dtype aLearningRate = 0.01);

	// Forward Propagation algorithm
	Matrix* forwardPropagation(Matrix* aInput);

	// Gradient Backward Propagation algorithm
	Matrix* backwardPropagation(Matrix* aGradient);

	// Update weights of fullyconnected layer
	void updateWeights();

	void setLearningRate(dtype aRate) {
		mLearningRate = aRate;
	}
	dtype getLearningRate() {
		return mLearningRate;
	}
	bool saveWeight(string path) {
		return mBias.saveToFile(path);
	}
	bool loadWeight(string path) {
		return mBias.loadFromFile(path);
	}
};