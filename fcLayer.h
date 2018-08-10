#pragma once
#include <iostream>
#include "matrix.h"
#include "layer.h"

using namespace std;


// Fully Connected Layer
class FcLayer : public Layer{
private:
	int mInputDim;
	int mOutputDim;

	Matrix mOutput;
	Matrix mInput;
	
	Matrix mWeight;

	Matrix temp;
	Matrix gradient;
	Matrix mCumulatedGradient;

public:
	FcLayer(int aInputDimension, int aOutputDimension, dtype aLearningRate = 0.01);

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
		return mWeight.saveToFile(path);
	}
	bool loadWeight(string path) {
		return mWeight.loadFromFile(path);
	}
};