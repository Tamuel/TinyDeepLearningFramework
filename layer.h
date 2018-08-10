#pragma once
#include <iostream>
#include "matrix.h"

using namespace std;

enum LayerType {FC_LAYER, SOFTMAX_LAYER, ACT_LAYER, BIAS_LAYER};

class Layer {
private:

protected:
	dtype mLearningRate;
	LayerType mLayerType;

public:
	Layer();

	virtual Matrix* forwardPropagation(Matrix* aInput) = 0;

	virtual Matrix* backwardPropagation(Matrix* aGradient) = 0;

	virtual void updateWeights() = 0;

	virtual void setLearningRate(dtype aRate) = 0;

	virtual dtype getLearningRate() = 0;

	virtual bool saveWeight(string aPath) = 0;

	virtual bool loadWeight(string aPath) = 0;

	LayerType type();
};