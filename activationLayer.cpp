#include "activationLayer.h"
#include "matrixOp.h"


ActivationLayer::ActivationLayer(activation aActivation) {
	mLayerType = ACT_LAYER;
	mActivation = aActivation;
}

Matrix* ActivationLayer::forwardPropagation(Matrix* aInput) {
	switch (mActivation) {
	case ACT_SIGMOID:
		mOutput = *aInput;
		mOutput.sig(); // Apply Sigmoid activation function
		break;
		
	case ACT_RELU:
		break;
	}

	return &mOutput;
}

Matrix* ActivationLayer::backwardPropagation(Matrix* aGradient) {
	gradient = mOutput;

	switch (mActivation) {
	case ACT_SIGMOID:
		gradient -= 1;
		gradient *= -1;
		gradient *= mOutput;
		gradient *= (*aGradient);
		break;

	case ACT_RELU:
		break;
	}


	return &gradient;
}

void ActivationLayer::updateWeights() {

}