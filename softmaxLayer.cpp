#include "softmaxLayer.h"
#include <math.h>

SoftmaxLayer::SoftmaxLayer() {
	mLayerType = SOFTMAX_LAYER;
}

Matrix* SoftmaxLayer::forwardPropagation(Matrix* aInput) {
	mOutput = *aInput;
	mOutput.retrieveDataFromDevice();
	dtype sigma;
	for (int c = 0; c < mOutput.col(); c++) {
		sigma = 0;
		for (int r = 0; r < mOutput.row(); r++)
			sigma += exp((*aInput)[r][c]);
		for (int r = 0; r < mOutput.row(); r++)
			mOutput[r][c] = exp((*aInput)[r][c]) / sigma;
	}
	mOutput.allocDevData();

	return &mOutput;
}

Matrix* SoftmaxLayer::backwardPropagation(Matrix* aGradient) {
	gradient = mOutput;

	gradient -= 1;
	gradient *= -1;
	
	gradient *= mOutput;
	gradient *= (*aGradient);

	return &gradient;
}

void SoftmaxLayer::updateWeights() {

}