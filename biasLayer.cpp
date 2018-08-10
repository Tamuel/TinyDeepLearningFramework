#include "biasLayer.h"
#include "matrixOp.h"


BiasLayer::BiasLayer(int aDimension, dtype aLearningRate) {
	mLayerType = BIAS_LAYER;

	mDimenstion = aDimension;
	mLearningRate = aLearningRate;

	mBias = Matrix(aDimension, 1);
	random(&mBias, -0.5, 0.5);
}

Matrix* BiasLayer::forwardPropagation(Matrix* aInput) {
	mInput = *aInput;
	Matrix::changeCudaStream();
	mOutput = mBias.repeat(1, aInput->col());
	mOutput += (*aInput); // Calculate output

	return &mOutput;
}

Matrix* BiasLayer::backwardPropagation(Matrix* aGradient) {
	gradient = (*aGradient);

	Matrix::changeCudaStream();
	if (mCumulatedGradient.col() != aGradient->col())
		mCumulatedGradient.reshape(mBias.row(), aGradient->col());

	mCumulatedGradient += (*aGradient);

	return &gradient;
}

void BiasLayer::updateWeights() {
	mCumulatedGradient *= mLearningRate;
	mCumulatedGradient /= mCumulatedGradient.col();

	mBias -= mCumulatedGradient.colSum();

	mCumulatedGradient(0);
}