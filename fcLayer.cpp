#include "fcLayer.h"
#include "matrixOp.h"
#include "srcMeasure.h"


FcLayer::FcLayer(int aInputDimension, int aOutputDimension, dtype aLearningRate) {
	mLayerType = FC_LAYER;

	mInputDim = aInputDimension;
	mOutputDim = aOutputDimension;
	mLearningRate = aLearningRate;
	
	mWeight = Matrix(mOutputDim, mInputDim);
	random(&mWeight, -0.5, 0.5);
	mCumulatedGradient = Matrix(mWeight.row(), mWeight.col());
	temp = Matrix(mOutputDim, mInputDim);
}

Matrix* FcLayer::forwardPropagation(Matrix* aInput) {
	mInput = *aInput;

	if (gradient.col() != aInput->col())
		gradient.reshape(mInputDim, aInput->col());
	if (mOutput.col() != aInput->col())
		mOutput.reshape(mOutputDim, aInput->col());

	mOutput.matMult(&mWeight, aInput, &mOutput);
	return &mOutput;
}

Matrix* FcLayer::backwardPropagation(Matrix* aGradient) {
	gradient.matMult(&mWeight, aGradient, &gradient, true, false);
	Matrix::changeCudaStream();
	temp.matMult(aGradient, &mInput, &temp, false, true);
	mCumulatedGradient += temp;

	return &gradient;
}

void FcLayer::updateWeights() {
	mCumulatedGradient *= mLearningRate;
	mCumulatedGradient /= mOutput.col();
	mWeight -= mCumulatedGradient;

	mCumulatedGradient(0);
}