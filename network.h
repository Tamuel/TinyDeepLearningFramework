#pragma once
#include <iostream>
#include "layer.h"
#include "matrix.h"
#include <vector>

using namespace std;

class Network {
private:
	vector<Layer*> mLayers;

	Matrix* mOutput;
	Matrix mLabel;
	dtype mLoss;

	bool isOutputUpdated;
	bool isLabelUpdated;

	int mBatchSize;
	int mNumberOfLearning;

public:
	Network(int aBatchSize);

	void addLayer(Layer* aLayer);

	Matrix* forwardPropagation(Matrix* aInput);

	void backwardPropagation();

	void updateWeights();

	dtype rmseLoss(Matrix* aLabel);

	void learningRateDecay(dtype aDecayRate);

	void saveWeights(string aFolderPath, int aStep);

	void loadWeights(string aFolderPath, int aStep);
};