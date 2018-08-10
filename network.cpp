#include "network.h"
#include "srcMeasure.h"

Network::Network(int aBatchSize) {
	isOutputUpdated = false;
	isLabelUpdated = false;
	mBatchSize = aBatchSize;
	mNumberOfLearning = 0;
}

void Network::addLayer(Layer* aLayer) {
	mLayers.push_back(aLayer);
}

Matrix* Network::forwardPropagation(Matrix* aInput) {
	SrcMeasure sm;
	//sm.startTime(1);
	mOutput = aInput;
	for (int i = 0; i < mLayers.size(); i++) {
		sm.startTime(0);
		mOutput = mLayers[i]->forwardPropagation(mOutput);
		sm.endTime(0, to_string(i) + " Layer");
	}
	
	isOutputUpdated = true;
	//sm.endTime(1, "Forward inner");
	return mOutput;
}

void Network::backwardPropagation() {
	if (isOutputUpdated && isLabelUpdated) {
		SrcMeasure sm;
		Matrix temp = ((*mOutput) - mLabel);
		Matrix* gradient = &temp; // Gradient of RMSE

		for (int i = mLayers.size() - 1; i >= 0; i--) {
			sm.startTime(0);
			gradient = mLayers[i]->backwardPropagation(gradient);
			sm.endTime(0, to_string(i) + " Layer");
		}
		updateWeights();
	}
	else {
		cout << "Need to forward propagate network and calculate loss\n";
	}

	isOutputUpdated = false;
	isLabelUpdated = false;
}

void Network::updateWeights() {
	for (int i = 0; i < mLayers.size(); i++)
		mLayers[i]->updateWeights();
}

dtype Network::rmseLoss(Matrix* aLabel) {
	if (isOutputUpdated) {
		mLabel = *aLabel;
		isLabelUpdated = true;
		mLoss = ((*mOutput - mLabel) ^ 2).sum() / aLabel->row();
	}
	else {
		cout << "Need to forward propagate network and get output\n";
		return 0;
	}
	return mLoss;
}

void Network::learningRateDecay(dtype aDecayRate) {
	for (int i = 0; i < mLayers.size(); i++) {
		mLayers[i]->setLearningRate(mLayers[i]->getLearningRate() * aDecayRate);
	}
}

void Network::saveWeights(string aFolderPath, int aStep) {
	CreateDirectory(aFolderPath.c_str(), NULL);
	for (int i = 0; i < mLayers.size(); i++) {
		bool success = false;
		switch (mLayers[i]->type()) {
		case FC_LAYER:
			success = mLayers[i]->saveWeight(aFolderPath + "/FC" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;

		case BIAS_LAYER:
			success = mLayers[i]->saveWeight(aFolderPath + "/BIAS" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;

		case ACT_LAYER:
			success = mLayers[i]->saveWeight(aFolderPath + "/ACT" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;

		case SOFTMAX_LAYER:
			success = mLayers[i]->saveWeight(aFolderPath + "/SOFTMAX" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;
		}
		if (!success) {
			cout << "Cannot load weights " << i << endl;
			exit(EXIT_FAILURE);
		}
	}
}

void Network::loadWeights(string aFolderPath, int aStep) {
	if (DEBUG_M) cout << "Loading networks weights\n";
	for (int i = 0; i < mLayers.size(); i++) {
		bool success = false;
		switch (mLayers[i]->type()) {
		case FC_LAYER:
			success = mLayers[i]->loadWeight(aFolderPath + "/FC" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;

		case BIAS_LAYER:
			success = mLayers[i]->loadWeight(aFolderPath + "/BIAS" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;

		case ACT_LAYER:
			success = mLayers[i]->loadWeight(aFolderPath + "/ACT" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;

		case SOFTMAX_LAYER:
			success = mLayers[i]->loadWeight(aFolderPath + "/SOFTMAX" + to_string(i) + "-" + to_string(aStep) + ".txt");
			break;
		}
		if (!success) {
			cout << "Cannot load weights " << i << endl;
			exit(EXIT_FAILURE);
		}
		else if (DEBUG_M) cout << "Load wieghts " << i << endl;
	}
}