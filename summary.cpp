#include "summary.h"
#include <windows.h>


Summary::Summary(string aSummaryFolder) {
	mSummaryFolder = aSummaryFolder;
	
	mTrainSummary.reshape(0, 5);
	mTestSummary.reshape(0, 3);
}

Summary::~Summary() {

}

void Summary::addTrainSummary(int aStep, dtype aTrainLoss, dtype aValLoss, dtype aTrainAcc, dtype aValAcc) {
	Matrix temp(1, 5);
	temp[0][0] = mLoaded ? mLastStep + aStep : aStep;
	temp[0][1] = aTrainLoss;
	temp[0][2] = aValLoss;
	temp[0][3] = aTrainAcc;
	temp[0][4] = aValAcc;

	mTrainSummary = mTrainSummary.concat(ROW, temp);
}

void Summary::addTestSummary(dtype aTestLoss, dtype aTestAcc) {
	Matrix temp(1, 3);
	temp[0][0] = currentStep();
	temp[0][1] = aTestLoss;
	temp[0][2] = aTestAcc;

	mTestSummary = mTestSummary.concat(ROW, temp);
}

void Summary::saveSummary() {
	CreateDirectory(mSummaryFolder.c_str(), NULL);
	
	mTrainSummary.saveToFile(mSummaryFolder + "/TrainValResult.txt");
	mTestSummary.saveToFile(mSummaryFolder + "/TestResult.txt");
}

bool Summary::loadSummary() {
	bool loadTrain = mTrainSummary.loadFromFile(mSummaryFolder + "/TrainValResult.txt", true);
	bool loadTest = mTestSummary.loadFromFile(mSummaryFolder + "/TestResult.txt", true);

	mLoaded = loadTrain && loadTest;
	if (mLoaded) mLastStep = mTrainSummary[mTrainSummary.row() - 1][0];
	
	return mLoaded;
}

int Summary::lastStep() {
	return mLastStep;
}

int Summary::currentStep() {
	return mTrainSummary[mTrainSummary.row() - 1][0];
}