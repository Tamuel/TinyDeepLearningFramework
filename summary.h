#pragma once
#include <iostream>
#include <string>
#include "matrix.h"

class Summary {
private:
	Matrix mTrainSummary;
	Matrix mTestSummary;

	string mSummaryFolder;

	bool mLoaded;
	int mLastStep;

public:

	Summary(string aSummaryFolder);

	~Summary();

	void addTrainSummary(int aStep, dtype aTrainLoss, dtype aValLoss, dtype aTrainAcc, dtype aValAcc);
	
	void addTestSummary(dtype aTestLoss, dtype aTestAcc);

	void saveSummary();

	bool loadSummary();

	int lastStep();

	int currentStep();
};