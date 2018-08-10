#include <iostream>
#include "dataProvider.h"
#include "matrix.h"
#include "matrixOp.h"
#include "network.h"
#include "fcLayer.h"
#include "biasLayer.h"
#include "activationLayer.h"
#include "softmaxLayer.h"
#include "srcMeasure.h"
#include "summary.h"

#define N_DATA 60000
#define N_TRAINING_SET 55000
#define N_VALIDATION_SET 5000
#define N_TEST_SET 10000

#define INPUT 28 * 28
#define OUTPUT 10

#define STEP 0
#define TRAIN_LOSS 1
#define VAL_LOSS 2
#define TRAIN_ACC 3
#define VAL_ACC 4

using namespace std;

void test(Matrix& other) {
	other.row();
	other.col();
}

Matrix matrixAdd(Matrix& m1, Matrix& m2) {
	return m1 + m2;
}

void main(void) {
	bool inputArg = true;
	int neurons = 1024;
	cout << "Number of neurons : ";
	if (inputArg) cin >> neurons;
	dtype learningRate = 0.3;
	cout << "Initial Learning Rate : ";
	if (inputArg) cin >> learningRate;
	dtype learningRateDecay = 0.995;
	cout << "Learning Rate Decay Rate: ";
	if (inputArg) cin >> learningRateDecay;
	int EPOCH = 10;
	cout << "EPOCH : ";
	if (inputArg) cin >> EPOCH;
	int BATCH_SIZE = 100;
	cout << "Batch Size : ";
	if (inputArg) cin >> BATCH_SIZE;


	ShowConsoleCursor(false);
	Matrix data = mnistDataProvider(
		"D:/mnist/train-images.idx3-ubyte",
		"D:/mnist/train-labels.idx1-ubyte",
		N_DATA
	);
	Matrix testData = mnistDataProvider(
		"D:/mnist/t10k-images.idx3-ubyte",
		"D:/mnist/t10k-labels.idx1-ubyte",
		N_TEST_SET
	);
	//Matrix data = twoMoonDataProvider("two_moon.txt", N_DATA);
	//data.print();

	Matrix trainingData = data.split(ROW, 0, N_TRAINING_SET);
	//trainingData.print();
	trainingData = trainingData.T();
	Matrix validationData = data.split(ROW, N_TRAINING_SET, N_VALIDATION_SET);
	//validationData.print();
	validationData = validationData.T();
	//Matrix testData = data.split(ROW, N_TRAINING_SET + N_VALIDATION_SET, N_TEST_SET);
	//testData.print();
	//testData = testData.T();
	// Summarize Step, Training Loss, Training Accuracy, Validation Loss, Validation Accuracy, Test Loss, Test Accuracy
	Summary summary("Summary_" + to_string(neurons));
	//summary.loadSummary();

	Network network(BATCH_SIZE); // Batch size is 100

	FcLayer fc1(INPUT, neurons, learningRate);
	BiasLayer bias1(neurons, learningRate);
	ActivationLayer fc1_sig(ACT_SIGMOID);
	FcLayer fc2(neurons, neurons, learningRate);
	BiasLayer bias2(neurons, learningRate);
	ActivationLayer fc2_sig(ACT_SIGMOID);
	FcLayer fc3(neurons, neurons, learningRate);
	BiasLayer bias3(neurons, learningRate);
	ActivationLayer fc3_sig(ACT_SIGMOID);
	FcLayer fc4(neurons, OUTPUT, learningRate);
	BiasLayer bias4(OUTPUT, learningRate);
	SoftmaxLayer softmax;

	network.addLayer(&fc1);
	network.addLayer(&bias1);
	network.addLayer(&fc1_sig);
	network.addLayer(&fc2);
	network.addLayer(&bias2);
	network.addLayer(&fc2_sig);
	//network.addLayer(&fc3);
	//network.addLayer(&bias3);
	//network.addLayer(&fc3_sig);
	network.addLayer(&fc4);
	network.addLayer(&bias4);
	network.addLayer(&softmax);

	cout << "last step " << summary.lastStep() << endl;

	//network.loadWeights("Weights_" + to_string(neurons), summary.lastStep());

	Matrix input(INPUT, BATCH_SIZE);
	Matrix output;
	dtype loss;

	SrcMeasure sm;
	system("CLS");

	sm.startTime(0);
	Matrix summary_temp(1, 5);
	int valInterval = 20;

	summary_temp(0);


	// Training
	for (int epoch = 0; epoch < EPOCH; epoch++) {
		for (int i = 0; i < N_TRAINING_SET; i += BATCH_SIZE) {
			gotoxy(0, 0);
			cout << "[Neurons : " << neurons << "][Epoch " << epoch << ", Data #" << i + BATCH_SIZE << "]=================" << endl;
			input.copyFrom(trainingData, 0, i);

			output = *network.forwardPropagation(&input); // Predict digit

			Matrix result = argMax(&output, COL);

			loss = network.rmseLoss(
				&onehot(
					&trainingData.split(COL, i, BATCH_SIZE).split(ROW, N_DATASET_ATTR - 1, 1),
					OUTPUT,
					BATCH_SIZE
				)
			); // Have to calculate loss for backpropagation
			summary_temp[0][TRAIN_LOSS] += loss;
			for (int acc = 0; acc < BATCH_SIZE; acc++) {
				if (int(trainingData[N_DATASET_ATTR - 1][i + acc]) == result[0][acc])
					summary_temp[0][TRAIN_ACC]++;
			}

			printf("Loss : %lf\n", loss / BATCH_SIZE);
			network.backwardPropagation(); // Backpropagate loss and update weights

			// Validate
			if ((i + BATCH_SIZE) % valInterval == 0) {
				summary_temp[0][STEP] = epoch * N_TRAINING_SET + i + BATCH_SIZE;
				COORD p = getxy();
				int correct = 0;
				for (int j = 0; j < N_VALIDATION_SET; j += BATCH_SIZE) {
					input.copyFrom(validationData, 0, j);
					output = *network.forwardPropagation(&input); // Predict digit
					result = argMax(&output, COL);
					loss = network.rmseLoss(
						&onehot(
							&validationData.split(COL, j, BATCH_SIZE).split(ROW, N_DATASET_ATTR - 1, 1),
							OUTPUT,
							BATCH_SIZE
						)
					);
					summary_temp[0][VAL_LOSS] += loss; // Get test loss
					for (int acc = 0; acc < BATCH_SIZE; acc++) {
						if (int(validationData[N_DATASET_ATTR - 1][j + acc]) == result[0][acc])
							correct++;
					}
					printf("[%d/%d]\n", j + BATCH_SIZE, N_VALIDATION_SET);
					gotoxy(0, p.Y);
				}
				summary_temp[0][VAL_LOSS] /= N_VALIDATION_SET;
				dtype testAccuracy = (dtype)correct / N_VALIDATION_SET * 100.0; // Get testset accuracy
				printf("\nTest accuracy : %.1f%%, Test loss : %lf\n", testAccuracy, summary_temp[0][VAL_LOSS]);
				summary_temp[0][VAL_ACC] = testAccuracy;
				summary_temp[0][TRAIN_LOSS] /= valInterval;
				summary_temp[0][TRAIN_ACC] /= (float(valInterval) / 100.0);

				summary.addTrainSummary(
					summary_temp[0][STEP],
					summary_temp[0][TRAIN_LOSS],
					summary_temp[0][VAL_LOSS],
					summary_temp[0][TRAIN_ACC],
					summary_temp[0][VAL_ACC]
				);
				summary_temp(0);

			}
		}
		network.learningRateDecay(learningRateDecay);
		summary.saveSummary();
		network.saveWeights("Weights_" + to_string(neurons), summary.currentStep());
	}

	summary.saveSummary();
	network.saveWeights("Weights_" + to_string(neurons), summary.currentStep());

	cout << "[Testing network]\n";
	// Testing
	Matrix result;
	summary_temp(0);
	summary_temp[0][0] = summary_temp[summary_temp.row() - 1][0];
	summary_temp[0][1] = 0;
	sm.endTime(0, "Learning Time");
	int correct = 0;
	for (int j = 0; j < N_TEST_SET; j += BATCH_SIZE) {
		input.copyFrom(testData, 0, j);
		output = *network.forwardPropagation(&input); // Predict digit
		result = argMax(&output, COL);
		loss = network.rmseLoss(
			&onehot(
				&testData.split(COL, j, BATCH_SIZE).split(ROW, N_DATASET_ATTR - 1, 1),
				OUTPUT,
				BATCH_SIZE
			)
		);
		summary_temp[0][1] += loss; // Get test loss
		for (int acc = 0; acc < BATCH_SIZE; acc++) {
			if (int(testData[N_DATASET_ATTR - 1][j + acc]) == result[0][acc])
				correct++;
		}
		printf("[Predict : %d, Label : %d] [%4d/%4d]", int(result[0][0]), int(testData[N_DATASET_ATTR - 1][j]), j + 1, N_TEST_SET);
		COORD p = getxy();
		gotoxy(0, p.Y);
	}
	summary_temp[0][1] /= N_TEST_SET;
	dtype testAccuracy = (dtype)correct / N_TEST_SET * 100; // Get testset accuracy
	printf("\nTest accuracy : %.1f%%, Test loss : %lf\n", testAccuracy, summary_temp[0][1]);
	summary_temp[0][2] = testAccuracy;
	summary.addTestSummary(summary_temp[0][1], summary_temp[0][2]);
	summary.saveSummary();

	if (true) { // Make Confusion Matrix
		data = data.T();
		Matrix confusingMatrix(OUTPUT, 2);
		cout << "[Training confusion matrix]\n";
		for (int i = 0; i < N_VALIDATION_SET; i += BATCH_SIZE) {
			gotoxy(0, 0);
			cout << "[Neurons : " << neurons << ", Data #" << i + BATCH_SIZE << "]=================" << endl;
			input.copyFrom(validationData, 0, i);

			output = *network.forwardPropagation(&input); // Predict digit

			Matrix result = argMax(&output, COL);

			loss = network.rmseLoss(
				&onehot(
					&validationData.split(COL, i, BATCH_SIZE).split(ROW, N_DATASET_ATTR - 1, 1),
					OUTPUT,
					BATCH_SIZE
				)
			); // Have to calculate loss for backpropagation

			for (int acc = 0; acc < BATCH_SIZE; acc++) {
				confusingMatrix[validationData[N_DATASET_ATTR - 1][i + acc]][1]++;
				if (int(validationData[N_DATASET_ATTR - 1][i + acc]) == result[0][acc])
					confusingMatrix[validationData[N_DATASET_ATTR - 1][i + acc]][0]++;
			}
		}
		confusingMatrix.saveToFile(to_string(neurons) + "_training_confusing.txt");
		confusingMatrix.print();

		confusingMatrix(0);
		for (int i = 0; i < N_TEST_SET; i += BATCH_SIZE) {
			gotoxy(0, 0);
			cout << "[Neurons : " << neurons << ", Data #" << i + BATCH_SIZE << "]=================" << endl;
			input.copyFrom(testData, 0, i);

			output = *network.forwardPropagation(&input); // Predict digit

			Matrix result = argMax(&output, COL);

			loss = network.rmseLoss(
				&onehot(
					&testData.split(COL, i, BATCH_SIZE).split(ROW, N_DATASET_ATTR - 1, 1),
					OUTPUT,
					BATCH_SIZE
				)
			); // Have to calculate loss for backpropagation

			for (int acc = 0; acc < BATCH_SIZE; acc++) {
				confusingMatrix[testData[N_DATASET_ATTR - 1][i + acc]][1]++;
				if (int(testData[N_DATASET_ATTR - 1][i + acc]) == result[0][acc])
					confusingMatrix[testData[N_DATASET_ATTR - 1][i + acc]][0]++;
			}
		}
		confusingMatrix.saveToFile(to_string(neurons) + "_testing_confusing.txt");
		confusingMatrix.print();
	}
}