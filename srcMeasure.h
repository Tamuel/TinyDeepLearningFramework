#ifndef SRC_MEASURE_H
#define SRC_MEASURE_H

#include <iostream>
#include <chrono>
#include <cmath>
#include <string>

#define NUMBER_OF_TIMER 10
#define DECIMAL 10

using namespace std;

class SrcMeasure {
private:

	// Chrono supported from C++ 11
	chrono::time_point<chrono::steady_clock> start_time[NUMBER_OF_TIMER];
	chrono::time_point<chrono::steady_clock> end_time[NUMBER_OF_TIMER];

public:
	SrcMeasure() {

	}

	// Start the timerIndex th timer
	void startTime(int timerIndex) {
		if (DEBUG_M)
			start_time[timerIndex] = chrono::high_resolution_clock::now();
	}

	// Return elapsed time of timerIndex th timer
	dtype endTime(int timerIndex, string s) {
		if (DEBUG_M) {
			end_time[timerIndex] = chrono::high_resolution_clock::now();
			long long nsTimeCount = chrono::duration_cast<chrono::nanoseconds>(end_time[timerIndex] - start_time[timerIndex]).count();
			dtype elapsedTime = nsTimeCount / pow(10, 9);
			cout << "[" << s << "] Time : " << f_to_s(elapsedTime, 8, 6, -6) << endl;
			return elapsedTime;
		}
		return 0;
	}

	string f_to_s(dtype floatingNumber, int length, int precision, int floor) {
		char* fChar = new char[length + 1];
		float fractionalPart = floatingNumber - (long)floatingNumber;

		int _length = length;
		int _precision = precision;
		string _expression = "lf";

		floatingNumber = floatingNumber / pow(DECIMAL, floor);
		floatingNumber = long long(floatingNumber);
		floatingNumber *= pow(DECIMAL, floor);

		if (abs(floatingNumber) >= pow(DECIMAL, length)) {
			_length = length - 3;
			_precision = length - 7;
			_expression = "e";
		}
		else if (abs(floatingNumber) >= pow(DECIMAL, length - 2)) {
			_precision = 0;
		}

		if (floor >= 0)
			_precision = 0;

		string format = "%" + to_string(_length) + "." + to_string(_precision) + _expression;
		snprintf(fChar, length + 1, format.c_str(), floatingNumber);

		string out(fChar);

		delete[] fChar;

		return out;
	}
};

#endif