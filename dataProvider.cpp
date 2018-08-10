#include "dataProvider.h"
#include <string>

int imgMagicNumber, labelMagicNumber;
int numberOfLabels, numberOfImages;
int numberOfRows, numberOfColumns;

void ShowConsoleCursor(bool showFlag)
{
	HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);

	CONSOLE_CURSOR_INFO     cursorInfo;

	GetConsoleCursorInfo(out, &cursorInfo);
	cursorInfo.bVisible = showFlag; // set the cursor visibility
	SetConsoleCursorInfo(out, &cursorInfo);
}

void gotoxy(int x, int y) {
	COORD pos = { x, y };
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
}

COORD getxy() {
	COORD pos;
	CONSOLE_SCREEN_BUFFER_INFO buf;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &buf);
	pos.X = buf.dwCursorPosition.X;
	pos.Y = buf.dwCursorPosition.Y;
	return pos;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void drawMatrix(string name, Matrix* input) {
	input->retrieveDataFromDevice();
	for (int j = 0; j < (*input).row(); j++) {
		gotoxy(0, 0);
		printf("row : %d, col : %d \n", (*input).row(), (*input).col());
		printf("%s[%d]\n", name.c_str(), j);
		for (int i = 0; i < (*input).col(); i++) {
			printf("%4d ", (int)(*input)[j][i]);
			if ((i + 1) % numberOfColumns == 0)
				printf("\n");
		}
	}
}

Matrix mnistDataProvider(string img_path, string label_path, int dataSize) {
	cout << "Start loading data" << endl;
	ifstream imgFileStream;
	ifstream labelFileStream;
	imgFileStream.open(img_path, ios::binary);
	labelFileStream.open(label_path, ios::binary);
	
	if (!imgFileStream.is_open()) {
		cout << "Cannot open file [" << img_path << "]\n";
		exit(EXIT_FAILURE);
	}
	if (!labelFileStream.is_open()) {
		cout << "Cannot open file [" << label_path << "]\n";
		exit(EXIT_FAILURE);
	}

	labelFileStream.read((char*)&labelMagicNumber, sizeof(int));
	labelFileStream.read((char*)&numberOfLabels, sizeof(int));
	imgFileStream.read((char*)&imgMagicNumber, sizeof(int));
	imgFileStream.read((char*)&numberOfImages, sizeof(int));
	imgFileStream.read((char*)&numberOfRows, sizeof(int));
	imgFileStream.read((char*)&numberOfColumns, sizeof(int));

	labelMagicNumber = ReverseInt(labelMagicNumber);
	numberOfLabels = ReverseInt(numberOfLabels);
	imgMagicNumber = ReverseInt(imgMagicNumber);
	numberOfImages = ReverseInt(numberOfImages);
	numberOfRows = ReverseInt(numberOfRows);
	numberOfColumns = ReverseInt(numberOfColumns);

	Matrix data(dataSize, N_DATASET_ATTR);

	COORD pos = getxy();
	unsigned char input = 0;
	for (int i = 0; i < dataSize; i++) { // Read while not eof
		gotoxy(0, pos.Y);
		printf("[%d/%d]            ", (i + 1), dataSize);
		for (int j = 0; j < N_DATASET_ATTR - 1; j++) {
			imgFileStream.read((char*)&input, sizeof(input));
			data[i][j] = (double)input / 255.0;
		}
		labelFileStream.read((char*)&input, sizeof(input));
		data[i][N_DATASET_ATTR - 1] = (int)input;
	}
	imgFileStream.close();
	labelFileStream.close();
	
	printf("\nComplete load file [%s]\n", img_path.c_str());
	printf("Complete load file [%s]\n", label_path.c_str());
	data.allocDevData();


	return data;
}

Matrix twoMoonDataProvider(string path, int dataSize) {
	cout << "Start loading data" << endl;
	ifstream fileStream;
	fileStream.open(path, ios::binary);

	if (!fileStream.is_open()) {
		cout << "Cannot open file [" << path << "]\n";
		exit(EXIT_FAILURE);
	}

	Matrix data(dataSize, N_DATASET_ATTR);

	COORD pos = getxy();
	unsigned char input = 0;
	for (int i = 0; i < dataSize; i++) { // Read while not eof
		gotoxy(0, pos.Y);
		printf("[%d/%d]            ", (i + 1), dataSize);
		for (int j = 0; j < N_DATASET_ATTR; j++) {
			fileStream >> data[i][j];
		}
	}
	fileStream.close();

	printf("\nComplete load file [%s]\n", path.c_str());

	data.allocDevData();

	return data;
}