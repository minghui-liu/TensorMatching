#include <iostream>
#include <fstream>
#include <cstdlib>
#include "../compute_feature.h"

using namespace std;

void saveMatrix(double *A, int h, int w, char *filename) {
	FILE *fp;
	fp = fopen(filename, "w");
	for (int i=0; i < h; i++) {
		for (int j=0; j < w; j++) {
			fprintf(fp, "%.4f ", A[i*w+j]); 
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void printMatrix(double *h_M, int height, int width) {
	for (int i=0; i < height; i++) {
		for (int j=0; j < width; j++)
			std::cout << h_M[i * width + j] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void printMatrix(int *h_M, int height, int width) {
	for (int i=0; i < height; i++) {
		for (int j=0; j < width; j++)
			std::cout << h_M[i * width + j] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	int nP = 10;
	int nT = nP * 50;
	ifstream inStream;
	double *P1, *P2;
	int *T1;
	double *F1, *F2;
	
	P1 = new double[2*nP];
	P2 = new double[2*nP];
	T1 = new int[3*nT];
	F1 = new double[3*nT];
	F2 = new double[3*nP*nP*nP];

	inStream.open("input/P1.mat", ios::in);	
	if (!inStream) {
		cerr << "Cannot open P1.mat file\n";
		exit(1);
	}
	for (int i = 0; i < 2*nP; i++)
		if(!(inStream >> P1[i])) return -1;
	inStream.close();
	
	inStream.open("input/P2.mat", ios::in);	
	if (!inStream) {
		cerr << "Cannot open P1.mat file\n";
		exit(1);
	}
	for (int i = 0; i < 2*nP; i++)
		if(!(inStream >> P2[i])) return -2;
	inStream.close();
	
	inStream.open("input/T1.mat", ios::in);	
	if (!inStream) {
		cerr << "Cannot open T1.mat file\n";
		exit(1);
	}
	for (int i = 0; i < 3*nT; i++)
		if(!(inStream >> T1[i])) return -3;
	inStream.close();

	cout << "P1" << endl;
	printMatrix(P1, 2, nP);
	cout << "P2" << endl;
	printMatrix(P2, 2, nP);
	cout << "T1" << endl;
	printMatrix(T1, 3, nT);
	
	computeFeature(P1, nP, P2, nP, T1, nT, F1, F2);
	
	cout << "F1" << endl;
	printMatrix(F1, 3, nT);
	cout << "F2" << endl;
	printMatrix(F2, 3, nP*nP*nP);
	
	return EXIT_SUCCESS;
}

