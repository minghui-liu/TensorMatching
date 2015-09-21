#include <iostream>
#include <cstdio>
#include <unistd.h>
#include "knn.h"

__global__
void adjust_range(double *d_M, int height, int width, double mult, double off) { 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= height || col >= width) return;
	int idx = row * width + col;
	d_M[idx] *= mult;
	d_M[idx] += off;
}

void saveMatrix(double *A, int h, int w, char *filename) {
	FILE *fp;
	fp = fopen(filename, "w");
	for (int i=0; i < h; i++) {
		for (int j=0; j < w; j++) {
			fprintf(fp, "%.16f ", A[i*w+j]); 
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void saveMatrix(int *A, int h, int w, char *filename) {
	FILE *fp;
	fp = fopen(filename, "w");
	for (int i=0; i < h; i++) {
		for (int j=0; j < w; j++) {
			fprintf(fp, "%d ", A[i*w+j]); 
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

int main(int argc, char *argv[]) {

	// Parse command-line arguments
	char vflag = 0;
	char sflag = 0;
	char tflag = 0;
	char opt;
	
	// process command-line options
	while ((opt = getopt(argc, argv, "vst")) != -1) {
		switch (opt) {
			case 'v':
				vflag = 1;	
				break;
			case 's':
				sflag = 1;
				break;
			case 't':
				tflag = 1;
				break;
			case '?':
				if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				abort();
		}
	}
	
	if (argc < 2 || optind == argc) {
	  	fprintf(stderr, "Must specify test size.\n");
		return EXIT_FAILURE;
	}
	
	// test size
	int nP1 = atoi(argv[optind++]);
	int nP2 = nP1;

	// printf("vflag = %d, sflag = %d, test_size = %d\n", vflag, sflag, nP1);
	// print non-opt args
	for (int index = optind; index < argc; index++)
		printf ("Non-option argument %s\n", argv[index]);
		
	int nT = nP1 * 50;
	int nNN = 300;
	
	// Allocate feature matrices
	double *F1 = (double*)malloc(3*nT*sizeof(double));
	double *F2 = (double*)malloc(3*nP2*nP2*nP2*sizeof(double));
	
	//
	double *dist = (double*)malloc(nNN*nT*sizeof(double));
	int *ind = (int*)malloc(nNN*nT*sizeof(int));
	
	// set seed
	srand(time(NULL));
	
	for (int i = 0; i < 3*nT; i++)
		F1[i] = (double)rand()/RAND_MAX - 0.5;
	for (int i = 0; i < 3*nP2*nP2*nP2; i++)
		F2[i] = (double)rand()/RAND_MAX - 0.5;
	
	for (int i = 0; i < nP2*nP2; i++) {
		F2[rand() % (3*nP2*nP2*nP2)] = 10;
	}
		
	if (sflag) {
		if (vflag) std::cout << "Outputting F1 ..." << std::endl;
		saveMatrix(F1, 3, nT, "F1.mat");
		if (vflag) std::cout << "Outputting F2 ..." << std::endl;
		saveMatrix(F2, 3, nP2*nP2*nP2, "F2.mat");
	}
	
	// Start timer
	clock_t begin = clock();
	
	knn(F2, nP2*nP2*nP2, F1, nT, 3, nNN, dist, ind);
	
	// Stop timer
	clock_t end = clock();
	double timespent = (double)(end - begin) / (double)CLOCKS_PER_SEC;
	
	if (sflag) {
		if (vflag) std::cout << "Outputting dist ..." << std::endl;
		saveMatrix(dist, nNN, nT, "dists.mat");
		if(vflag) std::cout << "Outputting ind ..." << std::endl;
		saveMatrix(ind, nNN, nT, "ind.mat");
	}
	
	if (tflag) std::cout << "Time: " << timespent << std::endl;
	
}























