#include <iostream>
#include <unistd.h>
#include <curand.h>
#include <cublas_v2.h>
#include "error_check.h"

extern void saveMatrix(double *A, int h, int w, char *filename);
extern void imageMatching(double *P1, int nP1, double *P2, int nP2, double *X2, char vflag, char sflag, char tflag);

__global__
void adjust_range(double *d_M, int height, int width, double mult, double off) { 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= height || col >= width) return;
	int idx = row * width + col;
	d_M[idx] *= mult;
	d_M[idx] += off;
}

__global__
void distort(double *d_M, int height, int width, double d) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= height || col >= width) return;
	int idx = row * width + col;
	d_M[idx] += d;
}

__global__
void merge(double *X, double *Y, double *M, int height, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= height || col >= width) return;
	if (row == 0) {
		M[row*width+col] = X[col];
	} else {
		M[row*width+col] = Y[col];
	}
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
			
	// cuBLAS handle
	cublasHandle_t handle;
	// Create cuBLAS context handle
	CUBLAS_CALL(cublasCreate(&handle));
	curandGenerator_t gen;
	// Create pseudo-random number generator
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	// Set seed 
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
	srand(time(NULL));


	// Allocate space for cordinate matrices
	double *P1 = (double*)malloc(2*nP1*sizeof(double));
	double *P2 = (double*)malloc(2*nP1*sizeof(double));
	double *d_P1, *d_P2;
	CUDA_SAFE_CALL(cudaMalloc(&d_P1, 2*nP1*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_P2, 2*nP2*sizeof(double)));

	// Array of X and Y cordinates on device
	double *d_PX, *d_PY;
	CUDA_SAFE_CALL(cudaMalloc(&d_PX, nP1*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_PY, nP1*sizeof(double)));

	// Generate n doubles on device 
	CURAND_CALL(curandGenerateUniformDouble(gen, d_PX, nP1));
	CURAND_CALL(curandGenerateUniformDouble(gen, d_PY, nP1));
	// Adjust range of random numbers
	dim3 dimBlock(1024, 1);
	dim3 dimGrid( (nP1 + dimBlock.x - 1)/dimBlock.x, (1 + dimBlock.y - 1)/dimBlock.y );
	adjust_range<<<dimGrid, dimBlock>>>(d_PX, 1, nP1, 2, -1);
	CUDA_CHECK_ERROR();
	adjust_range<<<dimGrid, dimBlock>>>(d_PY, 1, nP1, 2, -1);
	CUDA_CHECK_ERROR();

	// Merge result into one 2*nP matrix (d_P1)
	dimBlock = dim3(512, 2);
	dimGrid = dim3( (nP1 + dimBlock.x - 1)/dimBlock.x, (2 + dimBlock.y - 1)/dimBlock.y );
	merge<<<dimGrid, dimBlock>>>(d_PX, d_PY, d_P1, 2, nP1);
	CUDA_CHECK_ERROR();
	// Copy result back to P1
	CUDA_SAFE_CALL(cudaMemcpy(P1, d_P1, 2*nP1*sizeof(double), cudaMemcpyDeviceToHost));

	// Print P1
	if (sflag) {
		if (vflag)	std::cout << "Outputting P1 ..." << std::endl;
		saveMatrix(P1, 2, nP1, "output/P1.mat");
	}

	// Scale and rotation angle
	double scale = 0.5 + (double)rand() / RAND_MAX;
	double theta = 0.5 * (double)rand() / RAND_MAX;
	// Cos and sin value
	double c = cos(theta);
	double s = -sin(theta);

	if (vflag) {
		std::cout << "Rotating and scaling ..." << std::endl;
		std::cout << "scale = " << scale << std::endl;
		std::cout << "theta = " << theta << std::endl;
		std::cout << "c = " << c << std::endl;
		std::cout << "s = " << s << std::endl;
	}
	// Rotate
	CUBLAS_CALL(cublasDrot(handle, nP1, d_PX, 1, d_PY, 1, &c, &s));

	// Merge into d_P2
	merge<<<dimGrid, dimBlock>>>(d_PX, d_PY, d_P2, 2, nP1);
	CUDA_CHECK_ERROR();
	// Copy results back to P2
	CUDA_SAFE_CALL(cudaMemcpy(P2, d_P2, 2*nP2*sizeof(double), cudaMemcpyDeviceToHost));

	// Print P2
	if (sflag) {
		if (vflag)	std::cout << "Outputting P2 ..." << std::endl;
		saveMatrix(P2, 2, nP2, "output/P2.mat");
	}
	
	double *X2 = (double*)malloc(nP1*nP2*sizeof(double));
	imageMatching(P1, nP1, P2, nP2, X2, vflag, sflag, tflag);
	

	// Free some device memory
	CUDA_SAFE_CALL(cudaFree(d_PX));
	CUDA_SAFE_CALL(cudaFree(d_PY));
	CUDA_SAFE_CALL(cudaFree(d_P1));
	CUDA_SAFE_CALL(cudaFree(d_P2));

}

