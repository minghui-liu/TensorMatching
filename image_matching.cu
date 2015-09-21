#include <iostream>
#include <ctime>
#include <curand.h>
#include <algorithm>
#include <cublas_v2.h>
#include <thrust/reduce.h>
#include <ANN/ANN.h>
#include <unistd.h>
#include "compute_feature.h"
#include "knn.h"
#include "tensor_matching.h"
#include "error_check.h"
#include <nvToolsExt.h>

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

/* Utility functions */
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

/* Kernels */
__global__
void ind2sub(int nP, int *inds, int h, int w, int *i, int *j, int *k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= h || col >= w) return;
	i[IDX2C(row,col,h)] = inds[IDX2R(row,col,w)] / (nP*nP);
	j[IDX2C(row,col,h)] = (inds[IDX2R(row,col,w)] % (nP*nP)) / nP;
	k[IDX2C(row,col,h)] = inds[IDX2R(row,col,w)] % nP;
}

// Create an m-by-n tiling of a given matrix
__global__
void repmat(double *d_A, int h, int w, double *d_B, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= h || col >= w) return;
	for(int i=0; i < m; i++) {
		for(int j=0; j < n; j++) {
			//printf("thread %d, filling %d\n", row*w+col, (row+i*h)*w+(col+j*w));
			d_B[(row+i*h)*(w*n)+(col+j*w)] = d_A[row*w+col];
		}
	}
}

// Create indH (tensor index matrix)
// note that this is a 1-D kernel plz launch a 1-D grid
__global__
void createIndH(int *T1, int nT, int nNN, int nP, int *indH) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nT) return;
	// t1(:,tmp(:))'*nP2
	for (int t = 0; t < nNN; t++) {
		indH[3*nNN*idx + t*3] = T1[idx] * nP;
		indH[3*nNN*idx + t*3+1] = T1[nT+idx] * nP;
		indH[3*nNN*idx + t*3+2] = T1[2*nT+idx] * nP;
	}
}

__global__
void addIJK(int *indH, int height, int *i, int *j, int *k) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= height) return;
	indH[idx*3] += i[idx];
	indH[idx*3+1] += j[idx];
	indH[idx*3+2] += k[idx];
}

__global__
void createValH(double *dists, int h, int w, double average, double *valH) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= h || col >= w) return;
	valH[IDX2C(row,col,h)] = exp(-dists[IDX2R(row,col,w)]/average);
}

__global__
void initMatrix(double *M, int h, int w, double value) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= h || col >= w) return;
	M[IDX2R(row,col,w)] = value;
}

void imageMatching(double *P1, int nP1, double *P2, int nP2, double *X2, char vflag, char sflag, char tflag) {		
	// Start timer
	clock_t begin = clock();

	// cuBLAS handle
	//cublasHandle_t handle;
	// Create cuBLAS context handle
	//CUBLAS_CALL(cublasCreate(&handle));
	//curandGenerator_t gen;
	// Create pseudo-random number generator
	//CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	// Set seed 
	//CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
	//srand(time(NULL));

	nvtxRangeId_t id1 = nvtxRangeStartA("P1, P2 and T1");

	// Allocate space for cordinate matrices on device
	double *d_P1, *d_P2;
	CUDA_SAFE_CALL(cudaMalloc(&d_P1, 2*nP1*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_P2, 2*nP2*sizeof(double)));
	// Copy P1 and P2 from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_P1, P1, 2*nP1*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_P2, P2, 2*nP2*sizeof(double), cudaMemcpyHostToDevice));
	
	// Print P1
	if (sflag) {
		if (vflag)	std::cout << "Outputting P1 ..." << std::endl;
		saveMatrix(P1, 2, nP1, "output/P1.mat");
		if (vflag)	std::cout << "Outputting P2 ..." << std::endl;
		saveMatrix(P2, 2, nP2, "output/P2.mat");
	}

	// Generate triangles
	int nT = nP1 * 50;
	int *T1 = (int*)malloc(3*nT*sizeof(int));
	int *d_T1;
	CUDA_SAFE_CALL(cudaMalloc(&d_T1, 3*nT*sizeof(int)));

	// Randomly assign points to triangles (currently serial)
	for (int i=0; i < 3; i++) {
		for (int j=0; j < nT; j++) {
			T1[i*nT+j] = floor( (double)rand()/RAND_MAX*nP1 );
		}
	}
	
	// Remove duplicate points
	while (1) {
		char probFound = 0;
		for (int i=0; i < 3; i++)
			for (int j=0; j < nT; j++)
				if (T1[i*nT+j] == T1[((i+1)%3)*nT+j]) {
					T1[i*nT+j] = floor((double)rand()/RAND_MAX*nP1);
					probFound = 1;
				}

		if (!probFound) {
			break;
		}
	}
	
	// Copy to device
	CUDA_SAFE_CALL(cudaMemcpy(d_T1, T1, 3*nT*sizeof(int), cudaMemcpyHostToDevice));
	
	// Print T1
	if (sflag) {
		if (vflag)	std::cout << "Outputting T1 ..." << std::endl;
		saveMatrix(T1, 3, nT, "output/T1.mat");
	}
	
	nvtxRangeEnd(id1);
	
	nvtxRangeId_t id2 = nvtxRangeStartA("F1, F2");

	// Allocate feature matrices
	double *F1 = (double*)malloc(3*nT*sizeof(double));
	double *F2 = (double*)malloc(3*nP2*nP2*nP2*sizeof(double));

	// Call ComputeFeature() in computeFeature.cpp (currently serial)
	computeFeature(P1, nP1, P2, nP2, T1, nT, F1, F2);
	
	if (sflag) {
		if (vflag)	std::cout << "Outputting F1 ..." << std::endl;
		saveMatrix(F1, 3, nT, "output/F1.mat");
		if (vflag)	std::cout << "Outputting F2 ..." << std::endl;
		saveMatrix(F2, 3, nP2*nP2*nP2, "output/F2.mat");
	}

	// Number of nearst neighbors
	int nNN = 300;
	
	/* CUDA knn code has -nan problem use ANN for now
	// Allocate space for distance matrix and index matrix
	double *dists = (double*)malloc(nNN*nT*sizeof(double));
	int *inds = (int*)malloc(nNN*nT*sizeof(int));

	// Call kNN (cuda k nearst neighbor)
	knn(F2, nP2*nP2*nP2, F1, nT, 3, nNN, dists, inds);

	// Allocate space for distance matrix and index matrix on device
	// and copy to device
	int *d_inds;
	CUDA_SAFE_CALL(cudaMalloc(&d_inds, nNN*nT*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_inds, inds, nNN*nT*sizeof(int), cudaMemcpyHostToDevice));
	double *d_dists;
	CUDA_SAFE_CALL(cudaMalloc(&d_dists, nNN*nT*sizeof(double)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dists, dists, nNN*nT*sizeof(double), cudaMemcpyHostToDevice));
	*/
	
	nvtxRangeEnd(id2);
	
	nvtxRangeId_t id3 = nvtxRangeStartA("ANN search");
	
	/* ANN (serial) */
	double *dists = (double*)malloc(nNN*nT*sizeof(double));
	int *inds = (int*)malloc(nNN*nT*sizeof(int));
	
	ANNpointArray	dataPts;
	ANNpoint		queryPt;
	ANNidxArray		nnIdx;
	ANNdistArray	nnDist;
	ANNkd_tree*		kdTree;
	
	dataPts = annAllocPts(nP2*nP2*nP2, 3);
	queryPt = annAllocPt(3);
	nnIdx = new ANNidx[nNN];
	nnDist = new ANNdist[nNN];
	
	// copy F2 into dataPts
	for (int i = 0; i < nP2*nP2*nP2; i++) {
		for (int j = 0; j < 3; j++) {
			dataPts[i][j] = F2[j*nP2*nP2*nP2+i];
		}
	}
	
	// build kd tree
	kdTree = new ANNkd_tree(dataPts, nP2*nP2*nP2, 3);
	
	// query nT points
	for (int i = 0; i < nT; i++) {
		// get query point coordinates
		queryPt[0] = F1[i];
		queryPt[1] = F1[nT+i];
		queryPt[2] = F1[2*nT+i];
		
		// search
		kdTree->annkSearch(queryPt, nNN, nnIdx, nnDist, 10);
	
		for (int j = 0; j < nNN; j++) {
			inds[j*nT+i] = nnIdx[j];
			dists[j*nT+i] = nnDist[j];
		}
	}
	
	
	// square distance to euclidean distance
	std::transform(dists, dists+nNN*nT, dists, std::ptr_fun<double, double>(sqrt));
	
	// Copy inds and dists onto device
	int *d_inds;
	CUDA_SAFE_CALL(cudaMalloc(&d_inds, nNN*nT*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_inds, inds, nNN*nT*sizeof(int), cudaMemcpyHostToDevice));
	double *d_dists;
	CUDA_SAFE_CALL(cudaMalloc(&d_dists, nNN*nT*sizeof(double)));
	CUDA_SAFE_CALL(cudaMemcpy(d_dists, dists, nNN*nT*sizeof(double), cudaMemcpyHostToDevice));
	
	/* End of ANN code */
	
	if (sflag) {
		if (vflag)	std::cout << "Outputting dists ..." << std::endl;
		saveMatrix(dists, nNN, nT, "output/dists.mat");
		if (vflag)	std::cout << "Outputting inds ..." << std::endl;
		saveMatrix(inds, nNN, nT, "output/inds.mat");
	}
	
	nvtxRangeEnd(id3);
	

	nvtxRangeId_t id4 = nvtxRangeStartA("i, j and k");

	// Allocate space for i, j, k
	int *i, *j, *k;
	i = (int*)malloc(nNN*nT*sizeof(int));
	j = (int*)malloc(nNN*nT*sizeof(int));
	k = (int*)malloc(nNN*nT*sizeof(int));

	// Allocate space for i, j, k on device
	int *d_i, *d_j, *d_k; 
	CUDA_SAFE_CALL(cudaMalloc(&d_i, nNN*nT*sizeof(int))); 
	CUDA_SAFE_CALL(cudaMalloc(&d_j, nNN*nT*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&d_k, nNN*nT*sizeof(int)));

	// ind2sub
	// Convert linear indices into i, j, k subscripts
	dim3 dimBlock(32, 32);
	dim3 dimGrid( (nT + dimBlock.x - 1)/dimBlock.x, (nNN + dimBlock.y - 1)/dimBlock.y );
	ind2sub<<<dimGrid, dimBlock>>>(nP2, d_inds, nNN, nT, d_i, d_j, d_k);
	CUDA_CHECK_ERROR();

	// Copy results back into i, j, k
	CUDA_SAFE_CALL(cudaMemcpy(i, d_i, nNN*nT*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(j, d_j, nNN*nT*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(k, d_k, nNN*nT*sizeof(int), cudaMemcpyDeviceToHost));

	// Print i, j ,k
	if (sflag) {
		if (vflag)	std::cout << "Outputting i ..." << std::endl;
		saveMatrix(i, 1, nNN*nT, "output/i.mat");
		if (vflag)	std::cout << "Outputting j ..." << std::endl;
		saveMatrix(j, 1, nNN*nT, "output/j.mat");
		if (vflag)	std::cout << "Outputting k ..." << std::endl;
		saveMatrix(k, 1, nNN*nT, "output/k.mat");
	}
	
	nvtxRangeEnd(id4);
	
	nvtxRangeId_t id5 = nvtxRangeStartA("indH and valH");

	// Create indH (tensor index)
	int *indH = (int*)malloc(nT*nNN*3*sizeof(int));
	int *d_indH;
	CUDA_SAFE_CALL(cudaMalloc(&d_indH, nT*nNN*3*sizeof(int)));
	// Launch createIndH kernel
	dimBlock = dim3(1024);
	dimGrid = dim3((nT + dimBlock.x - 1)/dimBlock.x);
	createIndH<<<dimGrid, dimBlock>>>(d_T1, nT, nNN, nP2, d_indH);
	CUDA_CHECK_ERROR();

	// Launch addIJK kernel
	dimGrid = dim3((nT*nNN + dimBlock.x - 1)/dimBlock.x);
	addIJK<<<dimGrid, dimBlock>>>(d_indH, nT*nNN, d_i, d_j, d_k);
	CUDA_CHECK_ERROR();
	// Copy indH to host
	CUDA_SAFE_CALL(cudaMemcpy(indH, d_indH, nT*nNN*3*sizeof(int), cudaMemcpyDeviceToHost));

	// Create valH (tensor values)
	// mean(dists(:))
	double sum = thrust::reduce(dists, dists + nNN*nT);
	double average = sum / (nNN*nT);

	if (vflag) {
		std::cout << "sum of dists = " << sum << std::endl;
		std::cout << "average of dists = " << average << std::endl;
	}
	

	// Allocate valH on host
	double *valH = (double*)malloc(nNN*nT*sizeof(double));
	// Allocate valH on device
	double *d_valH;
	CUDA_SAFE_CALL(cudaMalloc(&d_valH, nNN*nT*sizeof(double)));
	// Launch createValH kernel
	dimBlock = dim3(32, 32);
	dimGrid = dim3( (nT + dimBlock.x - 1)/dimBlock.x, (nNN + dimBlock.y - 1)/dimBlock.y );
	createValH<<<dimGrid, dimBlock>>>(d_dists, nNN, nT, average, d_valH);
	CUDA_CHECK_ERROR();
	// Copy valH to host
	CUDA_SAFE_CALL(cudaMemcpy(valH, d_valH, nNN*nT*sizeof(double), cudaMemcpyDeviceToHost));

	// Print indH and valH
	if (sflag) {
		if (vflag)	std::cout << "Outputting indH ..." << std::endl;
		saveMatrix(indH, nT*nNN, 3, "output/indH.mat");
		if (vflag)	std::cout << "Outputting valH ..." << std::endl;
		saveMatrix(valH, nNN*nT, 1, "output/valH.mat");
	}

	
	nvtxRangeEnd(id5);
	nvtxRangeId_t id6 = nvtxRangeStartA("X and X2");

	// Allocate space for X
	double *d_X;
	CUDA_SAFE_CALL(cudaMalloc(&d_X, nP2*nP1*sizeof(double)));
	// Initilize values to 1/nP2
	dimGrid = dim3( (nP1 + dimBlock.x - 1)/dimBlock.x, (nP2 + dimBlock.y - 1)/dimBlock.y );
	// Launch initMatrix kernel
	initMatrix<<<dimGrid, dimBlock>>>(d_X, nP2, nP1, (double)(1.0/nP2));
	CUDA_CHECK_ERROR();

	// Create X2;
	double *d_X2;
	CUDA_SAFE_CALL(cudaMalloc(&d_X2, nP2*nP1*sizeof(double)));
		
	// Create score
	double score = -1;
	double *d_score;
	CUDA_SAFE_CALL(cudaMalloc(&d_score, sizeof(double)));
	
	nvtxRangeEnd(id6);
	nvtxRangeId_t id7 = nvtxRangeStartA("Tensor matching");

	// Call tensor matching
	tensorMatching(d_X, nP2, nP1, d_indH, d_valH, nNN*nT, 100, 1, 2, d_X2, d_score);
	
	nvtxRangeEnd(id7);
	
	// Copy result back to host
	CUDA_SAFE_CALL(cudaMemcpy(X2, d_X2, nP2*nP1*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&score, d_score, sizeof(double), cudaMemcpyDeviceToHost));
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
	if (sflag) {
		if (vflag)	std::cout << "Outputting X2 ..." << std::endl;
		saveMatrix(X2, nP2, nP1, "output/X2.mat");
	}
	if (vflag)	std::cout << "score = " << score << std::endl;
	
	//CURAND_CALL(curandDestroyGenerator(gen));
	//cublasDestroy(handle);
	
	// Stop timer
	clock_t end = clock();
	double timespent = (double)(end - begin) / (double)CLOCKS_PER_SEC;
	
	if (tflag)	std::cout << "Time elapsed: " << timespent << std::endl;
	
	
	
}


