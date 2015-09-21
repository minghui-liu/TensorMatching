#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>
#include "cublas_v2.h"
#include "error_check.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

//-----------------------------------------------------------------------------------------------//
//                                           Utilities                                           //
//-----------------------------------------------------------------------------------------------//

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

//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//


__global__ void cuComputeNorm(double *mat, int height, int width, double *norm) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= width) return;

	double val, sum = 0;
	for (int i = 0; i < height; i++) {
		val = mat[IDX2R(i,idx,width)];
		sum += val * val;
	}
	norm[idx] = sum;
}

__global__ void cuAddRNorm(double *dist, int height, int width, double *ref_norm) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	__shared__ double shared_vec[32];
	shared_vec[threadIdx.x] = ref_norm[x];
	__syncthreads();
	dist[IDX2R(y,x,width)] += shared_vec[threadIdx.x];
}


/* Scalar * A' * B */
__global__ void cuMatrixMult(double scalar, double *A, double *B, int m, int n, int k, double *C) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n || y >= m) return;
	
	double sum;
	for (int i = 0; i < k; i++) {
		sum += A[IDX2R(i,y,m)] * B[IDX2R(i,x,n)];
	}
	C[IDX2R(y,x,n)] = scalar * sum;

}


__global__ void cuInsertionSort(double *QtR, int *QtRind, int height, int width, double *dist, int *ind, int nNN) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= height) return;

	// Init ind
	for (int i = 0; i < width; i++) 
		QtRind[IDX2R(idx,i,width)] = i;

	for (int i = 1; i < width; i++) {
		double x = QtR[IDX2R(idx,i,width)];
		int j = i;
		while (j > 0 && QtR[IDX2R(idx,j-1,width)] > x) {
			QtR[IDX2R(idx,j,width)] = QtR[IDX2R(idx,j-1,width)];
			QtRind[IDX2R(idx,j,width)] = QtRind[IDX2R(idx,j-1,width)];
			j--;
		}
		QtR[IDX2R(idx,j,width)] = x;
		QtRind[IDX2R(idx,j,width)] = i;
	}

	// Copy to dist and ind
	for (int i = 0; i < nNN; i++) {
		dist[IDX2R(i,idx,height)] = QtR[IDX2R(idx,i,width)];
		ind[IDX2R(i,idx,height)] = QtRind[IDX2R(idx,i,width)];
	}

}


__global__ void cuAddQNormAndSqrt(double *dist, int height, int width, double *query_norm) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	dist[IDX2R(y,x,width)] = sqrt(dist[IDX2R(y,x,width)] + query_norm[x]);
}



//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS                                      //
//-----------------------------------------------------------------------------------------------//

int knn(double* ref_pts, int ref_width, double* query_pts, int query_width, int dim, int nNN, double* dist, int* ind) {
	// cuBLAS handle
	cublasHandle_t handle;
	CUBLAS_CALL(cublasCreate(&handle));

	// Allocate space for reference points and query points on device
	double *d_ref_pts, *d_query_pts;
	CUDA_SAFE_CALL(cudaMalloc(&d_ref_pts, dim*ref_width*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_query_pts, dim*query_width*sizeof(double)));
	// Copy reference points and query points from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_ref_pts, ref_pts, dim*ref_width*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_query_pts, query_pts, dim*query_width*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate space for reference norm and query norm matrix
	double *d_ref_norm, *d_query_norm;
	CUDA_SAFE_CALL(cudaMalloc(&d_ref_norm, ref_width*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_query_norm, query_width*sizeof(double)));

	// Compute reference norm
	dim3 dimBlock(1024);
	dim3 dimGrid((ref_width + dimBlock.x - 1)/dimBlock.x);
	cuComputeNorm<<<dimGrid, dimBlock>>>(d_ref_pts, dim, ref_width, d_ref_norm);
	CUDA_CHECK_ERROR();

	// Compute query norm
	dimGrid = dim3((query_width + dimBlock.x - 1)/dimBlock.x);
	cuComputeNorm<<<dimGrid, dimBlock>>>(d_query_pts, dim, query_width, d_query_norm);
	CUDA_CHECK_ERROR();

	// Allocate space for Q'*R and index matrix
	double *d_QtR;
	int *d_QtRind;
	CUDA_SAFE_CALL(cudaMalloc(&d_QtR, query_width*ref_width*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_QtRind, query_width*ref_width*sizeof(int)));

	/*// Compute Q'*R using cublas
	double alpha = -2.0;
	double beta = 0.0;
	CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, query_width, ref_width, dim, &alpha, d_query_pts, query_width, d_ref_pts, ref_width, &beta, d_QtR, ref_width));
	*/
	
	// Compute Q'R using kernel
	dimBlock = dim3(32,32);
	dimGrid = dim3((ref_width + dimBlock.x - 1)/dimBlock.x, (query_width + dimBlock.y - 1)/dimBlock.y);
	cuMatrixMult<<<dimGrid, dimBlock>>>(-2.0, d_query_pts, d_ref_pts, query_width, ref_width, dim, d_QtR);
	CUDA_CHECK_ERROR();

	// Add R norm to QtR
	dimBlock = dim3(32,32);
	dimGrid = dim3((ref_width + dimBlock.x - 1)/dimBlock.x);
	cuAddRNorm<<<dimGrid, dimBlock>>>(d_QtR, query_width, ref_width, d_ref_norm);
	CUDA_CHECK_ERROR();

	// Allocate space for distance and index matrix
	double *d_dist;
	int *d_ind;
	CUDA_SAFE_CALL(cudaMalloc(&d_dist, nNN*query_width*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc(&d_ind, nNN*query_width*sizeof(int)));

	// Insertion sort each column
	dimBlock = dim3(1024);
	dimGrid = dim3((query_width + dimBlock.x - 1)/dimBlock.x);
	cuInsertionSort<<<dimGrid, dimBlock>>>(d_QtR, d_QtRind, query_width, ref_width, d_dist, d_ind, nNN);
	CUDA_CHECK_ERROR();

	// Add R norm and square root
	dimBlock = dim3(32,32);
	dimGrid = dim3((ref_width + dimBlock.x - 1)/dimBlock.x, (query_width + dimBlock.y - 1)/dimBlock.y);
	cuAddQNormAndSqrt<<<dimGrid, dimBlock>>>(d_dist, nNN, query_width, d_query_norm);
	CUDA_CHECK_ERROR();

	// Copy dist and ind to host
	CUDA_SAFE_CALL(cudaMemcpy(dist, d_dist, nNN*query_width*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(ind, d_ind, nNN*query_width*sizeof(int), cudaMemcpyDeviceToHost));

	// done
	return 0;
}


//-----------------------------------------------------------------------------------------------//
//                                MATLAB INTERFACE & C EXAMPLE                                   //
//-----------------------------------------------------------------------------------------------//


/**
  * Example of use of kNN search CUDA.
  */

int main(void) {
	
	// Variables and parameters
	double* ref;                 // Pointer to reference point array
	double* query;               // Pointer to query point array
	double* dist;                // Pointer to distance array
	int*   ind;                 // Pointer to index array
	
	int    ref_nb     = 100;   // Reference point number
	int    query_nb   = 5;   // Query point number
	int    dim        = 3;     // Dimension of points
	int    k          = 10;     // Nearest neighbors to consider
	
	int iterations = 1;
	
	// Memory allocation
	ref    = (double *) malloc(ref_nb   * dim * sizeof(double));
	query  = (double *) malloc(query_nb * dim * sizeof(double));
	dist   = (double *) malloc(query_nb * k * sizeof(double));
	ind    = (int *)   malloc(query_nb * k * sizeof(int));
	
	// Init 
	srand(time(NULL));
	for (int i=0; i < ref_nb * dim; i++) ref[i] = (double)rand() / (double)RAND_MAX;
	for (int i=0; i < query_nb * dim; i++) query[i] = (double)rand() / (double)RAND_MAX;
	
	// Variables for duration evaluation
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	
	// Display informations
	printf("Number of reference points      : %6d\n", ref_nb  );
	printf("Number of query points          : %6d\n", query_nb);
	printf("Dimension of points             : %4d\n", dim     );
	printf("Number of neighbors to consider : %4d\n", k       );
	printf("Processing kNN search           :"                );
	
	saveMatrix(query, dim, query_nb, "query.mat");
	saveMatrix(ref, dim, ref_nb, "ref.mat");
	
	// Call kNN search CUDA
	cudaEventRecord(start, 0);
	for (int i=0; i<iterations; i++)
		knn(ref, ref_nb, query, query_nb, dim, k, dist, ind);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time/1000, iterations, elapsed_time/(iterations*1000));
	
	saveMatrix(dist, k, query_nb, "dist.mat");
	saveMatrix(ind, k, query_nb, "ind.mat");
	
	// Destroy cuda event object and free memory
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(ind);
	free(dist);
	free(query);
	free(ref);
}


