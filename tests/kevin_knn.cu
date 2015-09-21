#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include "cublas_v2.h"
#include "error_check.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

using namespace std;
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

void printMatrix(double *Pts, int h, int w)	// print point
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cout << Pts[i*w+j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
}

void printMatrix(int*Pts, int h, int w)	// print point
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cout << Pts[i*w+j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
}

//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//


__global__ void cuComputeNorm(double *mat, int height, int width, double *norm){
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
	//cublasHandle_t handle;
	//CUBLAS_CALL(cublasCreate(&handle));

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

int main(int argc, char **argv) {
	// Parse command-line arguments
	char vflag = 0;
	char tflag = 0;
	int query_nb, ref_nb, dim, k;
	char opt;
	
	// process command-line options
	while ((opt = getopt(argc, argv, "vtr:q:d:k:")) != -1) {
		switch (opt) {
			case 'v':
				vflag = 1;	
				break;
			case 't':
				tflag = 1;
				break;
			case 'q':
				query_nb = atoi(optarg);
				break;
			case 'r':
				ref_nb = atoi(optarg);
				break;
			case 'd':
				dim = atoi(optarg);
				break;
			case 'k':
				k = atoi(optarg);
				break;
			default:
				abort();
		}
	}
	
	if (argc < 8) {
	  	fprintf(stderr, "Must specify q r d and k.\n");
		return EXIT_FAILURE;
	}

	// print non-opt args
	for (int index = optind; index < argc; index++)
		printf ("Non-option argument %s\n", argv[index]);

	// std::cout << "q=" << query_nb << ", r=" << ref_nb << ", d=" << dim << ", k=" << k << ", v=" << (int)vflag << ", t=" << (int)tflag << std::endl;

	double *dataPts;			// data points
	double *queryPts;			// query point
	int *inds;					// near neighbor indices
	double *dists;				// near neighbor distances

	dataPts = new double[dim*ref_nb];		// allocate query point
	queryPts = new double[dim*query_nb];	// allocate data points
	inds = new int[query_nb*k];			// allocate near neigh indices
	dists = new double[query_nb*k];		// allocate near neighbor dists

	srand(time(NULL));
	// Generate data points
	for (int i = 0; i < dim * ref_nb; i++)
		dataPts[i] = (double)rand()/RAND_MAX;
	
	// Generate query points
	for (int i = 0; i < dim * query_nb; i++)
		queryPts[i] = (double)rand()/RAND_MAX;

	knn(dataPts, ref_nb, queryPts, query_nb, dim, k, dists, inds);

	if (vflag) {
		cout << "Data Points" << endl;
		printMatrix(dataPts, dim, ref_nb);
		cout << "Query Points" << endl;
		printMatrix(queryPts, dim, query_nb);

		knn(dataPts, ref_nb, queryPts, query_nb, dim, k, dists, inds);
		
		cout << "Distances" << endl;
		printMatrix(dists, k, query_nb);
		cout << "Indices" << endl;
		printMatrix(inds, k, query_nb);
	}
	

	return EXIT_SUCCESS;
}




