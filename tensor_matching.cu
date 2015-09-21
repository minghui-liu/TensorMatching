#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include "error_check.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

extern void saveMatrix(double*, int, int, char*);
extern __global__ void initMatrix(double *M, int h, int w, double value);


__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__
void calcScore(int *d_indH3, double *d_valH3, int Nt3, int sparse, double *d_score, double *d_Xtemp, double *d_Xout) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Nt3) return;
	
	if (sparse == 1)
		d_score[idx] = d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]];
	else
		d_score[idx] = 1;
	
	double temp0 = d_score[idx] * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]];
	double temp1 = d_score[idx] * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,2,3)]] * d_Xout[d_indH3[IDX2R(idx,0,3)]];
	double temp2 = d_score[idx] * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]];
	
	atomicAdd(&d_Xtemp[d_indH3[IDX2R(idx,0,3)]], temp0);
	atomicAdd(&d_Xtemp[d_indH3[IDX2R(idx,1,3)]], temp1);
	atomicAdd(&d_Xtemp[d_indH3[IDX2R(idx,2,3)]], temp2);
}

__global__
void lastIteration(int *d_indH3, double *d_Xout, int Nt3, double *d_score, double *d_ScoreOut) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Nt3) return;
	
	d_score[idx] = d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]];

}

__global__
void threeTimesSquare(double *d_score, int Nt3) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Nt3) return;

	d_score[idx] = 3 * d_score[idx] * d_score[idx];

}

__global__
void stocSingle(double *d_Xtemp, int N1, int N2, double *d_Xout) {
	int n2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (n2 >= N2) return;

	double Xnorm = 0;
	for (int n1 = 0; n1 < N1; n1++)
		Xnorm += d_Xtemp[IDX2R(n1,n2,N2)] * d_Xtemp[IDX2R(n1,n2,N2)];
	Xnorm = sqrt(Xnorm);
	for (int n1 = 0; n1 < N1; n1++)
		if (Xnorm !=0 )
				d_Xout[IDX2R(n1,n2,N2)] = d_Xtemp[IDX2R(n1,n2,N2)] / Xnorm;
}

__global__
void stocDouble(double *d_Xtemp, int N1, int N2, double *d_Xout) {
	int n2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (n2 >= N2) return;

	double Xnorm = 0;
	for (int n1 = 0; n1 < N1; n1++)
		Xnorm += d_Xtemp[IDX2R(n1,n2,N2)] * d_Xtemp[IDX2R(n1,n2,N2)];
	Xnorm = sqrt(Xnorm);
	if (Xnorm != 0)
  		for (int n1 = 0; n1 < N1; n1++)
    		d_Xout[IDX2R(n1,n2,N2)] = d_Xtemp[IDX2R(n1,n2,N2)] / Xnorm;

}



void tensorMatching(double* d_X, int N1, int N2, int* d_indH3, double* d_valH3, int Nt3 ,
          int nIter, int sparse, int stoc, double* d_Xout, double* d_ScoreOut) {
          
    int NN = N1 * N2;
    
	// Allocate d_Xtemp
	double* d_Xtemp;
	CUDA_SAFE_CALL(cudaMalloc(&d_Xtemp, NN*sizeof(double)));
	
	// Copy values of d_X into d_Xout
	CUDA_SAFE_CALL(cudaMemcpy(d_Xout, d_X, NN*sizeof(double), cudaMemcpyDeviceToDevice));
	
	// Allocate score matrix
	double *d_score;
	CUDA_SAFE_CALL(cudaMalloc(&d_score, Nt3*sizeof(double)));
	
	int maxIter = 100;
	int maxIter2 = 1;
	if (stoc == 2)
		maxIter2 = 10;
	
	for (int iter = 0; iter < maxIter; iter++) {

		// Set ScoreOut to 0
		CUDA_SAFE_CALL(cudaMemset(d_ScoreOut, 0, sizeof(double)));
		
		// Copy X into Xtemp
		CUDA_SAFE_CALL(cudaMemcpy(d_Xtemp, d_X, NN*sizeof(double), cudaMemcpyDeviceToDevice));
		
		// Debug
		double *Xtemp = (double *)malloc(NN*sizeof(double));
		double *Xout = (double *)malloc(NN*sizeof(double));
		
		// d == 3
		dim3 dimBlock(1024);
		dim3 dimGrid((Nt3 + dimBlock.x - 1)/dimBlock.x);
		calcScore<<<dimGrid, dimBlock>>>(d_indH3, d_valH3, Nt3, sparse, d_score, d_Xtemp, d_Xout);
		//cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();
			
		// Last iteration
		if (iter == (maxIter-1)) {
			// Create device pointer for d_score
			thrust::device_ptr<double> d_score_ptr(d_score);
			// Launch kernel
			lastIteration<<<dimGrid, dimBlock>>>(d_indH3, d_Xout, Nt3, d_score, d_ScoreOut);
			CUDA_CHECK_ERROR();
			// 3*score*score
			threeTimesSquare<<<dimGrid, dimBlock>>>(d_score, Nt3);
			CUDA_CHECK_ERROR();
			// do a sum reduction
			double sum = thrust::reduce(d_score_ptr, d_score_ptr + Nt3);
			// Set value of d_ScoreOut to sum
			initMatrix<<<1, 1>>>(d_ScoreOut, 1, 1, sum);
		}


		// Normalization
		if (stoc == 0) {
			// Create device pointer for d_Xtemp
			thrust::device_ptr<double> d_Xtemp_ptr(d_Xtemp);
			thrust::device_ptr<double> d_Xout_ptr(d_Xout);
			
			thrust::device_vector<double> d_XtempSquare(NN);
   			// Square d_Xtemp
			thrust::transform(d_Xtemp_ptr, d_Xtemp_ptr + NN, d_Xtemp_ptr, d_XtempSquare.begin(), thrust::multiplies<double>());
			// Sum of d_XtempSquare
   			double Xnorm = thrust::reduce(d_XtempSquare.begin(), d_XtempSquare.end());
  			// Square root of Xnorm		
      		Xnorm = sqrt(Xnorm);

          	// Initialize a device vector to hold Xnorm
          	thrust::device_vector<double> d_Xnorm(NN);
          	// Fill vector with Xnorm
          	thrust::fill(d_Xnorm.begin(), d_Xnorm.end(), Xnorm);
          	// Xout = Xtemp / Xnorm
          	thrust::transform(d_Xtemp_ptr, d_Xtemp_ptr + NN, d_Xnorm.begin(), d_Xout_ptr, thrust::divides<double>());

		} else {

			for (int iter2 = 0; iter2 < maxIter2; iter2++) {
				dimBlock = dim3(32, 32);
				dimGrid = dim3((N2 + dimBlock.x - 1)/dimBlock.x, (N1 + dimBlock.y - 1)/dimBlock.y);
				stocSingle<<<dimGrid, dimBlock>>>(d_Xtemp, N1, N2, d_Xout);

				if (stoc == 2)
					stocDouble<<<dimGrid, dimBlock>>>(d_Xtemp, N1, N2, d_Xout);

      		}

		}
		
		
	
	} // iter


}
