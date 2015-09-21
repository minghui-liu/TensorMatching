#include <iostream>
#include "error_check.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

extern void saveMatrix(double*, int, int, char*);
extern __global__ void initMatrix(double *M, int h, int w, double value);


__global__
void calcScore(int *d_indH3, double *d_valH3, int Nt3, int sparse, double *d_Xtemp, double *d_Xout) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Nt3) return;
	
	double score;
	if (sparse == 1)
		score = d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]];
	else
		score = 1;
		
   // printf("idx=%d,before: Xtemp[%d]=%.16f, Xtemp[%d]=%.16f, Xtemp[%d]=%.16f\n",idx,d_indH3[IDX2R(idx,0,3)],d_Xtemp[d_indH3[IDX2R(idx,0,3)]],d_indH3[IDX2R(idx,1,3)],d_Xtemp[d_indH3[IDX2R(idx,1,3)]],d_indH3[IDX2R(idx,2,3)],d_Xtemp[d_indH3[IDX2R(idx,2,3)]]);		
	
	d_Xtemp[d_indH3[IDX2R(idx,0,3)]] += (score * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]]);
	d_Xtemp[d_indH3[IDX2R(idx,1,3)]] += (score * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,2,3)]] * d_Xout[d_indH3[IDX2R(idx,0,3)]]);
	d_Xtemp[d_indH3[IDX2R(idx,2,3)]] += (score * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]]);
	
	// printf("idx=%d,after: Xtemp[%d]=%.16f, Xtemp[%d]=%.16f, Xtemp[%d]=%.16f\n",idx,d_indH3[IDX2R(idx,0,3)],d_Xtemp[d_indH3[IDX2R(idx,0,3)]],d_indH3[IDX2R(idx,1,3)],d_Xtemp[d_indH3[IDX2R(idx,1,3)]],d_indH3[IDX2R(idx,2,3)],d_Xtemp[d_indH3[IDX2R(idx,2,3)]]);	
	
	//printf("idx=%d,score=%f,valH[idx]=%f\n", idx, score, d_valH3[idx]);
	
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
	
	// Set ScoreOut to 0
	CUDA_SAFE_CALL(cudaMemset(d_ScoreOut, 0, sizeof(double)));
	
	// Copy X into Xtemp
	CUDA_SAFE_CALL(cudaMemcpy(d_Xtemp, d_X, NN*sizeof(double), cudaMemcpyDeviceToDevice));
	
	// Debug
	double *Xtemp = (double *)malloc(NN*sizeof(double));
	
	/* DEBUG */
	CUDA_SAFE_CALL(cudaMemcpy(Xtemp, d_Xtemp, NN*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Outputting Xtemp at iter 0 before d==3 ..." << std::endl;
	saveMatrix(Xtemp, N1, N2, "output/XtmepIter0_before.mat");
	std::cout << "NT3 = " << Nt3 << std::endl;

	// d == 3
	cudaDeviceSynchronize();
	dim3 dimBlock(1024);
	dim3 dimGrid((Nt3 + dimBlock.x - 1)/dimBlock.x);
	// calcScore<<<dimGrid, dimBlock>>>(d_indH3, d_valH3, Nt3, sparse, d_score, d_Xtemp, d_Xout, d_ScoreOut);
	calcScore<<<dimGrid, dimBlock>>>(d_indH3, d_valH3, Nt3, sparse, d_Xtemp, d_Xout);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	
	/* DEBUG */
	CUDA_SAFE_CALL(cudaMemcpy(Xtemp, d_Xtemp, NN*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Outputting Xtemp at iter 0 after d==3 ..." << std::endl;
	saveMatrix(Xtemp, N1, N2, "output/XtmepIter0_after.mat");

		
		
}
