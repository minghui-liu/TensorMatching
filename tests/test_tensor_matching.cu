#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include "error_check.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

void saveMatrix(double *A, int h, int w, char *filename);
void saveMatrix(int *A, int h, int w, char *filename);
void tensorMatching(double* pX, int N1, int N2, int* pIndH3, double* pValH3, int Nt3 ,
          int nIter, int sparse, int stoc, double* pXout, double* pScoreOut);
void cudaTensorMatching(double* d_X, int N1, int N2, int* d_indH3, double* d_valH3, int Nt3 ,
          int nIter, int sparse, int stoc, double* d_Xout, double* d_ScoreOut);     

int main() {

	srand(time(NULL));

	int N1=5;
	int N2=5;

	int Nt=10;

	int NN = N1*N2;

	double *X = new double[NN];
	int *indH = new int[Nt*3];
	double *valH = new double[Nt];

	double *Xout = new double[NN];
	double *ScoreOut = new double[1];

	for(int i=0; i < NN; i++)
		X[i] = 1.0/N2;
	for(int i=0; i < Nt*3; i++)
		indH[i]=rand() % Nt;
	for(int i=0; i < Nt; i++)
		valH[i]=(double)(rand())/(double)(RAND_MAX);
		
	saveMatrix(X, N1, N2, "output/X.mat");
	saveMatrix(indH, Nt, 3, "output/indH.mat");
	saveMatrix(valH, Nt, 1, "output/valH.mat");

	/*std::ifstream matfile;
	matfile.open("input/indH.mat");

	for (int i = 0; i < Nt*3; i++) {
		matfile >> indH[i];
	}
	matfile.close();

	matfile.open("input/valH.mat");

	for (int i = 0; i < Nt; i++) {
		matfile >> valH[i];
	}
	
	saveMatrix(X, N1, N2, "output/X.mat");
	saveMatrix(indH, Nt, 3, "output/indH.mat");
	saveMatrix(valH, Nt, 1, "output/valH.mat");*/
	
	tensorMatching(X, N1, N2, indH, valH, Nt, 100, 1, 2, Xout, ScoreOut);
	
	printf("%f\n", *ScoreOut);
	//saveMatrix(Xout, N1, N2, "output/Xout.mat");
	
	/* Parallel Tensor Matching */
	double *d_X;
	CUDA_SAFE_CALL(cudaMalloc(&d_X, NN*sizeof(double)));
	CUDA_SAFE_CALL(cudaMemcpy(d_X, X, NN*sizeof(double), cudaMemcpyHostToDevice));
	int *d_indH;
	CUDA_SAFE_CALL(cudaMalloc(&d_indH, Nt*3*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_indH, indH, Nt*3*sizeof(int), cudaMemcpyHostToDevice));
	double *d_valH;
	CUDA_SAFE_CALL(cudaMalloc(&d_valH, Nt*sizeof(double)));
	CUDA_SAFE_CALL(cudaMemcpy(d_valH, valH, Nt*sizeof(double), cudaMemcpyHostToDevice));
	double *d_Xout;
	CUDA_SAFE_CALL(cudaMalloc(&d_Xout, NN*sizeof(double)));
	double *d_ScoreOut;
	CUDA_SAFE_CALL(cudaMalloc(&d_ScoreOut, sizeof(double)));
	
	cudaTensorMatching(d_X, N1, N2, d_indH, d_valH, Nt, 100, 1, 2, d_Xout, d_ScoreOut);
	
	delete[] X;
	delete[] indH;
	delete[] valH;
	delete[] Xout;
  
}

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
void calcScore(int *d_indH3, double *d_valH3, int Nt3, int sparse, double *d_Xtemp, double *d_Xout) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Nt3) return;
	
	double score;
	if (sparse == 1)
		score = d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]];
	else
		score = 1;
		
	printf("before:idx=%d\nindH[idx,0]=%d,indH[idx,1]=%d,indH[idx,2]=%d\nXout[indH[idx,0]]=%f,Xout[indH[idx,1]]=%f,Xout[indH[idx,2]]=%f\nscore=%f\nXtemp[indH[idx,0]]=%f,Xtemp[indH[idx,1]]=%f,Xtemp[indH[idx,2]]=%f\nvalH[idx]=%f\n\n",
			idx,
			d_indH3[IDX2R(idx,0,3)], d_indH3[IDX2R(idx,1,3)], d_indH3[IDX2R(idx,2,3)],
			d_Xout[d_indH3[IDX2R(idx,0,3)]], d_Xout[d_indH3[IDX2R(idx,1,3)]] ,d_Xout[d_indH3[IDX2R(idx,2,3)]],
			score,
			d_Xtemp[d_indH3[IDX2R(idx,0,3)]], d_Xtemp[d_indH3[IDX2R(idx,1,3)]], d_Xtemp[d_indH3[IDX2R(idx,2,3)]],
			d_valH3[idx]);
			
			double temp0 = score * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,1,3)]] * d_Xout[d_indH3[IDX2R(idx,2,3)]];
			double temp1 = score * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,2,3)]] * d_Xout[d_indH3[IDX2R(idx,0,3)]];
			double temp2 = score * d_valH3[idx] * d_Xout[d_indH3[IDX2R(idx,0,3)]] * d_Xout[d_indH3[IDX2R(idx,1,3)]];
	
	atomicAdd(&d_Xtemp[d_indH3[IDX2R(idx,0,3)]], temp0);
	atomicAdd(&d_Xtemp[d_indH3[IDX2R(idx,1,3)]], temp1);
	atomicAdd(&d_Xtemp[d_indH3[IDX2R(idx,2,3)]], temp2);
	
	
			printf("after:idx=%d\ntemp0=%f,temp1=%f,temp2=%f\nindH[idx,0]=%d,indH[idx,1]=%d,indH[idx,2]=%d\nXout[indH[idx,0]]=%f,Xout[indH[idx,1]]=%f,Xout[indH[idx,2]]=%f\nscore=%f\nXtemp[indH[idx,0]]=%f,Xtemp[indH[idx,1]]=%f,Xtemp[indH[idx,2]]=%f\nvalH[idx]=%f\n\n",
			idx,
			temp0,temp1,temp2,
			d_indH3[IDX2R(idx,0,3)], d_indH3[IDX2R(idx,1,3)], d_indH3[IDX2R(idx,2,3)],
			d_Xout[d_indH3[IDX2R(idx,0,3)]], d_Xout[d_indH3[IDX2R(idx,1,3)]] ,d_Xout[d_indH3[IDX2R(idx,2,3)]],
			score,
			d_Xtemp[d_indH3[IDX2R(idx,0,3)]], d_Xtemp[d_indH3[IDX2R(idx,1,3)]], d_Xtemp[d_indH3[IDX2R(idx,2,3)]],
			d_valH3[idx]);
}

void cudaTensorMatching(double* d_X, int N1, int N2, int* d_indH3, double* d_valH3, int Nt3 ,
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

	// d == 3
	//cudaDeviceSynchronize();
	dim3 dimBlock(1024);
	dim3 dimGrid((Nt3 + dimBlock.x - 1)/dimBlock.x);
	// calcScore<<<dimGrid, dimBlock>>>(d_indH3, d_valH3, Nt3, sparse, d_score, d_Xtemp, d_Xout, d_ScoreOut);
	calcScore<<<dimGrid, dimBlock>>>(d_indH3, d_valH3, Nt3, sparse, d_Xtemp, d_Xout);
	//cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	
	/* DEBUG */
	CUDA_SAFE_CALL(cudaMemcpy(Xtemp, d_Xtemp, NN*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Outputting Xtemp at iter 0 after d==3 ..." << std::endl;
	saveMatrix(Xtemp, N1, N2, "output/XtmepIter0_after.mat");


}

void tensorMatching(double* pX, int N1, int N2, int* pIndH3, double* pValH3, int Nt3 ,
          int nIter, int sparse, int stoc, double* pXout, double* pScoreOut) {
          
    int NN = N1 * N2;
	double* pXtemp = new double[NN];
	for (int n = 0; n < NN; n++)
		pXout[n] = pX[n];
	double score;
	int maxIter = 100;
	int maxIter2 = 1;
	if (stoc == 2)
		maxIter2 = 10;      
       
    // loop body    
	*pScoreOut = 0;
	for (int n = 0; n < NN; n++)
		pXtemp[n] = 1 * pX[n];
		
	
	
	// d == 3
	for (int t = 0; t < Nt3; t++) {
	
		if (sparse == 1)
    		score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
  		else
    		score = 1;
    		
    	printf("s_before:t=%d\nindH[t,0]=%d,indH[t,1]=%d,indH[t,2]=%d\nXout[indH[t,0]]=%f,Xout[indH[t,1]]=%f,Xout[indH[t,2]]=%f\nscore=%f\nXtemp[indH[t,0]]=%f,Xtemp[indH[t,1]]=%f,Xtemp[indH[t,2]]=%f\nvalH[t]=%f\n\n",
			t,
			pIndH3[IDX2R(t,0,3)], pIndH3[IDX2R(t,1,3)], pIndH3[IDX2R(t,2,3)],
			pXout[pIndH3[IDX2R(t,0,3)]], pXout[pIndH3[IDX2R(t,1,3)]] ,pXout[pIndH3[IDX2R(t,2,3)]],
			score,
			pXtemp[pIndH3[IDX2R(t,0,3)]], pXtemp[pIndH3[IDX2R(t,1,3)]], pXtemp[pIndH3[IDX2R(t,2,3)]],
			pValH3[t]);

		double temp0 = score * pValH3[t] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
		double temp1 = score * pValH3[t] * pXout[pIndH3[IDX2R(t,2,3)]] * pXout[pIndH3[IDX2R(t,0,3)]];
		double temp2 = score * pValH3[t] * pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]];
		
		
  		pXtemp[pIndH3[IDX2R(t,0,3)]] += temp0;
  		pXtemp[pIndH3[IDX2R(t,1,3)]] += temp1;
 	  	pXtemp[pIndH3[IDX2R(t,2,3)]] += temp2;
 	  	
		//if (iter == (maxIter-1)) {
		//	score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]];
		//	*pScoreOut = *pScoreOut + 3*score*score;
		//}
		
		printf("s_after:t=%d\ntemp0=%f,temp1=%f,temp2=%f\nindH[t,0]=%d,indH[t,1]=%d,indH[t,2]=%d\nXout[indH[t,0]]=%f,Xout[indH[t,1]]=%f,Xout[indH[t,2]]=%f\nscore=%f\nXtemp[indH[t,0]]=%f,Xtemp[indH[t,1]]=%f,Xtemp[indH[t,2]]=%f\nvalH[t]=%f\n\n",
			t,
			temp0,temp1,temp2,
			pIndH3[IDX2R(t,0,3)], pIndH3[IDX2R(t,1,3)], pIndH3[IDX2R(t,2,3)],
			pXout[pIndH3[IDX2R(t,0,3)]], pXout[pIndH3[IDX2R(t,1,3)]] ,pXout[pIndH3[IDX2R(t,2,3)]],
			score,
			pXtemp[pIndH3[IDX2R(t,0,3)]], pXtemp[pIndH3[IDX2R(t,1,3)]], pXtemp[pIndH3[IDX2R(t,2,3)]],
			pValH3[t]);
	}
	
	/* DEBUG */
	std::cout << "Outputting Xtemp at iter 0 after d==3..." << std::endl;
	saveMatrix(pXtemp, N1, N2, "output/XtmepIter0_serial_after.mat");
}

/*
void tensorMatching(double* pX, int N1, int N2, int* pIndH3, double* pValH3, int Nt3 ,
          int nIter, int sparse, int stoc, double* pXout, double* pScoreOut) {
	int NN = N1 * N2;
	double* pXtemp = new double[NN];
	for (int n = 0; n < NN; n++)
		pXout[n] = pX[n];
	double score;
	int maxIter = 100;
	int maxIter2 = 1;
	if (stoc == 2)
		maxIter2 = 10;
		
	for (int iter = 0; iter < maxIter; iter++) {
		*pScoreOut = 0;
		
		for (int n = 0; n < NN; n++)
			pXtemp[n] = 1 * pX[n];
		
		// d == 3
		for (int t = 0; t < Nt3; t++) {
			if (sparse == 1)
        		score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
      		else
        		score = 1;
      		pXtemp[pIndH3[IDX2R(t,0,3)]] += score * pValH3[t] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
      		pXtemp[pIndH3[IDX2R(t,1,3)]] += score * pValH3[t] * pXout[pIndH3[IDX2R(t,2,3)]] * pXout[pIndH3[IDX2R(t,0,3)]];
     	  	pXtemp[pIndH3[IDX2R(t,2,3)]] += score * pValH3[t] * pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]];
			if (iter == (maxIter-1)) {
				score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]];
				*pScoreOut = *pScoreOut + 3*score*score;
			}
    	}
    	
		// normalization    
    	if (stoc == 0 ) {
    	
      		double pXnorm=0;
      		for (int n2 = 0; n2 < N2; n2++)
        		for (int n1 = 0; n1 < N1; n1++)
          			pXnorm += pXtemp[IDX2R(n1,n2,N2)] * pXtemp[IDX2R(n1,n2,N2)];
          			
      		pXnorm = sqrt(pXnorm);
      		
      		for (int n2 = 0; n2 < N2; n2++)
        		for (int n1 = 0; n1 < N1; n1++)
          			pXout[IDX2R(n1,n2,N2)] = pXtemp[IDX2R(n1,n2,N2)] / pXnorm; 
          			     		
    	} else {
    	
      		for (int iter2 = 0; iter2 < maxIter2; iter2++) {
        		for (int n2 = 0; n2 < N2; n2++) {
          			double pXnorm = 0;
          			for (int n1 = 0; n1 < N1; n1++)
            			pXnorm += pXtemp[IDX2R(n1,n2,N2)] * pXtemp[IDX2R(n1,n2,N2)];
          			pXnorm = sqrt(pXnorm);
          			for (int n1 = 0; n1 < N1; n1++)
            			if (pXnorm !=0 )
              				pXout[IDX2R(n1,n2,N2)] = pXtemp[IDX2R(n1,n2,N2)] / pXnorm;
        		}

        		if (stoc == 2)
		      		for (int n2 = 0; n2 < N2; n2++) {
		        		double pXnorm=0;
		        		for (int n1 = 0; n1 < N1; n1++)
		          			pXnorm += pXtemp[IDX2R(n1,n2,N2)] * pXtemp[IDX2R(n1,n2,N2)];
		        		pXnorm = sqrt(pXnorm);
		        		if (pXnorm != 0)
				      		for (int n1 = 0; n1 < N1; n1++)
				        		pXout[IDX2R(n1,n2,N2)] = pXtemp[IDX2R(n1,n2,N2)] / pXnorm;
		      		}
      		}
      		
    	} 
	}
  	delete[] pXtemp;
}
*/


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

