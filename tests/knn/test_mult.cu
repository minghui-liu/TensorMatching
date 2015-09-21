#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>


// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

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



int main() {
	
	int m = 2;
	int k = 3;
	int n = 4;
	
	double *A = (double *)malloc(k*m*sizeof(double));
	double *B = (double *)malloc(k*n*sizeof(double));
	double *C = (double *)malloc(m*n*sizeof(double));
	
	double *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, k*m*sizeof(double));
	cudaMalloc(&d_B, k*n*sizeof(double));
	cudaMalloc(&d_C, m*n*sizeof(double));
	
	for (int i = 0; i < k*m; i++)	A[i] = i;
	for (int i = 0; i < k*n; i++)	B[i] = 1.0;
	
	printMatrix(A, k, m);
	printMatrix(B, k, n);
	
	cudaMemcpy(d_A, A, k*m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, k*n*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(32,32);
	dim3 dimGrid( (n + dimBlock.x - 1)/dimBlock.x, (m + dimBlock.y - 1)/dimBlock.y );
	cuMatrixMult<<<dimGrid, dimBlock>>>(-2.0, d_A, d_B, m, n, k, d_C);
	
	cudaMemcpy(C, d_C, m*n*sizeof(double), cudaMemcpyDeviceToHost);
	

	printMatrix(C, m, n);

	
	return 0;

}

