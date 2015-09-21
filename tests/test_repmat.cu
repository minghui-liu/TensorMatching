#include <iostream>
#include <stdio.h>

//create an m-by-n tiling of a given matrix
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

void printMatrix(double *h_M, int height, int width) {
	for (int i=0; i < height; i++) {
		for (int j=0; j < width; j++)
			std::cout << h_M[i * width + j] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(void) {
	int h = 2;
	int w = 3;
	int m = 3;
	int n = 2;
	
	double *A = (double*)malloc(h*w*sizeof(double));
	double *B = (double*)malloc(h*m*w*n*sizeof(double));

	for (int i=0; i < h*w; i++)
		A[i] = i;

	printMatrix(A, h, w);

	double *d_A, *d_B;
	cudaMalloc(&d_A, h*w*sizeof(double));
	cudaMalloc(&d_B, h*m*w*n*sizeof(double));
	
	cudaMemcpy(d_A, A, h*w*sizeof(double), cudaMemcpyHostToDevice);
	dim3 dimBlock(32, 32);
	repmat<<<1, dimBlock>>>(d_A, h, w, d_B, m, n);

	cudaMemcpy(B, d_B, h*m*w*n*sizeof(double), cudaMemcpyDeviceToHost);

	printMatrix(B, h*m, w*n);

	free(A);
	free(B);
	cudaFree(d_A);
	cudaFree(d_B);

}
