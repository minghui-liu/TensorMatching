#include <iostream>
#include <cstdio>

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

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

/**
 * Given a matrix of size width*height, compute the square norm of each column.
 *
 * @param mat    : the matrix
 * @param width  : the number of columns for a colum major storage matrix
 * @param height : the number of rowm for a colum major storage matrix
 * @param norm   : the vector containing the norm of the matrix
 */
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

int main(void) {

	int dim = 2;
	int ref_nb = 20;
	int query_nb = 5;
	
	double *mat = (double*)malloc(dim * ref_nb * sizeof(double));
	double *norm = (double*)malloc(ref_nb * sizeof(double));
	
	double *d_mat, *d_norm;
	cudaMalloc(&d_mat, dim*ref_nb*sizeof(double));
	cudaMalloc(&d_norm, ref_nb*sizeof(double));
	
	for (int i = 0; i < ref_nb * dim; i++) {
		mat[i] = 2;
	}
	cudaMemcpy(d_mat, mat, dim*ref_nb*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(1024);
	dim3 dimGrid( (ref_nb + dimBlock.x - 1)/dimBlock.x );
	cuComputeNorm<<<dimGrid, dimBlock>>>(d_mat, dim, ref_nb, d_norm);
	
	cudaMemcpy(norm, d_norm, ref_nb*sizeof(double), cudaMemcpyDeviceToHost);
	
	saveMatrix(mat, dim, ref_nb, "mat.mat");
	saveMatrix(norm, 1, ref_nb, "norm.mat");
	
	return 0;
}
