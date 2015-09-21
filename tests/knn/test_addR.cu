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
 * Given the distance matrix of size width*height, adds the column vector
 * of size 1*height to each column of the matrix.
 *
 * @param dist   : the matrix
 * @param width  : the number of columns for a colum major storage matrix
 * @param pitch  : the pitch in number of column
 * @param height : the number of rowm for a colum major storage matrix
 * @param vec    : the vector to be added
 */
__global__ void cuAddRNorm(double *dist, int height, int width, double *ref_norm) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	__shared__ double shared_vec[32];
	shared_vec[threadIdx.x] = ref_norm[x];
	__syncthreads();
	dist[IDX2R(y,x,width)] += shared_vec[threadIdx.x];
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
	
	for (int i = 0; i < ref_nb * dim; i++)	mat[i] = 2;
	for (int i = 0; i < ref_nb; i++)	norm[i] = 1;
	
	cudaMemcpy(d_mat, mat, dim*ref_nb*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_norm, norm, ref_nb*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(32,32);
	dim3 dimGrid( (ref_nb + dimBlock.x - 1)/dimBlock.x, (dim + dimBlock.x - 1)/dimBlock.x );
	cuAddRNorm<<<dimGrid, dimBlock>>>(d_mat, dim, ref_nb, d_norm);
	
	cudaMemcpy(mat, d_mat, dim*ref_nb*sizeof(double), cudaMemcpyDeviceToHost);
	
	saveMatrix(mat, dim, ref_nb, "mat.mat");
	
	return 0;
}
