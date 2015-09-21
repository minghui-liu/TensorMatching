#include <cstdlib>
#include <ctime>
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

__global__ void cuAddQNormAndSqrt(double *dist, int height, int width, double *query_norm) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	dist[IDX2R(y,x,width)] = sqrt(dist[IDX2R(y,x,width)] + query_norm[x]);
}

int main() {

	int dim = 2;
	int query_nb = 5;
	int ref_nb = 20;
	
	double *mat = (double*)malloc(query_nb * ref_nb * sizeof(double));
	double *norm = (double*)malloc(ref_nb * sizeof(double));
	
	double *d_mat, *d_norm;
	cudaMalloc(&d_mat, query_nb*ref_nb*sizeof(double));
	cudaMalloc(&d_norm, ref_nb*sizeof(double));
	
	for (int i = 0; i < ref_nb * query_nb; i++)	mat[i] = 3;
	for (int i = 0; i < ref_nb; i++)	norm[i] = 1;
	
	cudaMemcpy(d_mat, mat, query_nb*ref_nb*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_norm, norm, ref_nb*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(32,32);
	dim3 dimGrid( (ref_nb + dimBlock.x - 1)/dimBlock.x, (dim + dimBlock.x - 1)/dimBlock.x );
	cuAddQNormAndSqrt<<<dimGrid, dimBlock>>>(d_mat, query_nb, ref_nb, d_norm);
	
	cudaMemcpy(mat, d_mat, query_nb*ref_nb*sizeof(double), cudaMemcpyDeviceToHost);
	
	saveMatrix(mat, dim, ref_nb, "mat.mat");
	
	return 0;

}



