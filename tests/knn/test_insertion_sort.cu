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

/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist        distance matrix
  * @param dist_pitch  pitch of the distance matrix given in number of columns
  * @param ind         index matrix
  * @param ind_pitch   pitch of the index matrix given in number of columns
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
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

int main() {
	srand(time(NULL));
	int dim = 2;
	int ref_nb = 20;
	int query_nb = 5;
	int nNN = 3;
	
	double *mat = (double*)malloc(query_nb * ref_nb * sizeof(double));
	int *mat_ind = (int*)malloc(query_nb * ref_nb * sizeof(int));
	double *dist = (double*)malloc(nNN*query_nb*sizeof(double));
	int *ind = (int*)malloc(nNN*query_nb*sizeof(int));
	
	double *d_mat;
	cudaMalloc(&d_mat, query_nb*ref_nb*sizeof(double));
	int *d_mat_ind;
	cudaMalloc(&d_mat_ind, query_nb*ref_nb*sizeof(int));
	double *d_dist;
	cudaMalloc(&d_dist, nNN*query_nb*sizeof(double));
	int *d_ind;
	cudaMalloc(&d_ind, nNN*query_nb*sizeof(int));
	
	for (int i = 0; i < query_nb * ref_nb; i++) {
		mat[i] = (double)rand()/RAND_MAX;
	}
	
	saveMatrix(mat, query_nb, ref_nb, "mat.mat");
	cudaMemcpy(d_mat, mat, query_nb*ref_nb*sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(1024);
	dim3 dimGrid( (ref_nb + dimBlock.x - 1)/dimBlock.x );
	cuInsertionSort<<<dimGrid, dimBlock>>>(d_mat, d_mat_ind, query_nb, ref_nb, d_dist, d_ind, nNN);
	
	cudaMemcpy(mat, d_mat, query_nb*ref_nb*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_ind, d_mat_ind, query_nb*ref_nb*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(dist, d_dist, nNN*query_nb*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ind, d_ind, nNN*query_nb*sizeof(int), cudaMemcpyDeviceToHost);
	
	saveMatrix(mat, query_nb, ref_nb, "mat_sort.mat");
	saveMatrix(mat_ind, query_nb, ref_nb, "mat_ind.mat");
	saveMatrix(dist, nNN, query_nb, "dist.mat");
	saveMatrix(ind, nNN, query_nb, "ind.mat");
	
	return 0;

}



