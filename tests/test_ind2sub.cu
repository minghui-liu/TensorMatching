#include <iostream>

__global__ void ind2sub(int nP, int *inds, int *i, int *j, int *k) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nP*nP*nP) return;
	i[idx] = inds[idx] / (nP*nP);
	j[idx] = (inds[idx] % (nP*nP)) / nP;
	k[idx] = inds[idx] % nP;
}


int main(void) {
	int nP = 5;
	int *inds = (int*)malloc(nP*nP*nP*sizeof(int));
	for (int i=0; i < nP*nP*nP; i++)
		inds[i] = i;
	int *d_inds;
	cudaMalloc(&d_inds, nP*nP*nP*sizeof(int));
	cudaMemcpy(d_inds, inds, nP*nP*nP*sizeof(int), cudaMemcpyHostToDevice);
	int *i = (int*)malloc(nP*nP*nP*sizeof(int));
	int *j = (int*)malloc(nP*nP*nP*sizeof(int));
	int *k = (int*)malloc(nP*nP*nP*sizeof(int));
	int *d_i, *d_j, *d_k;
	cudaMalloc(&d_i, nP*nP*nP*sizeof(int));
	cudaMalloc(&d_j, nP*nP*nP*sizeof(int));
	cudaMalloc(&d_k, nP*nP*nP*sizeof(int));
	ind2sub<<<1, 1024>>>(nP, d_inds, d_i, d_j, d_k);
	cudaMemcpy(i, d_i, nP*nP*nP*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(j, d_j, nP*nP*nP*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(k, d_k, nP*nP*nP*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int f=0; f < nP*nP*nP; f++) {
		std::cout << i[f] << ' ';
	}
	std::cout << std::endl;
	for (int f=0; f < nP*nP*nP; f++) {
		std::cout << j[f] << ' ';
	}
	std::cout << std::endl;
	for (int f=0; f < nP*nP*nP; f++) {
		std::cout << k[f] << ' ';
	}
	std::cout << std::endl;


}
