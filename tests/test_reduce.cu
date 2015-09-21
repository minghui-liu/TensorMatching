#include <iostream>
#include <ctime>
#include <thrust/reduce.h>

int main() {
	srand(time(NULL));
	double *A = (double*)malloc(10*sizeof(double));
	for (int i=0; i<10; i++)
		A[i] = (double)rand() / RAND_MAX;

	double sum = thrust::reduce(A, A+10);
	double avg = sum / 10;
	
	std::cout << "sum = " << sum << std::endl;
	std::cout << "avg = " << avg << std::endl;

	return 0;
}
