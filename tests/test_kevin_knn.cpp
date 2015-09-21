#include <iostream>
#include <cstdlib>						// C standard library
#include <cstring>						// string manipulation
#include <fstream>						// file I/O

using namespace std;					// make std:: accessible

int knn(double* ref_pts, int ref_width, double* query_pts, int query_width, int dim, int nNN, double* dist, int* ind);

void printMatrix(double *Pts, int h, int w)	// print point
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cout << Pts[i*w+j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
}

void printMatrix(int*Pts, int h, int w)	// print point
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cout << Pts[i*w+j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
}


int main(int argc, char **argv) {
	int	k = 5;					// number of nearest neighbors
	int	dim	= 2;				// dimension
	int nDataPts = 20;			// number of data points
	int nQueryPts = 10;

	ifstream dataStream;		// data file stream
	ifstream queryStream;		// query file stream
	
	double *dataPts;			// data points
	double *queryPts;			// query point
	int *inds;					// near neighbor indices
	double *dists;				// near neighbor distances
	
	dataStream.open("data.pts", ios::in);	
	if (!dataStream) {
		cerr << "Cannot open data file\n";
		exit(1);
	}

	queryStream.open("query.pts", ios::in);
	if (!queryStream) {
		cerr << "Cannot open query file\n";
		exit(1);
	}

	dataPts = new double[dim*nDataPts];		// allocate query point
	queryPts = new double[dim*nQueryPts];	// allocate data points
	inds = new int[nQueryPts*k];			// allocate near neigh indices
	dists = new double[nQueryPts*k];		// allocate near neighbor dists

	// Read data points
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < nDataPts; j++) {
			if(!(dataStream >> dataPts[i*nDataPts+j])) return -1;
		}
	}
	
	// Read query points
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < nQueryPts; j++) {
			if(!(queryStream >> queryPts[i*nQueryPts+j])) return -2;
		}
	}

	cout << "Data Points" << endl;
	printMatrix(dataPts, dim, nDataPts);
	cout << "Query Points" << endl;
	printMatrix(queryPts, dim, nQueryPts);

	knn(dataPts, nDataPts, queryPts, nQueryPts, dim, k, dists, inds);
	
	cout << "Distances" << endl;
	printMatrix(dists, k, nQueryPts);
	cout << "Indices" << endl;
	printMatrix(inds, k, nQueryPts);
	
	

	return EXIT_SUCCESS;
}

