#include <iostream>
#include <cstdlib>						// C standard library
#include <cstring>						// string manipulation
#include <fstream>						// file I/O
#include "../knn.h"

using namespace std;					// make std:: accessible

int				k				= 1;			// number of nearest neighbors
int				dim				= 2;			// dimension
double			eps				= 0;			// error bound
int				nDataPts		= 20;			// number of data points
int				nQueryPts		= 10;

istream*		dataIn			= NULL;			// input for data points
istream*		queryIn			= NULL;			// input for query points

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
	// Parse command-line arguments
	char vflag = 0;
	char tflag = 0;
	int query_nb, ref_nb, dim, k;
	char opt;
	
	// process command-line options
	while ((opt = getopt(argc, argv, "vtr:q:d:k:")) != -1) {
		switch (opt) {
			case 'v':
				vflag = 1;	
				break;
			case 't':
				tflag = 1;
				break;
			case 'q':
				query_nb = atoi(optarg);
				break;
			case 'r':
				ref_nb = atoi(optarg);
				break;
			case 'd':
				dim = atoi(optarg);
				break;
			case 'k':
				k = atoi(optarg);
				break;
			default:
				abort();
		}
	}
	
	if (argc < 8) {
	  	fprintf(stderr, "Must specify q r d and k.\n");
		return EXIT_FAILURE;
	}

	// print non-opt args
	for (int index = optind; index < argc; index++)
		printf ("Non-option argument %s\n", argv[index]);

	// std::cout << "q=" << query_nb << ", r=" << ref_nb << ", d=" << dim << ", k=" << k << ", v=" << (int)vflag << ", t=" << (int)tflag << std::endl;

	double *dataPts;			// data points
	double *queryPts;			// query point
	int *inds;					// near neighbor indices
	double *dists;				// near neighbor distances

	dataPts = new double[dim*ref_nb];		// allocate query point
	queryPts = new double[dim*query_nb];	// allocate data points
	inds = new int[query_nb*k];			// allocate near neigh indices
	dists = new double[query_nb*k];		// allocate near neighbor dists

	// Generate data points
	for (int i = 0; i < dim * ref_nb; i++)
		dataPts[i] = (double)rand()/RAND_MAX;
	
	// Generate query points
	for (int i = 0; i < dim * query_nb; i++)
		queryPts[i] = (double)rand()/RAND_MAX;


	if (vflag) {
		cout << "Data Points" << endl;
		printMatrix(dataPts, dim, ref_nb);
		cout << "Query Points" << endl;
		printMatrix(queryPts, dim, query_nb);

		knn(dataPts, ref_nb, queryPts, query_nb, dim, k, dists, inds);
		
		cout << "Distances" << endl;
		printMatrix(dists, k, query_nb);
		cout << "Indices" << endl;
		printMatrix(inds, k, query_nb);
	}
	

	return EXIT_SUCCESS;
}

