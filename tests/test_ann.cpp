#include <iostream>
#include <cstdlib>						// C standard library
#include <cstring>						// string manipulation
#include <fstream>						// file I/O
#include <unistd.h>	
#include <ANN/ANN.h>

using namespace std;					// make std:: accessible


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


int main(int argc, char **argv)
{
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


	/* ANN */
	double *dists = (double*)malloc(k*query_nb*sizeof(double));
	int *inds = (int*)malloc(k*query_nb*sizeof(int));
	
	ANNpointArray	dataPts;
	ANNpoint		queryPt;
	ANNidxArray		nnIdx;
	ANNdistArray	nnDist;
	ANNkd_tree*		kdTree;
	
	dataPts = annAllocPts(ref_nb, dim);
	queryPt = annAllocPt(dim);
	nnIdx = new ANNidx[k];
	nnDist = new ANNdist[k];

	// Generate dataPts
	srand(time(NULL));
	for (int i = 0; i < ref_nb; i++)
		for (int j = 0; j < dim; j++)
			dataPts[i][j] = (double)rand()/RAND_MAX;
	
	// build kd tree
	kdTree = new ANNkd_tree(dataPts, ref_nb, dim);
	
	// query nT points
	for (int i = 0; i < query_nb; i++) {
		// get query point coordinates
		for (int j = 0; j < dim; j++)
			queryPt[j] = (double)rand()/RAND_MAX;
		
		// search
		kdTree->annkSearch(queryPt, k, nnIdx, nnDist, 10);
	
		for (int j = 0; j < k; j++) {
			inds[j*query_nb+i] = nnIdx[j];
			dists[j*query_nb+i] = nnDist[j];
		}
	}
	
	if (vflag) {
		cout << "Distances" << endl;
		printMatrix(dists, k, query_nb);
		cout << "Indices" << endl;
		printMatrix(inds, k, query_nb);
	}

	return EXIT_SUCCESS;
}

