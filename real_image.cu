#include <iostream>
#include <unistd.h>
#include <cstdio>
#include "surflib.h"
#include "error_check.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

extern void saveMatrix(double *A, int h, int w, char *filename);
extern void imageMatching(double *P1, int nP1, double *P2, int nP2, double *X2, char vflag, char sflag, char tflag);

__global__
void matMax(double *M, int h, int w, double *out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= h) return;

	double max = M[IDX2R(idx,0,w)];
	double index = 0;
	for (int i = 1; i < w; i++) {
		if (M[IDX2R(idx,i,w)] > max) {
			max = M[IDX2R(idx,i,w)];
			index = i;
		}
	}
	out[idx] = index;
}


int main(int argc, char *argv[]) {
	// Parse command-line arguments
	char vflag = 0;
	char sflag = 0;
	char tflag = 0;
	char opt;	
	// process command-line options
	while ((opt = getopt(argc, argv, "vst")) != -1) {
		switch (opt) {
			case 'v':
				vflag = 1;	
				break;
			case 's':
				sflag = 1;
				break;
			case 't':
				tflag = 1;
				break;
			case '?':
				if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				abort();
		}
	}	
	if (argc < 3 || optind == argc) {
	  	fprintf(stderr, "Must specify two filenames.\n");
		return EXIT_FAILURE;
	}	
	// image files
	char *file1 = argv[optind++];
	char *file2 = argv[optind++];
	// print non-opt args
	for (int index = optind; index < argc; index++)
		printf ("Non-option argument %s\n", argv[index]);

	// Detect interesting points
	IplImage *img1, *img2;
 	img1 = cvLoadImage(file1);
 	img2 = cvLoadImage(file2);

	IpVec ipts1, ipts2;
	surfDetDes(img1,ipts1,false,4,4,2,0.0001f);
	surfDetDes(img2,ipts2,false,4,4,2,0.0001f);
	//int nP1 = ipts1.size();
	//int nP2 = ipts2.size();
	std::cout << "size of ipts1: " << ipts1.size() << std::endl;
	std::cout << "size of ipts2: " << ipts2.size() << std::endl;

	// Get matching interesting points
	IpPairVec matches;
  	getMatches(ipts1,ipts2,matches);
  	std::cout << "size of matches: " <<  matches.size() << std::endl;
  
  	//bool compareDist(std::pair<Ipoint, Ipoint> a, std::pair<Ipoint, Ipoint> b) {
  		
  	//}
  	
  	int nP1;
  	//if (matches.size() >= 100) 
  	//	nP1 = matches.size() / 2;
  	//else
  		nP1 = matches.size();
	
	int nP2 = nP1;

	// Allocate space for P1 and P2
	double *P1 = (double*)malloc(2*nP1*sizeof(double));
	double *P2 = (double*)malloc(2*nP2*sizeof(double));

	// Create P1 and P2
	for (int i = 0; i < nP1; i++) {
		P1[IDX2R(0,i,nP1)] = matches[i].first.x;
		P1[IDX2R(1,i,nP1)] = matches[i].first.y;
		P2[IDX2R(0,i,nP2)] = matches[i].second.x;
		P2[IDX2R(1,i,nP2)] = matches[i].second.y;
	}

	double *X2 = (double*)malloc(nP2*nP1*sizeof(double));

	imageMatching(P1, nP1, P2, nP2, X2, vflag, sflag, tflag);

	for (unsigned int i = 0; i < matches.size(); i++) {
	    drawPoint(img1,matches[i].first);
	    drawPoint(img2,matches[i].second);
	    //cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
	    //cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
	}

	// Allocate hard
	double *d_Hard;
	CUDA_SAFE_CALL(cudaMalloc(&d_Hard, nP1*sizeof(double)));

	// Copy X2 to device
	double *d_X2;
	CUDA_SAFE_CALL(cudaMalloc(&d_X2, nP1*nP2*sizeof(double)));
	CUDA_SAFE_CALL(cudaMemcpy(d_X2, X2, nP1*nP2*sizeof(double), cudaMemcpyHostToDevice));

	// Get max of X2 to determine matching	
	dim3 dimBlock = 1024;
	dim3 dimGrid = (nP1 + dimBlock.x - 1)/dimBlock.x ;
	matMax<<<dimGrid, dimBlock>>>(d_X2, nP1, nP2, d_Hard);

	// Copy Hard back to host
	double *Hard = (double*)malloc(nP1*sizeof(double));
	CUDA_SAFE_CALL(cudaMemcpy(Hard, d_Hard, nP1*sizeof(double), cudaMemcpyDeviceToHost));

	saveMatrix(Hard, nP1, 1, "output/Hard.mat");
	
	/*int count = 0;
	for (int i = 0; i < nP1; i++) {
		for (int j = 0; j < nP2; j++) {
			const int & w = img1->width;
			if (X2[IDX2R(i,j,nP2)] > 0.5) {
				if (i == j) count++;
				cvLine(img1, cvPoint(matches[i].first.x, matches[i].first.y), cvPoint(matches[j].second.x+w, matches[j].second.y), cvScalar(255,255,255),1);
				cvLine(img2, cvPoint(matches[i].first.x-w, matches[i].first.y), cvPoint(matches[j].second.x, matches[j].second.y), cvScalar(255,255,255),1);
			}
		}
	}
	std::cout << "count = " << count << std::endl;*/
	
	const int & w = img1->width;
	for (int i = 0; i < nP1; i++) {
		cvLine(img1, cvPoint(matches[i].first.x, matches[i].first.y), cvPoint(matches[Hard[i]].second.x+w, matches[Hard[i]].second.y), cvScalar(255,255,255),1);
		cvLine(img2, cvPoint(matches[i].first.x-w, matches[i].first.y), cvPoint(matches[Hard[i]].second.x, matches[Hard[i]].second.y), cvScalar(255,255,255),1);
	}

	cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
	cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
	cvShowImage("1", img1);
	cvShowImage("2",img2);
	cvWaitKey(0);

	return 0;

}

















