/* 
 * File: computeFeature.cpp
 * Rewritten using row major indexing
 */

#include <iostream>
#include <cmath>
#include "compute_feature.h"

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

void computeFeature(double *P1, int nP1, double* P2, int nP2, int* T1, int nT1, double* F1, double* F2) {
	const int nFeature = 3;
	
	for(int t=0; t < nT1; t++) {
    	computeFeatureSimple(P1, nP1, T1[IDX2R(0,t,nT1)], T1[IDX2R(1,t,nT1)], T1[IDX2R(2,t,nT1)], F1+t, nT1);
	}
	
	for(int i=0; i < nP2; i++)
		for(int j=0; j < nP2; j++)
			for(int k=0; k < nP2; k++)
				computeFeatureSimple(P2, nP2, i, j, k, F2+((i*nP2+j)*nP2+k), nP2*nP2*nP2);

}

void computeFeatureSimple(double *P, int nP, int i, int j, int k, double *F, int nT) {
	const int nFeature = 3;
	
	double vecX[nFeature];
	double vecY[nFeature];
	int ind[nFeature];
	
	ind[0] = i; ind[1] = j; ind[2] = k;
	double n;
	
	// duplicate indices
	if ( (ind[0]==ind[1]) || (ind[0]==ind[2]) || (ind[1]==ind[2]) ) {
		F[0] = F[nT] = F[2*nT] = -10;
		return;
	}
	for(int f=0; f < nFeature; f++) {
		// difference of x and y cordinate between two points
		vecX[f]=P[ind[((f+1)%3)]]-P[ind[f]];
		vecY[f]=P[nP+ind[((f+1)%3)]]-P[nP+ind[f]];
		// two-norm of differences
		double norm=sqrt(vecX[f]*vecX[f]+vecY[f]*vecY[f]);
		if(norm!=0) {
			vecX[f]/=norm;
			vecY[f]/=norm;
		} else {
			vecX[f]=0;
			vecY[f]=0;
		}
	}
	// compute feature
	for(int f=0;f<nFeature;f++) {
		F[f*nT] = vecX[((f+1)%3)]*vecY[f]-vecY[((f+1)%3)]*vecX[f];
	}

}



