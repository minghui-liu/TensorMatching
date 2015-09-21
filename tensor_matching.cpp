#include <iostream>
#include <cmath>

extern void saveMatrix(double *A, int h, int w, char *filename);

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

void tensorMatching(double* pX, int N1, int N2, int* pIndH3, double* pValH3, int Nt3 ,
          int nIter, int sparse, int stoc, double* pXout, double* pScoreOut) {
	int NN = N1 * N2;
	double* pXtemp = new double[NN];
	for (int n = 0; n < NN; n++)
		pXout[n] = pX[n];
	double score;
	int maxIter = 100;
	int maxIter2 = 1;
	if (stoc == 2)
		maxIter2 = 10;
		
	for (int iter = 0; iter < maxIter; iter++) {
		*pScoreOut = 0;
		
		for (int n = 0; n < NN; n++)
			pXtemp[n] = 1 * pX[n];
		
		// d == 3
		for (int t = 0; t < Nt3; t++) {
			if (sparse == 1)
        		score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
      		else
        		score = 1;
      		pXtemp[pIndH3[IDX2R(t,0,3)]] += score * pValH3[t] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
      		pXtemp[pIndH3[IDX2R(t,1,3)]] += score * pValH3[t] * pXout[pIndH3[IDX2R(t,2,3)]] * pXout[pIndH3[IDX2R(t,0,3)]];
     	  	pXtemp[pIndH3[IDX2R(t,2,3)]] += score * pValH3[t] * pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]];
			if (iter == (maxIter-1)) {
				score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]];
				*pScoreOut = *pScoreOut + 3*score*score;
			}
    	}
    	
		// normalization    
    	if (stoc == 0 ) {
    	
      		double pXnorm=0;
      		for (int n2 = 0; n2 < N2; n2++)
        		for (int n1 = 0; n1 < N1; n1++)
          			pXnorm += pXtemp[IDX2R(n1,n2,N2)] * pXtemp[IDX2R(n1,n2,N2)];
          			
      		pXnorm = sqrt(pXnorm);
      		
      		for (int n2 = 0; n2 < N2; n2++)
        		for (int n1 = 0; n1 < N1; n1++)
          			pXout[IDX2R(n1,n2,N2)] = pXtemp[IDX2R(n1,n2,N2)] / pXnorm; 
          			     		
    	} else {
    	
      		for (int iter2 = 0; iter2 < maxIter2; iter2++) {
        		for (int n2 = 0; n2 < N2; n2++) {
          			double pXnorm = 0;
          			for (int n1 = 0; n1 < N1; n1++)
            			pXnorm += pXtemp[IDX2R(n1,n2,N2)] * pXtemp[IDX2R(n1,n2,N2)];
          			pXnorm = sqrt(pXnorm);
          			for (int n1 = 0; n1 < N1; n1++)
            			if (pXnorm !=0 )
              				pXout[IDX2R(n1,n2,N2)] = pXtemp[IDX2R(n1,n2,N2)] / pXnorm;
        		}

        		if (stoc == 2)
		      		for (int n2 = 0; n2 < N2; n2++) {
		        		double pXnorm=0;
		        		for (int n1 = 0; n1 < N1; n1++)
		          			pXnorm += pXtemp[IDX2R(n1,n2,N2)] * pXtemp[IDX2R(n1,n2,N2)];
		        		pXnorm = sqrt(pXnorm);
		        		if (pXnorm != 0)
				      		for (int n1 = 0; n1 < N1; n1++)
				        		pXout[IDX2R(n1,n2,N2)] = pXtemp[IDX2R(n1,n2,N2)] / pXnorm;
		      		}
      		}
      		
    	} 
	}
  	delete[] pXtemp;
}























