#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>

// 1-D row major index
#define IDX2R(i,j,w) (((i)*(w))+(j))
// 1-D column major index
#define IDX2C(i,j,h) (((j)*(h))+(i))

// Prototypes
void saveMatrix(double *A, int h, int w, char *filename);
void saveMatrix(int *A, int h, int w, char *filename);
void tensorMatching(double* pX, int N1, int N2, int* pIndH3, double* pValH3, int Nt3 ,
          			int nIter, int sparse, int stoc, double* pXout, double* pScoreOut);

int main(void) {
	int nP1 = 50;
	int nP2 = nP1;
	int nT = 50*nP1;
	int nNN = 300;

	int *indH = (int*)malloc(3*nNN*nT*sizeof(int));
	double *valH = (double*)malloc(nNN*nT*sizeof(double));

	std::ifstream matfile;
	matfile.open("indH.mat");

	for (int i = 0; i < 3*nNN*nT; i++) {
		matfile >> indH[i];
	}
	matfile.close();

	matfile.open("valH.mat");

	for (int i = 0; i < nNN*nT; i++) {
		matfile >> valH[i];
	}

	std::cout << "Outputting indH serial ..." << std::endl;
	saveMatrix(indH, nNN*nT, 3, "indH_serial.mat");
	std::cout << "Outputting valH serial ..." << std::endl;
	saveMatrix(valH, nNN*nT, 1, "valH_serial.mat");

	double *X = (double*)malloc(nP2*nP1*sizeof(double));
	for (int i=0; i < nP2*nP1; i++)
			X[i] = (double)(1.0/(double)nP2);

	double *X2 = (double*)malloc(nP2*nP1*sizeof(double));

	double score;

	tensorMatching(X, nP2, nP1, indH, valH, nNN*nT, 100, 1, 2, X2, &score);


	std::cout << "Score = " << score << std::endl;
	std::cout << "Outputting X2 serial ..." << std::endl;
	saveMatrix(X2, nP2, nP1, "X2_serial.mat");


}

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
		

	*pScoreOut = 0;
		
	for (int n = 0; n < NN; n++)
		pXtemp[n] = 1 * pX[n];
			

	std::cout << "Outputting Xtemp at iter 0 before d==3 ..." << std::endl;
	saveMatrix(pXtemp, N1, N2, "XtempIter0_serial_before.mat");
		
	// d == 3
	for (int t = 0; t < Nt3; t++) {
		if (sparse == 1)
        	score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]; 
      	else
        	score = 1;
        
        printf("t=%d,before: Xtemp[%d]=%.16f, Xtemp[%d]=%.16f, Xtemp[%d]=%.16f\n",t,pIndH3[IDX2R(t,0,3)],pXtemp[pIndH3[IDX2R(t,0,3)]],pIndH3[IDX2R(t,1,3)],pXtemp[pIndH3[IDX2R(t,1,3)]],pIndH3[IDX2R(t,2,3)],pXtemp[pIndH3[IDX2R(t,2,3)]]);
        		
  		pXtemp[pIndH3[IDX2R(t,0,3)]] += (score * pValH3[t] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]]); 
  		pXtemp[pIndH3[IDX2R(t,1,3)]] += (score * pValH3[t] * pXout[pIndH3[IDX2R(t,2,3)]] * pXout[pIndH3[IDX2R(t,0,3)]]);
 	  	pXtemp[pIndH3[IDX2R(t,2,3)]] += (score * pValH3[t] * pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]]);
 	  	
 	  	printf("t=%d,after: Xtemp[%d]=%.16f, Xtemp[%d]=%.16f, Xtemp[%d]=%.16f\n",t,pIndH3[IDX2R(t,0,3)],pXtemp[pIndH3[IDX2R(t,0,3)]],pIndH3[IDX2R(t,1,3)],pXtemp[pIndH3[IDX2R(t,1,3)]],pIndH3[IDX2R(t,2,3)],pXtemp[pIndH3[IDX2R(t,2,3)]]);

     
     	//printf("t=%d,score=%f,valH[t]=%f,indH[t,0]=%d,indH[t,1]=%d,indH[t,2]=%d\n",t,score,pValH3[t],pIndH3[IDX2R(t,0,3)],pIndH3[IDX2R(t,1,3)],pIndH3[IDX2R(t,2,3)]);
     	  	
		/*if (iter == (maxIter-1)) {
			score = pXout[pIndH3[IDX2R(t,0,3)]] * pXout[pIndH3[IDX2R(t,1,3)]] * pXout[pIndH3[IDX2R(t,2,3)]];
			*pScoreOut = *pScoreOut + 3*score*score;
		}*/
	}
    	
    /* DEBUG */
	std::cout << "Outputting Xtemp at iter 0 after d==3..." << std::endl;
	saveMatrix(pXtemp, N1, N2, "XtmepIter0_serial_after.mat");
    	

}

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
