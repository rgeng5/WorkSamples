#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mmul.h"
//#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main(int argc, char* argv[]){
  int N = atoi(argv[1]);
  int n_tests = atoi(argv[2]);

  //array initialization
  //float* inputH = (float*)malloc(N * sizeof(float));
  //float* outputH = (float*)malloc(N * sizeof(float));
 
  float *A, *B, *C;
  cudaMallocManaged(&A, N*N*sizeof(float));
  cudaMallocManaged(&B, N*N*sizeof(float));
  cudaMallocManaged(&C, N*N*sizeof(float));

  //initialize arrays with random float numbers [-1,1]
  const int RANGE = 1000;
  for (int i = 0; i < N*N; i++){
  	float randA = rand() % (RANGE + 1);
	  A[i] = randA / 500.0f-1.0f;
      B[i] = randA / 500.0f-1.0f;
      C[i] = randA / 500.0f-1.0f;
      //printf ("%f\n", A[i]);
      //rintf ("%f\n", B[i]);
      //printf ("%f\n", C[i]);
  }

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    cudaEventRecord(start);
  
      for (int j=0; j<n_tests; ++j){
          mmul(handle, A, B, C, N);
      }
  
    cudaDeviceSynchronize();
  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);


	std::printf("%f\n", ms/n_tests);
    
    //Check for correctness
    //for (int j = 0; j < N; j++) {
        //for (int i = 0; i < N; i++) {

            //printf ("%f", A[j*N+1]);
            //printf ("%f", B[j*N+1]);
            //printf ("C %f\n", C[j*N+1]);
        //}
        //printf ("\n");
    //}
    
  
  return 0;
}