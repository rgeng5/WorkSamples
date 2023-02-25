#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main(int argc, char* argv[]){
  int N = atoi(argv[1]);

  thrust::host_vector<float> A(N);
  //initialize array with random float numbers [-1,1]
  const int RANGE = 1000;
  for (int i = 0; i < N; i++){
  	float randA = rand() % (RANGE + 1);
	  A[i] = randA / 500.0f-1.0f;
  }
  thrust::device_vector<float> dA = A;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  
    float results=thrust::reduce(dA.begin(), dA.end());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);


	  std::printf("%f\n", results);
    std::printf("%f\n", ms);

  return 0;
}