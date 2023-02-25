#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "count.cuh"


int main(int argc, char* argv[]){
  int N = atoi(argv[1]);

  thrust::host_vector<int> A(N);
  //initialize array with random float numbers [-1,1]
  const int RANGE = 500;
  for (int i = 0; i < N; i++){
  	int randA = rand() % (RANGE + 1);
	  A[i] = randA;
    //printf("In: %d\n",A[i]);
  }
  
    thrust::device_vector<int> d_in = A;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    values.reserve(N);
    counts.reserve(N);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  
    count(d_in, values, counts);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    int n_values=values.size();
    std::cout << values[n_values-1] << std::endl;
    std::cout << counts[n_values-1] << std::endl;
    printf("%f\n", ms);

  return 0;
}
