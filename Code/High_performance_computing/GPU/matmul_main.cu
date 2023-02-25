#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matmul.cuh"

int main(int argc, char* argv[])
{
  int n = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);

  //array initialization
  int size = n * sizeof(float);
  float* hA = (float*)malloc(size);
  float* hB = (float*)malloc(size);

 
  //initialize arrays with random float numbers [-1 1]
  const int RANGE = 1000;

  for (int i = 0; i < n; i++){
  	float randA = rand() % (RANGE + 1);
  	float randB = rand() % (RANGE + 1);
	hA[i] = randA / 500.0f-1.0f;
	hB[i] = randB / 500.0f-1.0f;
  }

  float *dA, *dB, *dC;
  cudaMalloc((float**)&dA,size);
  cudaMalloc((float**)&dB,size);
  cudaMalloc((float**)&dC,size);
  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
  
  const int blocksPerGrid = ( n + threads_per_block - 1 ) / threads_per_block;
    
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    	matmul_kernel<<<blocksPerGrid, threads_per_block>>>(dA, dB, dC, n);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Get the elapsed time in milliseconds
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	std::printf("%f\n", ms);

	cudaMemcpy(hB, dB, size, cudaMemcpyDeviceToHost);


  std::printf("%f\n", hB[n-1]);

  return 0;
}
