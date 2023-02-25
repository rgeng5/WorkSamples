#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "reduce.cuh"

int main(int argc, char* argv[])
{
  int N = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);

  //array initialization
  float* inputH = (float*)malloc(N * sizeof(float));
  float* outputH = (float*)malloc(N * sizeof(float));
 
  //initialize arrays with random float numbers [-1,1]
  const int RANGE = 1000;
  for (int i = 0; i < N; i++){
  	float randA = rand() % (RANGE + 1);
	inputH[i] = randA / 500.0f-1.0f;
  }


  float *inputD, *outputD;
  cudaMalloc((float**)&inputD,N * sizeof(float));
  cudaMalloc((float**)&outputD,(N+2*threads_per_block-1)/(2*threads_per_block) * sizeof(float));

  cudaMemcpy(inputD, inputH, N * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(outputD, outputH, n * sizeof(float), cudaMemcpyHostToDevice);
    
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//std::printf("%f\n", inputH[0]);

    	reduce(&inputD, &outputD, N, threads_per_block);
   
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Get the elapsed time in milliseconds
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	cudaMemcpy(outputH, outputD, sizeof(float), cudaMemcpyDeviceToHost);


  	std::printf("%f\n", outputH[0]);

	std::printf("%f\n", ms);
  return 0;
}
