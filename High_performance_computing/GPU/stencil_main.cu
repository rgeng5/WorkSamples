#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "stencil.cuh"

int main(int argc, char* argv[])
{
  int n = atoi(argv[1]);
  int R = atoi(argv[2]);
  int threads_per_block = atoi(argv[3]);

  //array initialization
  float* imageH = (float*)malloc(n * sizeof(float));
  float* outputH = (float*)malloc(n * sizeof(float));
  float* maskH = (float*)malloc((2*R+1) * sizeof(float));
 
  //initialize arrays with random float numbers [-1,1]
  const int RANGE = 1000;
  for (int i = 0; i < n; i++){
  	float randA = rand() % (RANGE + 1);
	imageH[i] = randA / 500.0f-1.0f;	
  }

  for (int i = 0; i < 2*R+1; i++){
  	float randB = rand() % (RANGE + 1);
	maskH[i] = randB / 500.0f-1.0f;
  }

  float *imageD, *outputD, *maskD;
  cudaMalloc((float**)&imageD,n * sizeof(float));
  cudaMalloc((float**)&outputD,n * sizeof(float));
  cudaMalloc((float**)&maskD,(2*R+1) * sizeof(float));

  cudaMemcpy(imageD, imageH, n * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(outputD, outputH, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(maskD, maskH, (2*R+1) * sizeof(float), cudaMemcpyHostToDevice);
    
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    	stencil(imageD, maskD, outputD,n,R,threads_per_block);
   
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Get the elapsed time in milliseconds
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	std::printf("%f\n", ms);

	cudaMemcpy(outputH, outputD, n * sizeof(float), cudaMemcpyDeviceToHost);


 // std::printf("image %f %f ;mask %f %f %f ;out %f\n", imageH[0], imageH[1], maskH[0], maskH[1], maskH[2], outputH[0]); 
  std::printf("%f\n", outputH[n-1]);

  return 0;
}
