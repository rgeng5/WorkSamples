#include <iostream>
#include <math.h>
__global__
void func(int a, int* dA) //Each thread computes ax+y and writes the result in one distinct entry of the dA array
{
  int x = threadIdx.x;
  int y = blockIdx.x;
  dA[8*y+x] = a*x+y;
}

int main(void)
{
  //From the host, allocates an array of 16 ints on the device called dA
  int* dA;
  int size = 16 * sizeof(int);
  cudaMalloc((int**)&dA, size);
 
  //generate a randomly
  const int RANGE = 100;
  int a = rand() % (RANGE + 1);

  //Launches a kernel with 2 blocks, each block having 8 threads
  func<<<2, 8>>>(a, dA);
  
  cudaDeviceSynchronize();

  //Copies back the data stored in the device array dA into a host array called hA
  int* hA = (int*)malloc(size);
  cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);
  
  //Prints (from the host) the 16 values stored in the host array separated by a single space each
  for (int i = 0; i < 16; i++)
	  std::printf("%d ", hA[i]);
	std::printf("\n"); 
  
  // Free global memory
  cudaFree(dA);
  
  return 0;
}
