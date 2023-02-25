#include <iostream>
#include <math.h>
__global__
void fac(int num,int* n, int* f)
{
  int index = threadIdx.x;
  int factorial = 1;
  int number=index+1;
  for (int i = 1; i<= number; i++){
     factorial *= i;
  }
  f[index] = factorial;
  std::printf("%d!=%d\n",number,f[index]);
}

int main(void)
{
  
  int *N;
  int *F;
  int num_N = 8;
  cudaMallocManaged(&N, num_N*sizeof(int));
  cudaMallocManaged(&F, num_N*sizeof(int));
 
  for (int i = 0; i<num_N; i++){ 
  N[i] = i+1;
  F[i] = 1;
  }
  fac<<<1, 8>>>(num_N,N,F);
  
  cudaDeviceSynchronize();
  
  return 0;
}
