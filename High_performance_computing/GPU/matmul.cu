#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
  
    // 2D Thread Index; computing C[ty][tx]
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue will end up storing the value of C[ty][tx]  

    float Pvalue = 0;

    for (int k = 0; k < n; k++)  { 
         float Melement = A[ty * n + k];
         float Nelement = B[k * n + tx];
         Pvalue += Melement * Nelement;
    }

    // Write matrix to device memory; each thread one element
    C[ty * n + tx] = Pvalue;

}
