#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "reduce.cuh"

// implements the 'first add during global load' version (Kernel 4) for the parallel reduction
// g_idata is the array to be reduced, and is available on the device.
// g_odata is the array that the reduced results will be written to, and is available on the device.
// expects a 1D configuration.
// uses only dynamically allocated shared memory.
__global__ void reduce_kernel(float* g_idata, float* g_odata, unsigned int n){
extern __shared__ int sdata[];
    // each thread loads 2 elements from global to shared mem and first add of the reduction 
    // perform first level of reduction upon reading from 
    // global memory and writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i   = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	if (i+blockDim.x < n){
        sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
        }
	__syncthreads();
    // do reduction in shared mem
    // reversed loop and threadID-based indexing
    	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    		if (tid < s) {
        		sdata[tid] += sdata[tid + s];
		}
    	__syncthreads();
    	}

    // write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    //std::printf("g_odata %f\n", g_odata[blockIdx.x]);
}


// the sum of all elements in the *input array should be written to the first element of the *input array.
// calls reduce_kernel repeatedly if needed. _No part_ of the sum should be computed on host.
// *input is an array of length N in device memory.
// *output is an array of length = (number of blocks needed for the first call of the reduce_kernel) in device memory.
// configures the kernel calls using threads_per_block threads per block.
// the function should end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float** input, float** output, unsigned int N, unsigned int threads_per_block){

unsigned int num_blocks = (N+2*threads_per_block-1)/(2*threads_per_block);
unsigned int size_input=N;


while(size_input>1){
	
	reduce_kernel<<<num_blocks, threads_per_block,2*threads_per_block*sizeof(float)>>>(*input,*output,size_input);
	size_input=num_blocks;
	num_blocks=(size_input+2*threads_per_block-1)/(2*threads_per_block);
	*input=*output;
}

	
	cudaDeviceSynchronize();
}
