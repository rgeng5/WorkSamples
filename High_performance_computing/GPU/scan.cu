#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "scan.cuh"

__global__ void hillis_steele(float *g_odata, const float *g_idata, unsigned int n) {
    //extern volatile __shared__  float temp[]; // allocated on invocation
    extern __shared__  float temp[]; // allocated on invocation

    int thid = threadIdx.x;
    int pout = 0, pin = 1;
    // load input into shared memory. 
    // **exclusive** scan: shift right by one element and set first output to 0
    //temp[thid] = (thid == 0) ? 0: g_idata[thid-1];
    if (thid<n){
    temp[thid] = g_idata[thid]; //inclusive scan now
    //printf("thid %d g_idata %f\n", thid, g_idata[thid]);
    }
    __syncthreads();

        

    for( int offset = 1; offset<n; offset *= 2 ) {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;

        if (thid >= offset){
            //printf("first %d second %d third %d\n", pout*n+thid, pin*n+thid, pin*n+thid - offset);
            temp[pout*n+thid] = temp[pin*n+thid] + temp[pin*n+thid - offset];
            
        }
	    else{
            temp[pout*n+thid] = temp[pin*n+thid];
        }

    __syncthreads(); // I need this here before I start next iteration 
    }
    
    g_odata[thid] = temp[pout*n+thid]; // write output
}


__global__ void correction(float *g_odata, float *subarray, float *correction) {
    g_odata[threadIdx.x]=subarray[threadIdx.x]+correction[blockIdx.x];
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
    unsigned int num_blocks = (n+threads_per_block-1)/threads_per_block;
    
    //first kernel call to scan subarrays
    //float * output;
    //cudaMallocManaged(&output, n*sizeof(float));
    //printf("input %f\n", input[n-1]);
    hillis_steele<<<num_blocks, threads_per_block, (threads_per_block+n+1)*sizeof(float)>>>(output, input, n);
    //printf("output %f\n", output[n-1]);

    //second kernel call to calculate corrections for subarrays
    float * input2, * output2;
    cudaMallocManaged(&input2, num_blocks*sizeof(float));
    cudaMallocManaged(&output2, num_blocks*sizeof(float));
    for (unsigned int j=1;j<=num_blocks;++j){
        input2[j]=output[j*threads_per_block-1];
    }
    input2[0]=0;
    cudaMallocManaged(&output2, num_blocks*sizeof(float));
    hillis_steele<<<(num_blocks+threads_per_block-1)/threads_per_block, num_blocks, (threads_per_block+num_blocks+1)*sizeof(float)>>>(output2, input2, num_blocks);
    //printf("finished 2nd");

    //third kernel call to correct subarrays
    float * output3;
    cudaMallocManaged(&output3, n*sizeof(float));
    correction<<<num_blocks, threads_per_block>>>(output3, output, output2);

    *output = *output3;
}