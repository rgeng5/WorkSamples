#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
  
    	extern __shared__ float s[];

//    	float *mask_ptr = s;   //size=2R+1                     
//	float *image_ptr = (float*)&mask_ptr[2*R+1]; //size=n+2R  
//	float *output_ptr = (float*)&image_ptr[n+2*R];    //size=n 
	
	//initialize pointers stored in the shared memory	
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int t = threadIdx.x;
	unsigned int b = blockIdx.x;

	for (int j = 0; j<2*R+1; j++){
		s[j]=mask[j];
	}	
	int mask_size=2*R+1;
	if (t-R<0){
		if (b*blockDim.x-R<0){
			s[mask_size+t]=1.0f;
		}else{
			s[mask_size+t]=image[index-R];
		}
	}else if (t+R>=blockDim.x){
		if (blockDim.x*b + t + R >= n){
			s[mask_size+t+R]=1.0f;
		}else{
			s[mask_size+t+R]=image[index+R];
		}
	}else{
		s[mask_size+t+R]=image[index];
	}

	__syncthreads();

	int output_start=mask_size+blockDim.x+2*R;
	for (int i = 0; i<2*R+1; i++){
		s[output_start+t] += s[mask_size+i+t]*s[i];

	//std::printf("image %f mask %f output %f\n", image_ptr[i+t], mask_ptr[i], output_ptr[index]);
	}
	__syncthreads();
	output[index]=s[output_start+t];

}


__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
int blocks = (n + threads_per_block - 1)/threads_per_block;
int sizeShared = sizeof(float)*(threads_per_block*2+2*R+2*R+1);

stencil_kernel<<<blocks,threads_per_block, sizeShared>>>(image, mask, output, n,R);

cudaDeviceSynchronize();


}

