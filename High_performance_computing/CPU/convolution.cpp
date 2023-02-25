#include <cstddef>
#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include "convolution.h"
// This function does a parallel version of the convolution process
// using OpenMP.

// "image" is an n by n grid stored in row-major order.
// "mask" is an m by m grid stored in row-major order.
// "output" stores the result as an n by n grid in row-major order.

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
size_t r = (m-1)/2; //radius

//image padding
float *image_padded = new float[(n+2*r)*(n+2*r)];
for (size_t i = 0; i < n+2*r; i++){ //rows
    for (size_t j = 0; j < n+2*r; j++){ //columns
        if (i < r || i >= n+r){
            if (j < r || j >= n+r){
                image_padded[i*(n+2*r)+j]=0.0;
            }
            else{
                image_padded[i*(n+2*r)+j]=1.0;
            }
        }
        else{
            if (j < r || j >= n+r){
                image_padded[i*(n+2*r)+j]=1.0;
            }
            else{
                image_padded[i*(n+2*r)+j]=image[(i-r)*n+(j-r)];
            }
        }
    }  
}
    
//printf ("Image_padded\n");
//for (size_t i = 0; i < n+2*r; i++){ //rows
    //for (size_t j = 0; j < n+2*r; j++){ //columns
        //printf("%f ",image_padded[i*(n+2*r)+j]);
    //}
        //printf("\n");
//}

int n_threads=omp_get_max_threads();
//printf ("Output\n");
int index_row_image_padded,index_column_image_padded;

#pragma omp parallel for num_threads(n_threads)
    for(size_t x=0;x<n;++x){
        for(size_t y=0;y<n;++y){
            for(size_t i=0;i<m;++i){
                for(size_t j=0;j<m;++j){
                    index_row_image_padded=x+i;  
                    index_column_image_padded=y+j;                   

                    output[x*n+y]+=image_padded[index_row_image_padded*(n+2*r)+index_column_image_padded]*mask[i*m+j];
                }
            }
                //printf("%f ",output[x*n+y]);
        }
        //printf("\n");
    }
}
