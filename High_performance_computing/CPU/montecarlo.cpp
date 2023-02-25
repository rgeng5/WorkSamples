#include <cstddef>
#include <omp.h>
#include "montecarlo.h"
#include <iostream>

// this function returns the number of points that lay inside
// a circle using OpenMP parallel for.
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.

int montecarlo(const size_t n, const float *x, const float *y, const float radius){
    size_t nThreadsUsed = omp_get_max_threads();
    int count = 0;
    //printf("threads %d\n", nThreadsUsed);

    #pragma omp parallel num_threads(nThreadsUsed)
    {
        #pragma omp for simd reduction(+:count)
            for (size_t i=0; i<n; i++){
                if (x[i] * x[i] + y[i] * y[i]  <= radius * radius){
                    //printf("test %f\n", x[i] * x[i] + y[i] * y[i]);
                    count += 1;
                }
            }       
    }
    //printf("count %d\n", count);
    return count;
}
