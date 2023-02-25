#include <cstddef>
#include <omp.h>
#include <stdlib.h>
#include <iostream>

// This function produces a parallel version of matrix multiplication C = A B using OpenMP.
// The resulting C matrix should be stored in row-major representation.
// Use mmul2 from HW02 task3. You may recycle the code from HW02.

// The matrices A, B, and C have dimension n by n and are represented as 1D arrays.

void mmul(const float* A, const float* B, float* C, const std::size_t n){
    int n_threads=omp_get_max_threads();
    //printf("Number of threads: %d\n", n_threads);

    #pragma omp parallel for num_threads(n_threads) //collapse(3)
        for (size_t i=0; i<n; ++i) {
            for (size_t k=0; k<n; ++k){ 
                for (size_t j=0; j<n; ++j){
                    C[i*n+j]+=A[i*n+k]*B[k*n+j];
                    //int n_threads=omp_get_thread_num();
                    //printf("Number of threads: %d\n", n_threads);
                }
            }
        } 
}
