#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    float alpha = 1.0f;
    float beta = 1.0f;
    int m = n;
    int k = n;
    int lda = n;
    int ldb = n;
    int ldc = n;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}