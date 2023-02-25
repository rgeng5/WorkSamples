#include <cstddef>
#include <cstdio>
#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include "msort.h"
#include <ratio>
#include <cmath>

// This function does a merge sort on the input array "arr" of length n.
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling

/* Function to sort an array using insertion sort*/
void insertionSort(int* arr, const std::size_t lo, const std::size_t hi){
    std::size_t i, j;
    for (i = lo; i < hi; ++i) {
        int tmp = arr[i + 1];
        j = i + 1;
  
        while (j > lo && arr[j - 1] > tmp) {
            arr[j] = arr[j - 1];
            j--;
        }

        //printf ("j %d arr[i] %d\n", j, arr[j]);
        arr[j] = tmp;
    }
}

void merge(int* A, std::size_t lo, std::size_t mid, std::size_t hi, int* B){
    std::size_t i = lo; 
    std::size_t j = mid;
 
    // While there are elements in the left or right runs...
    for (std::size_t k = lo; k < hi; ++k) {
        // If left run head exists and is <= existing right run head.
        if (i < mid && (j >= hi || A[i] <= A[j])) {
            B[k] = A[i];
            i = i + 1;
        } else {
            B[k] = A[j];
            j = j + 1;
        }
    }
}

void mergeSort(int* B,std::size_t lo, std::size_t hi, int* A, const std::size_t threshold){
    if (hi - lo < threshold)
        insertionSort(B, lo, hi);
    else
    {
        // split the run longer than 1 item into halves
        const std::size_t mid = (lo+hi)/2;
        // recursively sort both runs from array A[] into B[]
        #pragma omp task
        {
            mergeSort(A,lo,mid,B,threshold);
        }

        #pragma omp task
        {
            mergeSort(A,mid,hi,B,threshold);
        }
        // merge the resulting runs from array B[] into A[]
        #pragma omp taskwait
        {
            merge(B,lo,mid,hi,A);
        }
    }
}

void CopyArray(int* A, std::size_t lo, std::size_t hi, int* B)
{
    for(std::size_t k = lo; k < hi; ++k)
        B[k] = A[k];
}

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    int* A = arr;
    int* B = new int[n];
    CopyArray(A, 0, n, B);           // one time copy of A[] to B[]

    int n_threads=omp_get_max_threads();
    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp single
        {
            mergeSort(B, 0, n, A, threshold);   // sort data from B[] into A[]
        }
    }
}