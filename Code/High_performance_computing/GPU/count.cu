
// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts){

    const int N = d_in.size();    
    
    //sort the array
    thrust::device_vector<int> d_sorted = d_in;
    thrust::sort(d_sorted.begin(), d_sorted.begin() + N);

    //thrust::copy(d_sorted.begin(), d_sorted.end(), std::ostream_iterator<int>(std::cout, ","));
    //std::cout << std::endl;
    
    //find the number of jumps needed
    int dim=thrust::inner_product(d_sorted.begin(), d_sorted.end() - 1, d_sorted.begin() + 1, 0, 
                                    thrust::plus<int>(), thrust::not_equal_to<int>()) + 1;

    //printf("dim %d\n",dim);
    
    values.resize(dim);
    counts.resize(dim);
    
    //count for each value
    thrust::device_vector<int> B(N);
    thrust::fill(B.begin(), B.end(), 1);
    thrust::reduce_by_key(d_sorted.begin(), d_sorted.end(), B.begin(), values.begin(), counts.begin());

    //thrust::copy(values.begin(), values.end(), std::ostream_iterator<int>(std::cout, ","));
    //std::cout << std::endl;

    //thrust::copy(counts.begin(), counts.end(), std::ostream_iterator<int>(std::cout, ","));
    //std::cout << std::endl;


}