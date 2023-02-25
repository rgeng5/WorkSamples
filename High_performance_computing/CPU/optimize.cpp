#include <cstddef>
#include "optimize.h"
#include <stdlib.h>
#include <iostream>

// This file provides some macros and utils
// that can be used for optimizations using ILP
// (instruction level parallelism).
// The definition of OP, IDENT, and data_t can be changed
// to measure the performance for different optimizations,
// i.e. OP could be + or *; IDENT could be 0 or 1;
// data_t could be int or float.

data_t *get_vec_start(vec *v){
    data_t *d = new data_t[v->len];
    d=v->data;
    return d;
}

void optimize1(vec *v, data_t *dest){
    int length = v->len;
    data_t *d = get_vec_start(v); 
    //printf("%p\n", d);
    data_t temp = IDENT;
//#pragma simd
    for (int i = 0; i < length; i++){ 
        temp = temp OP d[i];
        //printf("%d ", d[i]);
    }
    *dest = temp;
}

void optimize2(vec *v, data_t *dest){
    int length = v->len;
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    // Combine 2 elements at a time 
    for (i = 0; i < limit; i += 2) {
        x = (x OP d[i]) OP d[i + 1];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}

void optimize3(vec *v, data_t *dest){
    int length = v->len;
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    /* Combine 2 elements at a time */
    for (i = 0; i < limit; i += 2) {
        x = x OP(d[i] OP d[i + 1]);
    }
    /* Finish any remaining elements */
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}

void optimize4(vec *v, data_t *dest){
    long length = v->len;
    long limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    long i;
    // Combine 2 elements at a time
    for (i = 0; i < limit; i += 2) {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1;
}

void optimize5(vec *v, data_t *dest){
    long length = v->len;
    long limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    data_t x2 = IDENT; //K=3
    long i;
    // Combine 3 elements at a time
    for (i = 0; i < limit; i += 3) { //L=3
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
        x2 = x2 OP d[i + 2];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1 OP x2;
}
}

