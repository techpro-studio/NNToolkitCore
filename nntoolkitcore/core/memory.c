//
// Created by Alex on 21.11.2020.
//

#include "memory.h"
#include "string.h"
#include "stdlib.h"

void* malloc_zeros(size_t __size) {
    void *ptr = malloc(__size);
    memset(ptr, 0, __size);
    return ptr;
}

float *f_malloc(unsigned long size) {
    return (float *) malloc_zeros(size * sizeof(float));
}

void f_copy(float *dst, const float *src, unsigned long size) {
    memcpy(dst, src, size * sizeof(float));
}

void f_zero(float *a, unsigned long size) {
    memset(a, 0, size * sizeof(float));
}
