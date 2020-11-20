//
// Created by Alex on 21.11.2020.
//

#include "memory.h"
#include "string.h"
#include "stdlib.h"


float *f_malloc(unsigned long size) {
    float *ptr = (float *)malloc(size * sizeof(float));
    f_zero(ptr, size);
    return ptr;
}

void f_copy(float *dst, const float *src, unsigned long size) {
    memcpy(dst, src, size * sizeof(float));
}

void f_zero(float *a, unsigned long size) {
    memset(a, 0, size * sizeof(float));
}
