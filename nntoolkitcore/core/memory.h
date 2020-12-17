//
// Created by Alex on 20.11.2020.
//

#ifndef memory_h
#define memory_h

#include "stdlib.h"

float *f_malloc(unsigned long size);

void f_copy(float *dst, const float *src, unsigned long size);

void f_zero(float *a, unsigned long size);

#endif //memory_h
