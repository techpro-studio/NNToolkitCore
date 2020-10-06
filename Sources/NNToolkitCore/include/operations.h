//
//  helpers.h
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef helpers_h
#define helpers_h

#include <stdio.h>


void MatMul(const float *a, const float *b, float* result, int m, int n, int k, float beta);

void MatMul2(const float *a, const float *b, float* result, int m, int n, int k);

void MatTrans(const float *a, float *b, int m, int n);

void VectorAdd(const float *a, const float *b, float *result, int size);

typedef float (*VectorDotF)(const float *a, const float *b, int size);

VectorDotF GetOptimized(int size);

float VectorDotDefault(const float *a, const float *b, int size);

void VectorMul(const float *a, const float *b, float* result, int size);

void VectorAddS(const float *a, float b, float *c, int size);

void VectorMulS(const float *a, float b, float *c, int size);

void VectorNeg(const float *a, float *c, int size);

void VectorDiv(const float *a, const float *b, float *c, int size);

void VectorSqrt(const float *a, float *c, int size);

void VectorExp(const float *a, float *c, int size);

void VectorTanh(const float *a, float *c, int size);

void VectorReciprocal(const float *a, float *c, int size);

void VectorMax(const float *a, const float *b, float *c, int size);

void VectorMin(const float *a, const float *b, float *c, int size);



#endif /* helpers_h */
