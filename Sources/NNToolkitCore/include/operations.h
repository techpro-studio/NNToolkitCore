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
#include "stdbool.h"
#include <TargetConditionals.h>

#define S_LOOP_START(size, var) for (int var = 0; var < size; ++var) {
#define S_LOOP_END }

#if TARGET_OS_MAC
    #include <dispatch/dispatch.h>
    #define P_LOOP_START(size, var) dispatch_apply(size, DISPATCH_APPLY_AUTO, ^(size_t var) {
    #define P_LOOP_END });
#else
    #define P_LOOP_START S_LOOP_START
    #define P_LOOP_END S_LOOP_END
#endif


void MatMul(const float *a, const float *b, float* result, int m, int n, int k, float beta);

void MatMul2(const float *a, const float *b, float* result, int m, int n, int k);

void MatMul3(const float *a, const float *b, bool a_transpose, bool b_transpose, float* result, int m, int n, int k, float beta);

void MatTrans(const float *a, float *b, int m, int n);

void VectorAdd(const float *a, const float *b, float *result, int size);

void VectorSum(const float *a, float* result,  int size);

typedef float (*VectorDotF)(const float *a, const float *b, int size);

VectorDotF GetOptimized(int size);

float VectorDotDefault(const float *a, const float *b, int size);

void VectorMul(const float *a, const float *b, float* result, int size);

void VectorAddS(const float *a, float b, float *c, int size);

void VectorMulS(const float *a, float b, float *c, int size);

void VectorDivS(const float *a, float b, float *c, int size);

void VectorNeg(const float *a, float *c, int size);

void VectorDiv(const float *a, const float *b, float *c, int size);

void VectorSqrt(const float *a, float *c, int size);

void VectorExp(const float *a, float *c, int size);

void VectorTanh(const float *a, float *c, int size);

void VectorReciprocal(const float *a, float *c, int size);

void VectorMax(const float *a, const float *b, float *c, int size);

void VectorMin(const float *a, const float *b, float *c, int size);



#endif /* helpers_h */
