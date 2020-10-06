//
//  helpers.c
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "operations.h"
#include <Accelerate/Accelerate.h>
#include <simd/simd.h>

float VectorDotDefault(const float *a, const float *b, int size){
    float result;
    vDSP_dotpr(a, 1, b, 1, &result, size);
    return result;
}

#define vector_dot(NUM)  float VectorDot##NUM(const float* a, const float *b, int size)\
{\
    float sum = 0.0f;\
    int iterations = size / NUM;\
    for (int i = 0; i < iterations; ++i)\
    {\
        simd_float##NUM _a = ((simd_float##NUM*) a)[i];\
        simd_float##NUM _b = ((simd_float##NUM*) b)[i];\
        sum += simd_dot(_a, _b);\
    }\
    int left = size % NUM;\
    for (int i = 0; i < left; ++i)\
    {\
        sum += a[iterations * NUM + i] * b[iterations * NUM + i];\
    }\
    return sum;\
}

vector_dot(2)
vector_dot(3)
vector_dot(4)
vector_dot(8)
vector_dot(16)


typedef enum {
    two = 2, three = 3, four = 4, eight = 8, sixteen = 16
}optimal_vector_size;

optimal_vector_size getOptimalVectorSize(int size){
    int optimalIndex = 0;
    int smallestValue = 230;
    optimal_vector_size values [] = {two, three, four, eight, sixteen};
    int full[5] = {size / 2, size / 3, size / 4, size / 8, size / 16};
    int last[5] = {size % 2, size % 3, size % 4, size % 8, size % 16};
    for (int i = 0; i < 5; ++i){
        if (full[i] == 0){
            continue;
        }
        int sum = full[i] + last[i];
        if (sum < smallestValue){
            smallestValue = sum;
            optimalIndex = i;
        }
    }
    return values[optimalIndex];
}


VectorDotF GetOptimized(int size){
    if (size > 4000){
        return VectorDotDefault;
    }
    optimal_vector_size value = getOptimalVectorSize(size);
    switch (value) {
        case two:
            return VectorDot2;
        case three:
            return VectorDot3;
        case four:
            return VectorDot4;
        case eight:
            return VectorDot8;
        case sixteen:
            return VectorDot16;
        default:
            return VectorDotDefault;
    }
}

void MatMul(const float *a, const float *b, float* result, int m, int n, int k, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, beta, result, n);
}

void MatMul2(const float *a, const float *b, float* result, int m, int n, int k) {
    vDSP_mmul(a, 1, b, 1, result, 1, m, n, k);
}

void MatTrans(const float *a, float *b, int m, int n) {
    vDSP_mtrans(a, 1, b, 1, m, n);
}

void VectorAdd(const float *a, const float * b,float *result, int size){
    vDSP_vadd(a, 1, b, 1, result, 1, size);
}






void VectorMul(const float *a, const float *b, float* result, int size){
    vDSP_vmul(a, 1, b, 1, result, 1, size);
}

void VectorMulS(const float *a, float b, float *c, int size) {
    vDSP_vsmul(a, 1, &b, c, 1, size);
}

void VectorAddS(const float *a, float b, float *c, int size){
    vDSP_vsadd(a, 1, &b, c, 1, size);
}

void VectorNeg(const float *a, float *c, int size){
    vDSP_vneg(a, 1, c, 1, size);
}

void VectorSqrt(const float *a, float *c, int size){
    vvsqrtf(c, a, &size);
}

void VectorExp(const float *a, float *c, int size) {
    vvexpf(c, a, &size);
}

void VectorTanh(const float *a, float *c, int size) {
    vvtanhf(c, a, &size);
}

void VectorReciprocal(const float *a, float *c, int size) {
    vvrecf(c, a, &size);
}

void VectorDiv(const float *a, const float *b, float *c, int size) {
    vDSP_vdiv(b, 1, a, 1, c, 1, size);
}

void VectorMax(const float *a, const float *b, float *c, int size){
    vDSP_vmax(a, 1, b, 1, c, 1, size);
}

void VectorMin(const float *a, const float *b, float *c, int size){
    vDSP_vmin(a, 1, b, 1, c, 1, size);
}
