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

float op_vec_dot_default(const float *a, const float *b, int size){
    float result;
    vDSP_dotpr(a, 1, b, 1, &result, size);
    return result;
}

#define vector_dot_(NUM)  float VectorDot##NUM(const float* a, const float *b, int size)\
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

vector_dot_(2)
vector_dot_(3)
vector_dot_(4)
vector_dot_(8)
vector_dot_(16)

typedef enum {
    two = 2, three = 3, four = 4, eight = 8, sixteen = 16
}optimal_vector_size;

optimal_vector_size get_optimal_vector_size(int size){
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


op_vec_dot_fn op_vec_dot_get_optimized(int size){
    if (size > 4000){
        return op_vec_dot_default;
    }
    optimal_vector_size value = get_optimal_vector_size(size);
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
            return op_vec_dot_default;
    }
}

void op_mat_mul(const float *a, const float *b, float* result, int m, int n, int k, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, beta, result, n);
}

void op_mat_mul_wt(const float *a, const float *b, bool a_transpose, bool b_transpose, float* result, int m, int n, int k, float beta){
    cblas_sgemm(CblasRowMajor, a_transpose ? CblasTrans : CblasNoTrans, b_transpose ? CblasTrans : CblasNoTrans, m, n, k, 1.0, a, k, b, n, beta, result, n);
}

void op_mat_transp(const float *a, float *b, int m, int n) {
    vDSP_mtrans(a, 1, b, 1, m, n);
}

void op_vec_add(const float *a, const float * b, float *result, int size){
    vDSP_vadd(a, 1, b, 1, result, 1, size);
}

void op_vec_sum(const float *a, float* result, int size){
    vDSP_sve(a, 1, result, size);
}

void op_vec_mul(const float *a, const float *b, float* result, int size){
    vDSP_vmul(a, 1, b, 1, result, 1, size);
}

void op_vec_mul_sc(const float *a, float b, float *c, int size) {
    vDSP_vsmul(a, 1, &b, c, 1, size);
}

void op_vec_div_sc(const float *a, float b, float *c, int size){
    vDSP_vsdiv(a, 1, &b, c, 1, size);
}

void op_vec_add_sc(const float *a, float b, float *c, int size){
    vDSP_vsadd(a, 1, &b, c, 1, size);
}

void op_vec_neg(const float *a, float *c, int size){
    vDSP_vneg(a, 1, c, 1, size);
}

void op_vec_sqrt(const float *a, float *c, int size){
    vvsqrtf(c, a, &size);
}

void op_vec_exp(const float *a, float *c, int size) {
    vvexpf(c, a, &size);
}

void op_vec_tanh(const float *a, float *c, int size) {
    vvtanhf(c, a, &size);
}

void op_vec_reciprocal(const float *a, float *c, int size) {
    vvrecf(c, a, &size);
}

void op_vec_div(const float *a, const float *b, float *c, int size) {
    vDSP_vdiv(b, 1, a, 1, c, 1, size);
}

void op_vec_max(const float *a, const float *b, float *c, int size){
    vDSP_vmax(a, 1, b, 1, c, 1, size);
}

void op_vec_min(const float *a, const float *b, float *c, int size){
    vDSP_vmin(a, 1, b, 1, c, 1, size);
}
