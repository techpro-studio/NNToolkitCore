//
//  helpers.c
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "operations.h"
#include <Accelerate/Accelerate.h>

void MatMul(const float *a, const float *b, float* result, int m, int n, int k, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, beta, result, n);
}

void MatMul2(const float *a, const float *b, float* result, int m, int n, int k) {
    vDSP_mmul(a, 1, b, 1, result, 1, m, n, k);
}

void VectorAdd(const float *a, const float * b, float *result, int size){
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
