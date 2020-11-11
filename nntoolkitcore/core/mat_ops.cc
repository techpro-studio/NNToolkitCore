//
// Created by Alex on 11.11.2020.
//

#include "mat_ops.h"
#include <Accelerate/Accelerate.h>


void op_mat_mul(const float *a, const float *b, float* result, int m, int n, int k) {
    vDSP_mmul(a, 1, b, 1, result, 1, m, n, k);
//    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, result, n);
}

void op_mat_transp(const float *a, float *b, int m, int n) {
    vDSP_mtrans(a, 1, b, 1, m, n);
}