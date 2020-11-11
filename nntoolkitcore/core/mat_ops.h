//
// Created by Alex on 11.11.2020.
//

#ifndef op_mat_mul_h
#define op_mat_mul_h

void op_mat_mul(const float *a, const float *b, float* result, int M, int N, int K);

void op_mat_transp(const float *a, float *b, int M, int N);

#endif //op_mat_mul_h
