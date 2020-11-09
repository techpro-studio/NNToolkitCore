//
//  Header.h
//  
//
//  Created by Alex on 08.11.2020.
//

#ifndef arm_neon_ops_h
#define arm_neon_ops_h


//
// Created by Alex on 07.11.2020.
//

#include "operations.h"
#include <arm_neon.h>
#include "stdlib.h"
#include "math.h"


#if defined __cplusplus
extern "C" {
#endif

void op_mat_mul_n(const float *a, const float *b, float* c, int M, int N, int K);

void op_mat_transp_n(const float *a, float *b, int m, int n);

void op_vec_sub_n(const float *a, const float *b, float *c, int size);

void op_vec_sum_n(const float *a, float* c, int size);

void op_vec_mul_n(const float *a, const float *b, float* c, int size);

float op_vec_dot_n(const float *a, const float *b, int size);

void op_vec_add_n(const float *a, const float * b, float *c, int size);

void op_vec_clamp_n(const float *a, float *c, float min, float max, int size);

void op_vec_max_sc_n(const float *a, float b, float *c, int size);

void op_vec_add_sc_n(const float *a, float b, float *c, int size);

void op_vec_sub_sc_n(const float *a, float b, float *c, int size);

void op_vec_mul_sc_n(const float *a, float b, float *c, int size);

void op_vec_div_sc_n(const float *a, float b, float *c, int size);

void op_vec_neg_n(const float *a, float *c, int size);

void op_vec_div_n(const float *a, const float *b, float *c, int size);

void op_vec_sqrt_n(const float *a, float *c, int size);

void op_vec_exp_n(const float *a, float *c, int size);

void op_vec_tanh_n(const float *a, float *c, int size);

void op_vec_reciprocal_n(const float *a, float *c, int size);

void op_vec_max_n(const float *a, const float *b, float *c, int size);

void op_vec_min_n(const float *a, const float *b, float *c, int size);

#if defined __cplusplus
}
#endif




#endif /* Header_h */
