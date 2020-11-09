//
//  helpers.h
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef ops_h
#define ops_h


#if defined __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "stdbool.h"


void op_mat_mul(const float *a, const float *b, float* result, int m, int n, int k);

void op_mat_transp(const float *a, float *b, int m, int n);

void op_vec_sub(const float *a, const float *b, float *result, int size);

void op_vec_sum(const float *a, float* result, int size);

void op_vec_mul(const float *a, const float *b, float* result, int size);


void op_vec_add(const float *a, const float *b, float *result, int size);

float op_vec_dot(const float *a, const float *b, int size);

void op_vec_clamp(const float *a, float *c, float min, float max, int size);

void op_vec_max_sc(const float *a, float b, float *c, int size);

void op_vec_add_sc(const float *a, float b, float *c, int size);

void op_vec_sub_sc(const float *a, float b, float *c, int size);

void op_vec_mul_sc(const float *a, float b, float *c, int size);

void op_vec_div_sc(const float *a, float b, float *c, int size);

void op_vec_neg(const float *a, float *c, int size);

void op_vec_div(const float *a, const float *b, float *c, int size);

void op_vec_sqrt(const float *a, float *c, int size);

void op_vec_exp(const float *a, float *c, int size);

void op_vec_tanh(const float *a, float *c, int size);

void op_vec_reciprocal(const float *a, float *c, int size);

void op_vec_max(const float *a, const float *b, float *c, int size);

void op_vec_min(const float *a, const float *b, float *c, int size);

#if defined __cplusplus
}
#endif

#endif /* ops_h */
