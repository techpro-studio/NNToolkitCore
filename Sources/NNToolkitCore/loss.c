//
//  File.c
//  
//
//  Created by Alex on 17.10.2020.
//

#include "loss.h"
#include "operations.h"


float mean_squared_error(float* y, float * y_pred, int size) {
    float buffer[size];
    op_vec_sub(y, y_pred, buffer, size);
    op_vec_mul(buffer, buffer, buffer, size);
    float result = 0.0f;
    op_vec_sum(buffer, &result, size);
    return result / size;
}

void mean_squared_error_derivative(float* y, float * y_pred, float *d_y_pred, int size) {
    op_vec_sub(y, y_pred, d_y_pred, size);
    float k = -2.0 / size;
    op_vec_mul_sc(d_y_pred, k, d_y_pred, size);
}
