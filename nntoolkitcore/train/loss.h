//
//  File.h
//  
//
//  Created by Alex on 17.10.2020.
//

#ifndef loss_h
#define loss_h

#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif

float mean_squared_error(float *y, float *y_pred, int size, int batch);

void mean_squared_error_derivative(float *y, float *y_pred, float *d_y_pred, int size, int batch);

float categorical_crossentropy(float *y, float *y_pred, int c, int batch);

void categorical_crossentropy_derivative(float *y, float *y_pred, float *d_y_pred, int c, int batch);

#if defined __cplusplus
}
#endif

#endif /* File_h */
