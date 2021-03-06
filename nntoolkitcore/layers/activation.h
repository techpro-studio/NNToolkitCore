//
//  activation.h
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef activation_h
#define activation_h


#include "stdbool.h"

#if defined __cplusplus
extern "C" {
#endif

typedef void(*ActivationFunctionImpl)(void *, const float *, float *, int);
typedef void(*ActivationFunctionDerivative)(void *, const float *, const float*, float *, int);

typedef void(*ActivationImplementerDestroy)(void *);

typedef struct ActivationFunctionStruct* ActivationFunction;

ActivationFunction ActivationFunctionCreate(int size, ActivationImplementerDestroy destroy_fn, void *implementer, ActivationFunctionImpl function, ActivationFunctionDerivative derivative, ActivationFunctionDerivative cached_derivative);

void ActivationFunctionDestroy(ActivationFunction filter);

void ActivationFunctionApply(ActivationFunction filter, const float *input, float *output);

void ActivationFunctionCalculateGradient(ActivationFunction filter, const float *z, const float *a, const float* d_out, float *output);



#if defined __cplusplus
}
#endif

#endif /* activation_h */
