//
//  activation.h
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef activation_h
#define activation_h

#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif

typedef void(*ActivationFunctionImpl)(void *, const float *, float *, int);

typedef void(*ActivationImplementerDestroy)(void *);

typedef struct ActivationFunctionStruct* ActivationFunction;

ActivationFunction ActivationFunctionCreate(int size, ActivationImplementerDestroy destroyFn, void *implementer, ActivationFunctionImpl function);

ActivationFunction ActivationFunctionCreateSimple(int size, ActivationFunctionImpl function);

void ActivationFunctionDestroy(ActivationFunction );

void ActivationFunctionApply(ActivationFunction , const float *, float *);


#if defined __cplusplus
}
#endif

#endif /* activation_h */
