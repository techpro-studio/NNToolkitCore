//
//  activation.c
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "activation.h"
#include "stdlib.h"
#include <string.h>

struct ActivationFunctionStruct {
    void *implementer;
    int inputSize;
    ActivationImplementerDestroy destroyFn;
    ActivationFunctionImpl function;
};

void ActivationFunctionApply(ActivationFunction  filter, const float * input, float * output){
    filter->function(filter->implementer, input, output, filter->inputSize);
}

void ActivationFunctionDestroy(ActivationFunction filter){
    if (filter->implementer){
        filter->destroyFn(filter->implementer);
    }
    free(filter);
}

ActivationFunction ActivationFunctionCreate(int size, ActivationImplementerDestroy destroyFn,void *implementer, ActivationFunctionImpl function) {
    ActivationFunction filter = malloc(sizeof(struct ActivationFunctionStruct));
    filter->implementer = implementer;
    filter->inputSize = size;
    filter->destroyFn = destroyFn;
    filter->function = function;
    return filter;
}

ActivationFunction ActivationFunctionCreateSimple(int size, ActivationFunctionImpl function){
    return ActivationFunctionCreate(size, NULL, NULL, function);
}
