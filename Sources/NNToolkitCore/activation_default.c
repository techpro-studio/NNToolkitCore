//
//  activation_default_functions.c
//  audio_test
//
//  Created by Alex on 28.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "activation_default.h"
#include "operations.h"
#include "string.h"
#include "stdlib.h"
#include <dispatch/dispatch.h>



//SIGMOID



void activation_sigmoid(void *implementer, const float *input, float *output, int size) {
    VectorNeg(input, output, size);
    VectorExp(output, output, size);
    VectorAddS(output, 1, output, size);
    VectorReciprocal(output, output, size);
}

ActivationFunction ActivationFunctionCreateSigmoid(int inputSize){
    return ActivationFunctionCreateSimple(inputSize, activation_sigmoid);
}



// TANH



void activation_tanh(void *implementer, const float *input, float *output, int size){
    VectorTanh(input,output, size);
}

ActivationFunction ActivationFunctionCreateTanh(int inputSize){
    return ActivationFunctionCreateSimple(inputSize, activation_tanh);
}



//Identity



void activation_identity(void *implementer, const float *input, float *output, int size){
    if (input == output){
        return;
    }
    memcpy(output, input, size * sizeof(float));
}

ActivationFunction ActivationFunctionCreateIdentity(int inputSize) {
    return ActivationFunctionCreateSimple(inputSize, activation_identity);
}



// ReLU



typedef struct {
    float *zeros;
    float a;
} ReLUImplementer;

void activation_relu(void *implementer, const float *input, float *output, int size){
    ReLUImplementer * impl = (ReLUImplementer *) implementer;
    VectorMax(input, impl->zeros, output, size);
    if (impl->a != 1.0){
        VectorMulS(output, impl->a, output, size);
    }
}

void ReLUImplementerDestroy(void * ptr){
    ReLUImplementer* impl = (ReLUImplementer *) ptr;
    free(impl->zeros);
    free(ptr);
}

ActivationFunction ActivationFunctionCreateReLU(int inputSize, float a) {
    ReLUImplementer* implementer = malloc(sizeof(ReLUImplementer));
    implementer->zeros = malloc(inputSize * sizeof(float));
    implementer->a = a;
    memset(implementer->zeros, 0, inputSize * sizeof(float));
    return ActivationFunctionCreate(inputSize, ReLUImplementerDestroy, implementer, activation_relu);
}



// HARD SIGMOID



typedef struct {
    float *zeros;
    float *ones;
} HardSigmoidImplementer;

void activation_hard_sigmoid(void *implementer, const float *input, float *output, int size){
    HardSigmoidImplementer* impl = (HardSigmoidImplementer *)implementer;
    VectorMulS(input, 0.2, output, size);
    VectorAddS(output, 0.5, output, size);
    VectorMin(output, impl->ones, output, size);
    VectorMax(output, impl->zeros, output, size);
}

void HardSigmoidImplementerDestroy(void * ptr){
    HardSigmoidImplementer* impl = (HardSigmoidImplementer *) ptr;
    free(impl->zeros);
    free(ptr);
}


ActivationFunction ActivationFunctionCreateHardSigmoid(int inputSize){
    HardSigmoidImplementer* implementer = malloc(sizeof(HardSigmoidImplementer));
    implementer->zeros = malloc(2 * inputSize * sizeof(float));
    implementer->ones = implementer->zeros + inputSize;
    memset(implementer->zeros, 0, 2 * inputSize * sizeof(float));
    VectorAddS(implementer->ones, 1.0f, implementer->ones, inputSize);
    return ActivationFunctionCreate(inputSize, HardSigmoidImplementerDestroy, implementer, activation_hard_sigmoid);
}


// SOFTMAX

typedef struct {
    int vectorSize;
} SoftmaxImplementer;




void softmax(const float *input, float *output, int vectorSize){
    VectorExp(input, output, vectorSize);
    float sum = 0.0f;
    VectorSum(output, &sum, vectorSize);
    VectorDivS(output, sum, output, vectorSize);
}

void activation_softmax(void *implementer, const float *input, float *output, int size){
    SoftmaxImplementer* impl = (SoftmaxImplementer *)implementer;
    if (size == 1){
        softmax(input, output, impl->vectorSize);
        return;
    }
    dispatch_apply(size, DISPATCH_APPLY_AUTO, ^(size_t index) {
        size_t offset = index * impl->vectorSize;
        softmax(input + offset, output + offset, impl->vectorSize);
    });
}


void SoftmaxImplementerDestroy(void *ptr){
    free(ptr);
}

ActivationFunction ActivationFunctionCreateSoftmax(int inputSize, int vectorSize){
    SoftmaxImplementer *implementer = malloc(sizeof(SoftmaxImplementer));
    implementer->vectorSize = vectorSize;
    return ActivationFunctionCreate(inputSize, SoftmaxImplementerDestroy, implementer, activation_softmax);
}







