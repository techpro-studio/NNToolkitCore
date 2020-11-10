//
//  activation_default_functions.c
//  audio_test
//
//  Created by Alex on 28.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "activation_default.h"
#include "ops.h"
#include "string.h"
#include "loop.h"
#include "stdlib.h"



ActivationFunction ActivationFunctionCreateSimple(int size, ActivationFunctionImpl function, ActivationFunctionImpl derivative) {
    return ActivationFunctionCreate(size, NULL, NULL, function, derivative, NULL);
}

ActivationFunction ActivationFunctionCreateSimpleWithCached(int size, ActivationFunctionImpl function, ActivationFunctionImpl derivative, ActivationFunctionImpl cachedDerivative) {
    return ActivationFunctionCreate(size, NULL, NULL, function, derivative, cachedDerivative);
}

//SIGMOID

// 1 / 1 - exp(-x)
void activation_sigmoid(void *implementer, const float *input, float *output, int size) {
    op_vec_neg(input, output, size);
    op_vec_exp(output, output, size);
    op_vec_add_sc(output, 1, output, size);
    op_vec_reciprocal(output, output, size);
}


// sigmoid(x)' = sigmoid(x)* (1 * sigmoid(x));

void activation_sigmoid_cached_derivative(void *implementer, const float *input, float *output, int size) {
    op_vec_neg(input, output, size);
    op_vec_add_sc(output, 1, output, size);
    op_vec_mul(input, output, output, size);
}


void activation_sigmoid_derivative(void *implementer, const float *input, float *output, int size) {
    activation_sigmoid(implementer, input, output, size);
    activation_sigmoid_cached_derivative(implementer, output, output, size);
}



ActivationFunction ActivationFunctionCreateSigmoid(int inputSize){
    return ActivationFunctionCreateSimpleWithCached(inputSize, activation_sigmoid, activation_sigmoid_derivative, activation_sigmoid_cached_derivative);
}



// TANH



void activation_tanh(void *implementer, const float *input, float *output, int size){
    op_vec_tanh(input, output, size);
}

void activation_tanh_cached_derivative(void *implementer, const float *input, float *output, int size) {
    op_vec_mul(input, input, output, size);
    op_vec_neg(output, output, size);
    op_vec_add_sc(output, 1, output, size);
}

void activation_tanh_derivative(void *implementer, const float *input, float *output, int size) {
    activation_tanh(implementer, input, output, size);
    activation_tanh_cached_derivative(implementer, output, output, size);
}


ActivationFunction ActivationFunctionCreateTanh(int inputSize){
    return ActivationFunctionCreateSimpleWithCached(inputSize, activation_tanh, activation_tanh_derivative, activation_tanh_cached_derivative);
}

//Identity


void activation_identity_derivative(void *implementer, const float *input, float *output, int size) {
    memcpy(output, input, size * sizeof(float));
}

void activation_identity(void *implementer, const float *input, float *output, int size){
    if (input == output){
        return;
    }
    memcpy(output, input, size * sizeof(float));
}

ActivationFunction ActivationFunctionCreateIdentity(int inputSize) {
    return ActivationFunctionCreateSimple(inputSize, activation_identity, activation_identity_derivative);
}


// ReLU



typedef struct {
    float a;
} ReLUImplementer;

void activation_relu_derivative(void *implementer, const float *input, float *output, int size) {
    op_vec_clamp(input, output, 0, 1, size);
}

void activation_relu(void *implementer, const float *input, float *output, int size){
    ReLUImplementer * impl = (ReLUImplementer *) implementer;
    op_vec_max_sc(input, 0, output, size);
    if (impl->a != 1.0){
        op_vec_mul_sc(output, impl->a, output, size);
    }
}

void ReLUImplementerDestroy(void * ptr){
    free(ptr);
}

ActivationFunction ActivationFunctionCreateReLU(int inputSize, float a) {
    ReLUImplementer* implementer = malloc(sizeof(ReLUImplementer));
//    implementer->max_fn = op_vec_max_sc_get_optimized(inputSize);
    implementer->a = a;
//    implementer->clamp_fn = op_vec_clamp_get_optimized(inputSize);
    return ActivationFunctionCreate(inputSize, ReLUImplementerDestroy, implementer, activation_relu, activation_relu_derivative, NULL);
}



// HARD SIGMOID



//typedef struct {
//    op_vec_clamp_fn clamp_fn;
//} HardSigmoidImplementer;
//
//void activation_hard_sigmoid(void *implementer, const float *input, float *output, int size){
//    HardSigmoidImplementer* impl = (HardSigmoidImplementer *)implementer;
//    op_vec_mul_sc(input, 0.2f, output, size);
//    op_vec_add_sc(output, 0.5f, output, size);
//    impl->clamp_fn(output, output, 0, 1, size);
//}
//
//void activation_hard_sigmoid_derivative(void *implementer, const float *input, float *output, int size) {
//#warning implement this;
//    memcpy(output, input, size * sizeof(float));
//}
//
//void HardSigmoidImplementerDestroy(void * ptr){
//    free(ptr);
//}
//
//ActivationFunction ActivationFunctionCreateHardSigmoid(int input_size){
//    HardSigmoidImplementer* implementer = malloc(sizeof(HardSigmoidImplementer));
//    implementer->clamp_fn = op_vec_clamp_get_optimized(input_size);
//    return ActivationFunctionCreate(input_size, HardSigmoidImplementerDestroy, implementer, activation_hard_sigmoid, activation_hard_sigmoid_derivative, NULL);
//}


// SOFTMAX


typedef struct {
    int vector_size;
} SoftmaxImplementer;


void softmax(const float *input, float *output, int vector_size){
    op_vec_exp(input, output, vector_size);
    float sum = 0.0f;
    op_vec_sum(output, &sum, vector_size);
    op_vec_div_sc(output, sum, output, vector_size);
}

void activation_softmax(void *implementer, const float *input, float *output, int size){
    SoftmaxImplementer* impl = (SoftmaxImplementer *)implementer;
    if (size == 1){
        softmax(input, output, impl->vector_size);
        return;
    }
    P_LOOP_START(size, index)
        size_t offset = index * impl->vector_size;
        softmax(input + offset, output + offset, impl->vector_size);
    P_LOOP_END
}

void activation_softmax_derivative(void *implementer, const float *input, float *output, int size) {
#warning implement this;
}


void SoftmaxImplementerDestroy(void *ptr){
    free(ptr);
}

ActivationFunction ActivationFunctionCreateSoftmax(int inputSize, int vectorSize){
    SoftmaxImplementer *implementer = malloc(sizeof(SoftmaxImplementer));
    implementer->vector_size = vectorSize;
    return ActivationFunctionCreate(inputSize, SoftmaxImplementerDestroy, implementer, activation_softmax, activation_softmax_derivative, NULL);
}







