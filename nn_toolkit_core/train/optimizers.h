//
// Created by Alex on 02.11.2020.
//

#ifndef optimizers_h
#define optimizers_h



void sum_batch_gradient(float * gradients, float* gradient, int size, int batch);

typedef struct {
    float learning_rate;
} SGD;

int sgd_optimize(SGD optimizer, float *gradient, float *weights, int size);


#endif //optimizers_h