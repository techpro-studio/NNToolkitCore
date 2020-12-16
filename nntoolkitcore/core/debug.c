//
//  degub.c
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "debug.h"
#include "stdlib.h"

void print_vector(const float *vector, int size)
{
    for(int i = 0; i < size; ++i)
        printf("%e ", vector[i]);
}

void print_matrix(const float *matrix, int rows, int columns) {
    printf("\n start matrix");
    for (int r = 0; r < rows; ++r){
        printf("\n row %d ", r);
        print_vector(matrix + r * columns, columns);
    }
    printf("\n end matrix");
}

void getTensorIndexOfFlatten(const int*shape, int shapeSize, int flattenIndex, int*indices){
    int remaining = flattenIndex;
    for (int i = 0; i < shapeSize; ++i){
        if (i == shapeSize - 1){
            indices[i] = remaining;
            return;
        }
        int divider = 0;
        for (int j = i + 1; j < shapeSize; ++j){
            divider += shape[j];
        }
        indices[i] = remaining / divider;
        remaining -= indices[i] * divider;
    }
}

void print_tensor(const float *tensor, int*shape, int shape_size) {
    if (shape_size == 0){
        return;
    }
    if (shape_size == 1){
        print_vector(tensor, *shape);
        return;
    }
    if (shape_size == 2){
        print_matrix(tensor, shape[0], shape[1]);
        return;
    }
    int sum = 1;
    for (int i = 0; i < shape_size - 1; ++i)
        sum *= shape[i];
    int vectorSize = shape[shape_size - 1];
    for (int i = 0; i < sum; ++i)
    {
        int indices[shape_size - 1];
        getTensorIndexOfFlatten(shape, shape_size - 1, i, indices);
        printf("\n Index: ");
        for (int j = 0; j < shape_size - 1; ++j)
            printf("%d ", indices[j]);
        print_vector(tensor + vectorSize * i, vectorSize);
    }
}
