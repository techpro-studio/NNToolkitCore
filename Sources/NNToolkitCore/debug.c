//
//  degub.c
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "debug.h"
#include <CoreFoundation/CoreFoundation.h>

void printVector(const float *vector, int size)
{
    for(int i = 0; i < size; ++i)
        printf("%f ", vector[i]);
}

void printMatrix(const float *matrix, int rows, int columns) {
    printf("\n start matrix");
    for (int r = 0; r < rows; ++r){
        printf("\n row %d ", r);
        printVector(matrix + r * columns, columns);
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

void printTensor(const float *tensor, int*shape, int shapeSize) {
    int sum = 1;
    for (int i = 0; i < shapeSize - 1; ++i)
        sum *= shape[i];
    int vectorSize = shape[shapeSize - 1];
    for (int i = 0; i < sum; ++i)
    {
        int indices[shapeSize - 1];
        getTensorIndexOfFlatten(shape, shapeSize - 1, i, indices);
        printf("\n Index: ");
        for (int j = 0; j < shapeSize - 1; ++j)
            printf("%d ", indices[j]);
        printVector(tensor + vectorSize * i, vectorSize);
    }
}
