//
//  degub.h
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef degub_h
#define degub_h

#include <stdio.h>
#if defined __cplusplus
extern "C" {
#endif

void printVector(const float *vector, int size);

void printMatrix(const float *matrix, int rows, int columns);

void printTensor(const float *tensor, int *shape, int shapeSize);

#if defined __cplusplus
}
#endif

#endif /* degub_h */
