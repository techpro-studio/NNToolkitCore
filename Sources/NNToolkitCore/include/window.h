//
//  window.h
//  
//
//  Created by Alex on 04.10.2020.
//

#ifndef window_h
#define window_h

#include <stdio.h>

typedef void(*window_fn) (float *, int);

void hamming_window(float *vector, int size);

void hann_window(float *vector, int size);

void blackman_window(float *vector, int size);

#endif /* File_h */
