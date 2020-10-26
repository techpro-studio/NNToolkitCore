//
//  activation_default_functions.h
//  audio_test
//
//  Created by Alex on 28.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef activation_default_functions_h
#define activation_default_functions_h

#include <stdio.h>
#include "activation.h"

#if defined __cplusplus
extern "C" {
#endif

ActivationFunction ActivationFunctionCreateIdentity(int input_size);

ActivationFunction ActivationFunctionCreateSoftmax(int input_size, int vector_size);

ActivationFunction ActivationFunctionCreateSigmoid(int input_size);

ActivationFunction ActivationFunctionCreateReLU(int input_size, float a);

ActivationFunction ActivationFunctionCreateTanh(int input_size);


#if defined __cplusplus
}
#endif

#endif /* activation_default_functions_h */
