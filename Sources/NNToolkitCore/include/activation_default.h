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

ActivationFunction* ActivationFunctionCreateIdentity(int inputSize);

ActivationFunction * ActivationFunctionCreateSigmoid(int inputSize);

ActivationFunction * ActivationFunctionCreateReLU(int inputSize, float a);

ActivationFunction * ActivationFunctionCreateTanh(int inputSize);

ActivationFunction * ActivationFunctionCreateHardSigmoid(int inputSize);

#if defined __cplusplus
}
#endif

#endif /* activation_default_functions_h */
