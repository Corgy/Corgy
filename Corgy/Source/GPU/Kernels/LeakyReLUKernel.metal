//
//  LeakyReLU.metal
//  Corgy
//
//  Created by HOPE on 5/2/18.
//

#include <metal_stdlib>
using namespace metal;

#include "KernelParamType.h"

kernel void LeakyReLU(device float *input [[ buffer(0) ]],
                      constant LeakyReLUParam *param [[ buffer(1) ]],
                      uint id [[thread_position_in_grid]]
                 ) {
    float negativeScope = param->negativeScope;
    
    if (input[id] < 0) input[id] = negativeScope * input[id];
}
