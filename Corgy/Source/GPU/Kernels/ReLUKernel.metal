//
//  ReLUKernel.metal
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

#include <metal_stdlib>
using namespace metal;
#include "KernelParamType.h"

kernel void ReLU(device float *input [[ buffer(0) ]],
                 uint id [[thread_position_in_grid]]
                 ) {
    if (input[id] < 0) input[id] = 0;
}
