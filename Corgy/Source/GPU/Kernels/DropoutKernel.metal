//
//  DropoutKernel.metal
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

#include <metal_stdlib>
using namespace metal;
#include "KernelParamType.h"
#include "Utils.h"

kernel void Dropout(device float *input [[ buffer(0) ]],
                    constant DropoutParam *param [[ buffer(1) ]],
                    uint id [[thread_position_in_grid]]
                    ) {
    float r = rand(param->seed, id, ((int)(size_t)param) + 832910);
    if (r < param->p) input[id] = 0;
}
