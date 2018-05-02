//
//  PoolKernel.metal
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

#include <metal_stdlib>
using namespace metal;
#include "KernelParamType.h"

kernel void PoolAVG(const device float *input [[ buffer(0) ]],
                    device float *output [[ buffer(1) ]],
                    constant PoolParam *param [[ buffer(2) ]],
                    uint id [[thread_position_in_grid]]
                    ) {
    int batchID = id / param->outputParam.sizePerBatch;
    int channelID = (id % param->outputParam.sizePerBatch) / param->outputParam.sizePerChannel;
    int outputRow = (id % param->outputParam.sizePerChannel) / param->outputParam.width;
    int outputCol = (id % param->outputParam.sizePerChannel) % param->outputParam.width;
    // TODO: this is simple situation in a default size 2 pooling layer
    int inputRow = 2 * outputRow;
    int inputCol = 2 * outputCol;
    int inputIndex = batchID * (param->inputParam.sizePerBatch) +
    channelID * (param->inputParam.sizePerChannel) +
    inputRow * (param->inputParam.width) + inputCol;
    int m = input[inputIndex] + input[inputIndex+1] +
    input[inputIndex+(param->inputParam.width)] + input[inputIndex+1+(param->inputParam.width)];
    output[id] = float(m) / 4;
}

kernel void PoolMAX(const device float *input [[ buffer(0) ]],
                    device float *output [[ buffer(1) ]],
                    constant PoolParam *param [[ buffer(2) ]],
                    uint id [[thread_position_in_grid]]
                    ) {
    int batchID = id / param->outputParam.sizePerBatch;
    int channelID = (id % param->outputParam.sizePerBatch) / param->outputParam.sizePerChannel;
    int outputRow = (id % param->outputParam.sizePerChannel) / param->outputParam.width;
    int outputCol = (id % param->outputParam.sizePerChannel) % param->outputParam.width;
    // TODO: this is simple situation in a default size 2 pooling layer
    int inputRow = 2 * outputRow;
    int inputCol = 2 * outputCol;
    int inputIndex = batchID * (param->inputParam.sizePerBatch) +
    channelID * (param->inputParam.sizePerChannel) +
    inputRow * (param->inputParam.width) + inputCol;
    float m = max(input[inputIndex], input[inputIndex+1]);
    float n = max(input[inputIndex+(param->inputParam.width)], input[inputIndex+1+(param->inputParam.width)]);
    output[id] = max(m, n);    
}
