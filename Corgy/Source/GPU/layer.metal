//
//  layer.metal
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

#include <metal_stdlib>
using namespace metal;

struct VariableParam {
    int size;
    int batch;
    int sizePerBatch;
    int channel;
    int sizePerChannel;
    int width;
    int height;
};

struct PoolParam {
    VariableParam inputParam;
    VariableParam outputParam;
    int poolSizeX;
    int poolSizeY;
};

kernel void ReLU(device float *input [[ buffer(0) ]],
                 uint id [[thread_position_in_grid]]
                 ) {
    if (input[id] < 0) input[id] = 0;
}

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
    int m = max(input[inputIndex], input[inputIndex+1]);
    int n = max(input[inputIndex+(param->inputParam.width)], input[inputIndex+1+(param->inputParam.width)]);
    output[id] = max(m, n);
}

// Example of a kernel that reads an input and writes to an output
kernel void testNeg2(const device float *input [[ buffer(0) ]],
                     device float *output [[ buffer(1) ]],
                     uint2 gsize [[threadgroups_per_grid]],       // number of thread groups
                     uint2 gid [[threadgroup_position_in_grid]], // coordinate of current thread group
                     uint2 tsize [[threads_per_threadgroup]],     // number of thread in one thread group
                     uint2 tid [[thread_position_in_threadgroup]]// coordinate of current thread
                     ) {
    uint id = (tsize.x * tsize.y) * (gsize.x * gid.y + gid.x) + tsize.x * tid.y + tid.x;
    output[id] = -input[id];
}

// Example of an inplace kernel
kernel void testNeg(device float *input [[ buffer(0) ]],
                    uint2 gsize [[threadgroups_per_grid]],       // number of thread groups
                    uint2 gid [[threadgroup_position_in_grid]], // coordinate of current thread group
                    uint2 tsize [[threads_per_threadgroup]],     // number of thread in one thread group
                    uint2 tid [[thread_position_in_threadgroup]]// coordinate of current thread
                    ) {
    uint id = (tsize.x * tsize.y) * (gsize.x * gid.y + gid.x) + tsize.x * tid.y + tid.x;
    input[id] = -input[id];
}
