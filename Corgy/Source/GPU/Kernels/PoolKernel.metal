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
    int poolW = param->poolSizeW;
    int poolH = param->poolSizeH;
    int strideW = param->strideW;
    int strideH = param->strideH;
    int padW = param->padW;
    int padH = param->padH;
    
    // output size per batch
    int ospb = param->outputParam.sizePerBatch;
    // output size per channel
    int ospc = param->outputParam.sizePerChannel;
    int ow = param->outputParam.width;
    
    int ispb = param->inputParam.sizePerBatch;
    int ispc = param->inputParam.sizePerChannel;
    int iw = param->inputParam.width;
    int ih = param->inputParam.height;
    
    int batchID = id / ospb;
    int channelID = (id % ospb) / ospc;
    int outputRow = (id % ospc) / ow;
    int outputCol = (id % ospc) % ow;
    
    int topLeftCol = outputCol * strideW - padW;
    int topLeftRow = outputRow * strideH - padH;
    
    float result = 0;
    int numEl = 0;
    for (int dy = 0; dy < poolH; dy++) {
        for (int dx = 0; dx < poolW; dx++) {
            int row = topLeftRow + dy;
            int col = topLeftCol + dx;
            if (row >= 0 && row < ih && col >= 0 && col < iw) {
                result += input[batchID * ispb + channelID * ispc + row * iw + col];
                numEl += 1;
            }
        }
    }
    output[id] = result / numEl;
    
    
//    // TODO: this is simple situation in a default size 2 pooling layer
//    int inputRow = 2 * outputRow;
//    int inputCol = 2 * outputCol;
//    int inputIndex = batchID * (param->inputParam.sizePerBatch) +
//    channelID * (param->inputParam.sizePerChannel) +
//    inputRow * (param->inputParam.width) + inputCol;
//    float m = input[inputIndex] + input[inputIndex+1] +
//    input[inputIndex+(param->inputParam.width)] + input[inputIndex+1+(param->inputParam.width)];
//    output[id] = m / 4;
}

kernel void PoolMAX(const device float *input [[ buffer(0) ]],
                    device float *output [[ buffer(1) ]],
                    constant PoolParam *param [[ buffer(2) ]],
                    uint id [[thread_position_in_grid]]
                    ) {
    int poolW = param->poolSizeW;
    int poolH = param->poolSizeH;
    int strideW = param->strideW;
    int strideH = param->strideH;
    int padW = param->padW;
    int padH = param->padH;
    
    // output size per batch
    int ospb = param->outputParam.sizePerBatch;
    // output size per channel
    int ospc = param->outputParam.sizePerChannel;
    int ow = param->outputParam.width;
    
    int ispb = param->inputParam.sizePerBatch;
    int ispc = param->inputParam.sizePerChannel;
    int iw = param->inputParam.width;
    int ih = param->inputParam.height;
    
    int batchID = id / ospb;
    int channelID = (id % ospb) / ospc;
    int outputRow = (id % ospc) / ow;
    int outputCol = (id % ospc) % ow;
    
    int topLeftCol = outputCol * strideW - padW;
    int topLeftRow = outputRow * strideH - padH;
    
    float result = -100000000;
    for (int dy = 0; dy < poolH; dy++) {
        for (int dx = 0; dx < poolW; dx++) {
            int row = topLeftRow + dy;
            int col = topLeftCol + dx;
            if (row >= 0 && row < ih && col >= 0 && col < iw) {
                result = max(result, input[batchID * ispb + channelID * ispc + row * iw + col]);
            }
        }
    }
    output[id] = result;
}
