//
//  ImageToMatrix.metal
//  Corgy
//
//  Created by buqian zheng on 5/3/18.
//

#include <metal_stdlib>
using namespace metal;
#include "KernelParamType.h"

kernel void ImageToMatrix(const device float *input [[ buffer(0) ]],
                          device float *output [[ buffer(1) ]],
                          constant ImageToMatParam *param [[ buffer(2) ]],
                          uint id [[thread_position_in_grid]]
                          ) {
    int i = id / param->outputParam.width;
    int j = id % param->outputParam.width;
    
    int channel = j / param->kernelSizeSquared;
    int num = j % param->kernelSizeSquared;
    int row = i / param->kernelPerRow + num / param->kernelSize;
    int col = i % param->kernelPerRow + num % param->kernelSize;
    
    output[i * param->outputParam.width + j] = input[channel * param->inputParam.sizePerChannel + row * param->inputParam.width + col];
}

kernel void WeightToMatrix(const device float *input [[ buffer(0) ]],
                           device float *output [[ buffer(1) ]],
                           constant WeightToMatParam *param [[ buffer(2) ]],
                           uint id [[thread_position_in_grid]]) {
    int kernelSize = param->inputParam.width;
    int kernelSizeSquared = kernelSize * kernelSize;
    int i = id / param->outputParam.width;
    int j = id % param->outputParam.width;
    int h = i % kernelSizeSquared / kernelSize;
    int w = i % kernelSizeSquared % kernelSize;
    output[i * param->outputParam.width + j] =
        input[j * param->inputParam.sizePerBatch + i / kernelSizeSquared * param->inputParam.sizePerChannel + h * kernelSize + w];
}
