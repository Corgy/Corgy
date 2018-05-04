//
//  MatrixToImage.metal
//  Corgy
//
//  Created by buqian zheng on 5/3/18.
//

#include <metal_stdlib>
using namespace metal;
#include "KernelParamType.h"

kernel void Mat2VarWithBias(const device float *input [[ buffer(0) ]],
                            // b is bias, I don't know why but Xcode highlights the word
                            // `bias`. Incase `bias` is a reserved word, I use name b here
                            // to represent the bias.
                            const device float *b [[ buffer(1) ]],
                            device float *output [[ buffer(2) ]],
                            constant MatToVariableWithBias *param [[ buffer(3) ]],
                            uint id [[thread_position_in_grid]]
                            ) {
    int outputWidth = param->outputParam.width;
    int c = id / param->outputParam.sizePerChannel;
    int h = (id % param->outputParam.sizePerChannel) / outputWidth;
    int w = id % outputWidth;
    output[id] = input[(w + h * outputWidth) * param->inputParam.width + c] + b[c];
}

kernel void Mat2VarWithoutBias(const device float *input [[ buffer(0) ]],
                               device float *output [[ buffer(1) ]],
                               constant MatToVariableWithoutBias *param [[ buffer(2) ]],
                               uint id [[thread_position_in_grid]]
                               ) {
    int outputWidth = param->outputParam.width;
    int c = id / param->outputParam.sizePerChannel;
    int h = (id % param->outputParam.sizePerChannel) / outputWidth;
    int w = id % outputWidth;
    output[id] = input[(w + h * outputWidth) * param->inputParam.width + c];
}
