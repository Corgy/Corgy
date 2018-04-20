//
//  Conv2D.swift
//  CLI
//
//  Created by HOPE on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

// TODO: support dilation, droup and bias
func Conv2D(inChannels: Int,
            outChannels: Int,
            kernelSize: Int,
            stride: Int = 1,
            padding: Int = 0,
            dilation: Int = 1,
            groups: Int = 1,
            bias: Bool = true
    ) -> (Variable) -> Variable {
    let kernelWeights = Variable(outChannels, inChannels, kernelSize * kernelSize)
    let bias = Variable(outChannels)
    
    for i in 0..<kernelWeights.value.count {
        kernelWeights.value[i] = Float(i)
    }
    print(kernelWeights.value)
    
    for i in 0..<bias.value.count {
        bias.value[i] = Float(i) * 0.01
    }
    print(bias.value)
    
    let ret = {
        (_ input: Variable)->Variable in
        // input is inChannels * width * height
        // output is ouputChannels * (width - (kernelSize - 1) * 2) * (height - (kernelSize - 1) * 2)
        let dimNum = input.shape.count
        
        if dimNum == 0 {
            return Variable(0)
        }
        
        let N = input.shape[0]
        let inputHeight = input.shape[2]
        let inputWidth = input.shape[3]
        let outputWidth = inputWidth - 2 * (kernelSize - 1)
        let outputHeight = inputHeight - 2 * (kernelSize - 1)
        let output = Variable(N, outChannels, outputHeight, outputWidth)
        
        for i in 0..<N {
            for kernelTopLeftRow in 0..<inputHeight - kernelSize {
                for kernelTopLeftCol in 0..<inputWidth - kernelSize {
                    let kernelButtomRightRow = kernelTopLeftRow + kernelSize
                    let kernelButtomRightCol = kernelTopLeftCol + kernelSize
                    
                    let slicedImages = Variable(inChannels, kernelSize, kernelSize)
                    
                    for ii in 0..<inChannels {
                        for jj in kernelTopLeftRow..<kernelButtomRightRow {
                            for kk in kernelTopLeftCol..<kernelButtomRightCol {
                                slicedImages[ii, jj - kernelTopLeftRow, kk - kernelTopLeftCol] =
                                    input[i, ii, jj, kk]
                            }
                        }
                    }
                    
                    for outChannelIndex in 0..<outChannels {
                        
                        let kernels = Variable(inChannels, kernelSize, kernelSize)
                        for ii in 0..<inChannels {
                            for jj in 0..<kernelSize {
                                for kk in 0..<kernelSize {
                                    kernels[ii, jj, kk] =
                                        kernelWeights[outChannelIndex, ii, jj, kk]
                                }
                            }
                        }
                        
                        var sum = Float(0.0)
                        for inChannelIndex in 0..<inChannels {
                            let kernel = Variable(kernelSize, kernelSize)
                            let image  = Variable(kernelSize, kernelSize)
                            
                            for ii in 0..<kernelSize {
                                for jj in 0..<kernelSize {
                                    kernel[ii,jj] = kernels[inChannelIndex, ii, jj]
                                    image[ii, jj] = slicedImages[inChannelIndex, ii, jj]
                                }
                            }
                            
                            for ii in 0..<kernelSize {
                                for jj in 0..<kernelSize {
                                    sum += kernel[ii, jj] * image[ii, jj]
                                }
                            }
                        }
                        
                        sum += bias[outChannelIndex]
                        output[i, outChannelIndex, kernelTopLeftRow, kernelButtomRightCol] = sum
                    }
                }
            }
        }
        return output
    }
    return ret
}
