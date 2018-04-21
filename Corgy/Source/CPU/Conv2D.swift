//
//  Conv2D.swift
//  CLI
//
//  Created by HOPE on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
extension CPU {
  // TODO: support dilation, group and bias
  public static func Conv2D(inChannels: Int,
                            outChannels: Int,
                            kernelSize: Int,
                            stride: Int = 1,
                            padding: Int = 0,
                            dilation: Int = 0,
                            groups: Int = 1,
                            weight: Variable,
                            bias: Variable = Variable(0)
    ) -> (Variable) -> Variable {
    assert(weight.shape.count == 4 && weight.shape[2] == weight.shape[3] && weight.shape[2] == kernelSize)
    let kernelWeights = weight
    
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
      let outputWidth = inputWidth - kernelSize + 1
      let outputHeight = inputHeight - kernelSize + 1
      let output = Variable(N, outChannels, outputHeight, outputWidth)
      
      for i in 0..<N {
        for kernelTopLeftRow in 0..<outputHeight {
          for kernelTopLeftCol in 0..<outputWidth {
            let kernelButtomRightRow = kernelTopLeftRow + kernelSize
            let kernelButtomRightCol = kernelTopLeftCol + kernelSize
//            print(kernelTopLeftRow, kernelTopLeftCol)
//            print(kernelButtomRightRow, kernelButtomRightCol)
            let slicedImages = Variable(inChannels, kernelSize, kernelSize)
            
            for ii in 0..<inChannels {
              for jj in kernelTopLeftRow..<kernelButtomRightRow {
                for kk in kernelTopLeftCol..<kernelButtomRightCol {
                  slicedImages[ii, jj - kernelTopLeftRow, kk - kernelTopLeftCol] =
                    input[i, ii, jj, kk]
                }
              }
            }
            
            print("SlicedImages", slicedImages)
            for outChannelIndex in 0..<outChannels {
              let kernels = Variable(inChannels, kernelSize, kernelSize)
              for ii in 0..<inChannels {
                for jj in 0..<kernelSize {
                  for kk in 0..<kernelSize {
                    kernels[ii, jj, kk] =
                      kernelWeights[ii, outChannelIndex, jj, kk]
                  }
                }
              }
              print(kernels)
              var sum = Float(0.0)
              for inChannelIndex in 0..<inChannels {
                for ii in 0..<kernelSize {
                  for jj in 0..<kernelSize {
                    let n1 = kernels[inChannelIndex, ii, jj]
                    let n2 = slicedImages[inChannelIndex, ii, jj]
                    sum += n1 * n2
                  }
                }
              }
              
              if bias.value.count > 0 {
                sum += bias[outChannelIndex]
              }
              
              output[i, outChannelIndex, kernelTopLeftRow, kernelTopLeftCol] = sum
              print(i, outChannelIndex, kernelTopLeftRow, kernelTopLeftCol, sum)
            }
          }
        }
      }
      return output
    }
    
    return ret
  }
}