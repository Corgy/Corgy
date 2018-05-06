//
//  Conv2D.swift
//  Corgy
//
//  Created by HOPE on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
import QuartzCore

extension CPU {
  // TODO: support dilation, group and bias
  public static func Conv2D(inChannels: Int,
                            outChannels: Int,
                            kernelSize: Int,
                            stride: Int = 1,
                            padding: Int = 0,
                            dilation: Int = 1,
                            groups: Int = 1,
                            weight: Variable,
                            bias: Variable = Variable(0)
    ) -> Layer {
    var weightShape = weight.shape
    assert(weightShape.count == 4 && weightShape[2] == weightShape[3] && weightShape[2] == kernelSize)
    let kernelWeights = weight
    
    return { (_ input) in
      // input is inChannels * width * height
      // output is ouputChannels * (width - (kernelSize - 1) * 2) * (height - (kernelSize - 1) * 2)
      var input = input
      input.trimDimension(atMost: 1)
      input = input.padding(paddingWith: padding)
      var inputShape = input.shape
      inputShape.insert(1, at:0)
      input.shape = inputShape
        
      let dimNum = input.shape.count
        
      if dimNum == 0 {
        return Variable(0)
      }
      
      let N = inputShape[0]
      let inputHeight = inputShape[2]
      let inputWidth = inputShape[3]
      let outputWidth = inputWidth - kernelSize + 1
      let outputHeight = inputHeight - kernelSize + 1
      let output = Variable(N, outChannels, outputHeight, outputWidth)
//      timing("CPU conv: ") {
          for i in 0..<N {
            for kernelTopLeftRow in 0..<outputHeight {
              for kernelTopLeftCol in 0..<outputWidth {
                let kernelButtomRightRow = kernelTopLeftRow + kernelSize
                let kernelButtomRightCol = kernelTopLeftCol + kernelSize
                let slicedImages = Variable(inChannels, kernelSize, kernelSize)
                
                for ii in 0..<inChannels {
                  for jj in kernelTopLeftRow..<kernelButtomRightRow {
                    for kk in kernelTopLeftCol..<kernelButtomRightCol {
                        let index1 = ii * slicedImages.indexAuxilary[0]
                            + (jj - kernelTopLeftRow) * slicedImages.indexAuxilary[1]
                            + (kk - kernelTopLeftCol) * slicedImages.indexAuxilary[2]
                        
                        let index2 = i * input.indexAuxilary[0]
                            + ii * input.indexAuxilary[1]
                            + jj * input.indexAuxilary[2]
                            + kk * input.indexAuxilary[3]
                        
                        slicedImages.value[index1] = input.value[index2]
                    }
                  }
                }
                
                for outChannelIndex in 0..<outChannels {
                  let kernels = Variable(inChannels, kernelSize, kernelSize)
                  for ii in 0..<inChannels {
                    for jj in 0..<kernelSize {
                      for kk in 0..<kernelSize {
                        let index1 = ii * kernels.indexAuxilary[0]
                        + jj * kernels.indexAuxilary[1]
                        + kk * kernels.indexAuxilary[2]
                        
                        let index2 = outChannelIndex * kernelWeights.indexAuxilary[0]
                            + ii * kernelWeights.indexAuxilary[1]
                            + jj * kernelWeights.indexAuxilary[2]
                            + kk * kernelWeights.indexAuxilary[3]
                        
                        kernels.value[index1] = kernelWeights.value[index2]
                      }
                    }
                  }
                    
                  var sum = Float(0.0)
                  for inChannelIndex in 0..<inChannels {
                    for ii in 0..<kernelSize {
                      for jj in 0..<kernelSize {
                        let index1  = inChannelIndex * kernels.indexAuxilary[0]
                                    + ii * kernels.indexAuxilary[1]
                                    + jj * kernels.indexAuxilary[2]
                        
                        let index2  = inChannelIndex * slicedImages.indexAuxilary[0]
                            + ii * slicedImages.indexAuxilary[1]
                            + jj * slicedImages.indexAuxilary[2]
                        
                        let n1 = kernels.value[index1]
                        let n2 = slicedImages.value[index2]
                        sum += n1 * n2
                      }
                    }
                  }
                  
                  if bias.count > 0 {
                    sum += bias[outChannelIndex]
                  }
                  let index = i * output.indexAuxilary[0]
                             + outChannelIndex * output.indexAuxilary[1]
                             + kernelTopLeftRow * output.indexAuxilary[2]
                             + kernelTopLeftCol * output.indexAuxilary[3]
                    
                  output.value[index] = sum
                }
              }
            }
          }
//      }
      return output
    }
  }
}
