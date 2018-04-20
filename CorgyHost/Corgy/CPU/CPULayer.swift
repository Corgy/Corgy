//
//  CPULayer.swift
//  CPU
//
//  Created by buqian zheng on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public struct CPU {
  private init() {}
  
  /// an inplace ReLU layer, it will modify
  /// and return the input
  public static let relu = { (_ input: Variable) -> Variable in
    for i in 0..<input.value.count {
      if input.value[i] < 0 {
        input.value[i] = 0
      }
    }
    return input
  }
  
  /// pooling layer
  /// - parameter strideStep: default stride is poolSize
  /// - parameter dilation: not supported yet
  /// - parameter padding: not supported yet
  public static func pool2dGen(poolSize: (Int, Int),
                               strideStep: (Int, Int) = (-1,-1),
                               poolType: PoolType,
                               dilation: (Int, Int) = (1,1),
                               padding: (Int, Int) = (0,0)
    ) -> Layer {
    return { (_ input: Variable) -> Variable in
      let batchSize = input.shape[0]
      let channels = input.shape[1]
      let height = input.shape[2]
      let width = input.shape[3]
      
      let poolW = poolSize.0
      let poolH = poolSize.1
      
      let strideW = strideStep.0 > 0 ? strideStep.0 : poolW
      let strideH = strideStep.1 > 0 ? strideStep.1 : poolH
      
      let outH = Int((1 + Float(height - poolH) / Float(strideH)).rounded(.up))
      let outW = Int((1 + Float(width  - poolW) / Float(strideW)).rounded(.up))
      
      let output = Variable(batchSize, channels, outH, outW)
      
      for batch in 0..<batchSize {
        for channel in 0..<channels {
          for row in stride(from: 0, to: height, by: strideH) {
            for col in stride(from: 0, to: width, by: strideW) {
              var result: Variable.DataType
              switch poolType {
              case .Average:
                result = 0
                var numElements = Float(0)
                for y in 0..<poolH {
                  for x in 0..<poolW {
                    if y + row < height && x + col < width {
                      result += input[batch, channel, row+y, col+x]
                      numElements += 1
                    }
                  }
                }
                result /= numElements
              case .Max:
                result = -Float.infinity
                for y in 0..<poolH {
                  for x in 0..<poolW {
                    if y + row < height && x + col < width {
                      result = max(result, input[batch, channel, row+y, col+x])
                    }
                  }
                }
              }
              output[batch, channel, row/strideH, col/strideW] = result
            }
          }
        }
      }
      return output
    }
  }
}

extension CPU {
  
}
