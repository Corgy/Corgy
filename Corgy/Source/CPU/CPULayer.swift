//
//  CPULayer.swift
//  Corgy
//
//  Created by buqian zheng on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public enum CPU {
    
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static let ReLU = { (_ input: Variable) -> Variable in
        for i in 0..<input.value.count {
            if input.value[i] < 0 {
                input.value[i] = 0
            }
        }
        return input
    }
    
    public static let flatten = { (_ input: Variable) -> Variable in
        var size = 1
        for num in input.getShape() {
            size = size * num
        }
        
        input.setShape([1, size])
        return input
    }
    
    /// return a Pooling layer
    /// - parameter strideStep: default stride is poolSize
    /// - parameter dilation: not supported yet
    /// - parameter padding: not supported yet
    public static func Pool(poolSize: (Int, Int),
                            strideStep: (Int, Int) = (-1,-1),
                            poolType: PoolType,
                            dilation: (Int, Int) = (1,1),
                            padding: (Int, Int) = (0,0)
        ) -> Layer {
        return { (_ input) in
            let varShape = input.getShape()
            let batchSize = varShape[0]
            let channels = varShape[1]
            let height = varShape[2]
            let width = varShape[3]
            
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
    
    /// return an inplace Dropout layer, it will modify
    /// and return the input
    /// - parameter p: percentage to be dropped out
    /// - parameter seed: optional, random seed
    public static func Dropout(_ p: Double, seed: Int?) -> Layer {
        return { (_ input) in
            assert(p >= 0 && p <= 1)
            srand48(seed ?? Int(arc4random()))
            for i in 0..<input.value.count {
                if input.value[i] < 0 {
                    input.value[i] = drand48() > p ? input.value[i] : 0
                }
            }
            return input
        }
    }
    
    /// return an Full Connected layer
    /// - parameter weight: dimension: num output class * input dimension
    /// - parameter bias: dimension: num output class * 1
    public static func FullConnected(weight: Variable, bias: Variable?) -> Layer {
        return { (_ input) in
            let weightShape = weight.getShape()
            assert(weightShape.count == 2 && weightShape[1] == input.value.count)
            
            if bias != nil {
                assert(bias?.getShape().count == 1 && bias?.getShape()[0] == weight.getShape()[0])
            }
            let out = Variable(weight.getShape()[0])
            for i in 0..<weight.getShape()[0] {
                out[i] = bias?[i] ?? 0
                for j in 0..<weight.getShape()[1] {
                    out[i] += input.value[j] * weight[i, j]
                }
            }
            return out
        }
    }
}

