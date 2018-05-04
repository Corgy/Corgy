//
//  CPULayer.swift
//  Corgy
//
//  Created by buqian zheng on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
import QuartzCore
public enum CPU {
    
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static let ReLU = { (_ input: Variable) -> Variable in
        for i in 0..<input.size {
            if input.value[i] < 0 {
                input.value[i] = 0
            }
        }
        return input
    }
    
    public static func LeakyReLU(negativeScope: Float = 0.01) -> Layer {
        return {(_ input) in
            for i in 0..<input.size {
                if input.value[i] < 0 {
                    input.value[i] = negativeScope * input.value[i]
                }
            }
            return input
        }
    }
    
    public static let flatten = { (_ input: Variable) -> Variable in
        var size = 1
        for num in input.shape {
            size = size * num
        }
        
        input.shape = [1, size]
        return input
    }
    
    /// return a Pooling layer
    /// - parameter strideStep: default stride is poolSize, implemented
    /// - parameter dilation: not supported yet
    /// - parameter padding: implemented
    public static func Pool(poolSize: (Int, Int),
                            stride strideStep: (Int, Int)? = nil,
                            poolType: PoolType,
                            dilation: (Int, Int) = (1, 1),
                            padding: (Int, Int) = (0, 0)
        ) -> Layer {
        return { (_ input) in
            let varShape = input.shape
            let batchSize = varShape[0]
            let channels = varShape[1]
            let height = varShape[2]
            let width = varShape[3]
            
            let (poolW, poolH) = poolSize
            
            let (strideW, strideH) = strideStep ?? (poolW, poolH)
            let (padW, padH) = padding
            
            let outH = Int((1 + Float(height - poolH) / Float(strideH)).rounded(.up))
            let outW = Int((1 + Float(width  - poolW) / Float(strideW)).rounded(.up))
            
            let output = Variable(batchSize, channels, outH, outW)
            
            for batch in 0..<batchSize {
                for channel in 0..<channels {
                    /**
                     * If padding is 1, then the coordinate system in
                     * top left: (-1, -1), bottom right: (inW+1, inH+1)
                     */
                    for row in stride(from: -padH, through: height+padH-poolH, by: strideH) {
                        for col in stride(from: -padW, through: width+padW-poolW, by: strideW) {
                            var result: Variable.DataType
                            switch poolType {
                            case .Average:
                                result = 0
                                var numElements = Float(0)
                                for y in 0..<poolH {
                                    for x in 0..<poolW {
                                        // currentRow = row + y
                                        // currentCol = col + x
                                        if y + row < height && y + row >= 0 &&
                                           x + col < width && x + col >= 0 {
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
                                        if y + row < height && y + row >= 0 &&
                                            x + col < width && x + col >= 0 {
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
            for i in 0..<input.size {
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
            let t1 = CACurrentMediaTime()
            let weightShape = weight.shape
            assert(weightShape.count == 2 && weightShape[1] == input.size)
            
            if bias != nil {
                assert(bias?.shape.count == 1 && bias?.shape[0] == weight.shape[0])
            }
            let out = Variable(weight.shape[0])
//            timing("Fully connected: ") {
                for i in 0..<weight.shape[0] {
                    out[i] = bias?[i] ?? 0
                    for j in 0..<weight.shape[1] {
                        out[i] += input.value[j] * weight[i, j]
                    }
                }
//            }
            let t2 = CACurrentMediaTime()
            print("CPU FC: \((t2 - t1) * 1000.0)")
            return out
        }
    }
}
