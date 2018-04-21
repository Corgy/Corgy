//
//  GPULayer.swift
//  Corgy
//
//  Created by buqian zheng on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
import Metal

/// main struct for Corgy GPU framework
public struct Corgy {
    
    public init() { }

    // TODO: solve reference cycle on closure and network
    
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static func ReLU(network: NeuralNetwork) -> Layer {
        return { (_ input: Variable) -> Variable in
            submitWork(network, name: "ReLU", data: input)
            return input
        }
    }
    
    /// return a Pooling layer
    /// - parameter strideStep: default stride is poolSize
    /// - parameter dilation: not supported yet
    /// - parameter padding: not supported yet
    public static func Pool2D(network: NeuralNetwork,
                              poolSize: (Int, Int),
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
            submitWork(network, name: "Pool2D", data: input, output, param: nil)
            return output
        }
    }
    
    /// return an inplace DropOut layer, it will modify
    /// and return the input
    /// - parameter p: percentage to be dropped out
    /// - parameter seed: optional, random seed
    public static func DropOut(network: NeuralNetwork, p: Double) -> Layer {
        return { (_ input: Variable) -> Variable in
            assert(p >= 0 && p <= 1)
            submitWork(network, name: "DropOut", data: input)
            return input
        }
    }
    
    /// return an Full Connected layer
    /// - parameter weight: dimension: num output class * input dimension
    /// - parameter bias: dimension: num output class * 1
    public static func FullConnected(network: NeuralNetwork, weight: Variable, bias: Variable?) -> Layer {
        return { (_ input: Variable) -> Variable in
            assert(weight.shape.count == 2 && weight.shape[1] == input.value.count)
            if bias != nil {
                assert(bias?.shape.count == 1 && bias?.shape[0] == weight.shape[0])
            }
            let output = Variable(weight.shape[0])
            submitWork(network, name: "FullConnected", data: input, output)
            return output
        }
    }
}

extension Corgy {
    
    
    
}

