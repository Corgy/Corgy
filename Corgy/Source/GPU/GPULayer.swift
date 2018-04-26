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
    
    
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static func ReLU(network: NeuralNetwork) -> Layer {
        return { [unowned network] (_ input) in
            submitWork(network, name: "ReLU", in: input)
            return input
        }
    }
    public init() { }
    

    /// return a Pooling layer
    /// - parameter strideStep: default stride is poolSize
    /// - parameter dilation: not supported yet
    /// - parameter padding: not supported yet
    /// - require: results of input.width / poolSize.x and
    ///            input.height / poolSize.y must be integer
    public static func Pool2D(network: NeuralNetwork,
                              poolSize: (Int, Int),
                              strideStep: (Int, Int) = (-1,-1),
                              poolType: PoolType,
                              dilation: (Int, Int) = (1,1),
                              padding: (Int, Int) = (0,0)
        ) -> Layer {
        return { [unowned network] (_ input) in
            let batchSize = input.getShape()[0]
            let channels = input.getShape()[1]
            let height = input.getShape()[2]
            let width = input.getShape()[3]

            let poolW = poolSize.0
            let poolH = poolSize.1

            let strideW = strideStep.0 > 0 ? strideStep.0 : poolW
            let strideH = strideStep.1 > 0 ? strideStep.1 : poolH

            let outH = Int((1 + Float(height - poolH) / Float(strideH)).rounded(.up))
            let outW = Int((1 + Float(width  - poolW) / Float(strideW)).rounded(.up))

            let output = Variable(batchSize, channels, outH, outW)
            let name: String
            switch poolType {
            case .Average:  name = "Pool2DAVG"
            case .Max:      name = "Pool2DMAX"
            }
            submitWork(network, name: name, in: input, output)
            return output
        }
    }
/*
    /// return an inplace DropOut layer, it will modify
    /// and return the input
    /// - parameter p: percentage to be dropped out
    /// - parameter seed: optional, random seed
    public static func DropOut(network: NeuralNetwork, p: Double) -> Layer {
        return { [unowned network] (_ input) in
            assert(p >= 0 && p <= 1)
            submitWork(network, name: "DropOut", data: input)
            return input
        }
    }

    /// return an Full Connected layer
    /// - parameter weight: dimension: num output class * input dimension
    /// - parameter bias: dimension: num output class * 1
    public static func FullConnected(network: NeuralNetwork, weight: Variable, bias: Variable?) -> Layer {
        return { [unowned network] (_ input) in
            assert(weight.shape.count == 2 && weight.shape[1] == input.value.count)
            if bias != nil {
                assert(bias?.shape.count == 1 && bias?.shape[0] == weight.shape[0])
            }
            let output = Variable(weight.shape[0])
            submitWork(network, name: "FullConnected", data: input, output)
            return output
        }
    }
 */
}

public extension Corgy {
    public static func Neg(network: NeuralNetwork) -> Layer {
        return { [unowned network] ( _ input) in
            submitWork(network, name: "testNeg", in: input)
            return input
        }
    }
    public static func Neg2(network: NeuralNetwork) -> Layer {
        return { [unowned network] ( _ input) in
            let output = Variable(input.getShape())
            submitWork(network, name: "testNeg2", in: input, output)
            return output
        }
    }
}


