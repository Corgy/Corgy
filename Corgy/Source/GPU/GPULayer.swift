//
//  GPULayer.swift
//  Corgy
//
//  Created by buqian zheng on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
import Metal

let THREAD_PER_GROUP = 128

/// main struct for Corgy GPU framework
public enum Corgy {
    
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static func ReLU() -> Layer {
        return {(_ input) in
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.value.count), 1, 1)
            let threadGroups = MTLSizeMake((input.value.count + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            submitWork(name: "ReLU", in: input, param: param)
            return input
        }
    }

    /// return a Pooling layer
    /// - parameter strideStep: default stride is poolSize
    /// - parameter dilation: not supported yet
    /// - parameter padding: not supported yet
    /// - require: results of input.width / poolSize.x and
    ///            input.height / poolSize.y must be integer
    public static func Pool(poolSize: (Int, Int),
                            strideStep: (Int, Int) = (-1,-1),
                            poolType: PoolType,
                            dilation: (Int, Int) = (1,1),
                            padding: (Int, Int) = (0,0)
        ) -> Layer {
        return { (_ input) in
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
            
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.value.count), 1, 1)
            let threadGroups = MTLSizeMake((output.value.count + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            let poolParam = PoolParam(inputParam: input.getParam(),
                                      outputParam: output.getParam(),
                                      poolSizeX: poolW,
                                      poolSizeY: poolH)
            
            let paramBuffer = makeBuffer(param: poolParam, length: MemoryLayout<PoolParam>.stride)
            
            submitWork(name: "Pool\(poolType.rawValue)", in: input, output, param: param, parameterBuffer: paramBuffer)
            
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
    static func getGroupNum(_ eleNum: Int, _ eleNumPerGroup: Int) -> Int {
        return  (eleNum + eleNumPerGroup - 1) / eleNumPerGroup
    }
    
    public static func Neg() -> Layer {
        return { ( _ input) in
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.value.count), 1, 1)
            let threadGroups = MTLSizeMake(getGroupNum(input.value.count, THREAD_PER_GROUP), 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            submitWork(name: "testNeg", in: input, param: param)
            return input
        }
    }
    
    
    public static func Neg2() -> Layer {
        return { ( _ input) in
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.value.count), 1, 1)
            let threadGroups = MTLSizeMake(getGroupNum(input.value.count, THREAD_PER_GROUP), 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            let output = Variable(input.getShape())
            submitWork(name: "testNeg2", in: input, output, param: param)
            return output
        }
    }
}




