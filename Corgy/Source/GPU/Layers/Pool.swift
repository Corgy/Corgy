//
//  Pool.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal
import QuartzCore

public enum PoolType: String {
    case Average = "AVG"
    case Max     = "MAX"
}

public extension Corgy {
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
            let t1 = CACurrentMediaTime()
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
            
            let poolParam = PoolParam(inputParam: input.param,
                                      outputParam: output.param,
                                      poolSizeX: poolW,
                                      poolSizeY: poolH)
            
            let paramBuffer = makeBuffer(poolParam)
            
            submitWork(name: "Pool\(poolType.rawValue)", in: input, output, param: param, parameterBuffer: paramBuffer)

            let t2 = CACurrentMediaTime()
            print("Pool \((t2 - t1) * 1000.0)")
            return output
        }
    }
}
