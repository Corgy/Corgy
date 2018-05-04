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
    /// TODO: add full support for stride, dilation and padding
    /// - parameter strideStep: default stride is poolSize
    /// - parameter dilation: not supported yet
    /// - parameter padding: (padH, padW)
    /// - require: results of input.width / poolSize.x and
    ///            input.height / poolSize.y must be integer
    public static func Pool(poolSize: (Int, Int),
                            stride strideStep: (Int, Int)? = nil,
                            poolType: PoolType,
                            dilation: (Int, Int) = (1, 1),
                            padding: (Int, Int) = (0, 0)
        ) -> Layer {
        return { (_ input) in
            let t1 = CACurrentMediaTime()
            let batchSize = input.shape[0]
            let channels = input.shape[1]
            let height = input.shape[2]
            let width = input.shape[3]
            
            let (poolW, poolH) = poolSize
            
            let (strideW, strideH) = strideStep ?? (poolW, poolH)
            
            let outH = Int((Float(height + 2 * padding.1 - poolH) / Float(strideH) + 1).rounded(.down))
            let outW = Int((Float(width + 2 * padding.0 - poolW) / Float(strideW) + 1).rounded(.down))
            
            let output = Variable(batchSize, channels, outH, outW)
            
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.size), 1, 1)
            let threadGroups = MTLSizeMake((output.size + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            let poolParam = PoolParam(inputParam: input.param,
                                      outputParam: output.param,
                                      poolSizeW: poolW,
                                      poolSizeH: poolH,
                                      strideW: strideW,
                                      strideH: strideH,
                                      padW: padding.0,
                                      padH: padding.1)
            
            let paramBuffer = makeBuffer(poolParam)
            
            submitWork(name: "Pool\(poolType.rawValue)", in: input, output, param: param, parameterBuffer: paramBuffer)

            let t2 = CACurrentMediaTime()
//            print("Pool \((t2 - t1) * 1000.0)")
            return output
        }
    }
}
