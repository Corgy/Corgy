//
//  Dropout.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal

public extension Corgy {
    /// return an inplace Dropout layer, it will modify
    /// and return the input
    /// - parameter p: percentage to be dropped out
    /// - parameter seed: optional, random seed
    public static func Dropout(p: Double, seed: UInt32? = nil) -> Layer {
        return { (_ input) in
            assert(p >= 0 && p <= 1)
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.count), 1, 1)
            let threadGroups = MTLSizeMake((input.count + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            let dropoutParam = DropoutParam(inputParam: input.param, p: p, seed: seed ?? arc4random())
            let paramBuffer = makeBuffer(dropoutParam)
            submitWork(name: "Dropout", in: input, param: param, parameterBuffer: paramBuffer)
            return input
        }
    }
}
