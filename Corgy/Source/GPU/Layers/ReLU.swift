//
//  ReLU.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal

extension Corgy {
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
}
