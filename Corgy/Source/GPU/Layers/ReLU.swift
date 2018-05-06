//
//  ReLU.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal

public extension Corgy {
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static let ReLU: Layer = { (_ input) in
        let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.count), 1, 1)
        let threadGroups = MTLSizeMake((input.count + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
        
        let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        submitWork(name: "ReLU", in: input, param: param)
        return input
    }
    
    public static let flatten = CPU.flatten
}
