//
//  Layer.swift
//  CorgyHost
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

typealias Layer = (_: Variable) -> Variable

/// the namespace for CPU functional layer
/// - ReLU
/// - Pooling
enum CPU {
    /// an inplace ReLU layer, it will modify
    /// and return the input
    static let relu = { (_ input: Variable) -> Variable in
        for i in 0..<input.value.count {
            if input.value[i] < 0 {
                input.value[i] = 0
            }
        }
        return input
    }
    
}
