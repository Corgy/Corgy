//
//  Layer.swift
//  Corgy
//
//  Created by buqian zheng on 4/19/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public struct GPU {
    private init() {}
    
    /// an inplace ReLU layer, it will modify
    /// and return the input
    public static let ReLU = { (_ input: Variable) -> Variable in
        for i in 0..<input.value.count {
            if input.value[i] < 0 {
                input.value[i] = 0
            }
        }
        return input
    }
}

extension GPU {
    
}
