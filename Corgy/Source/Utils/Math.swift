//
//  Math.swift
//  Corgy
//
//  Created by buqian zheng on 5/4/18.
//

import Foundation

public enum Math {
    public static func sigmoid(_ x: Float) -> Float {
        return 1 / (1 + exp(-x))
    }
    
    public static func softMax(_ x: [Float]) -> [Float] {
        let sum = x.reduce(0, +)
        return x.map { $0 / sum}
    }
    
}

public extension Array where Element: Comparable {
    /// - require: array should have at least one element
    public func argmax() -> (Int, Element) {
        // FIXME
        assert(count > 0)
        var index = 0
        var element = self[0]
        
        for i in 1..<count {
            if self[i] > element {
                element = self[i]
                index = i
            }
        }
        return (index, element)
    }
}
