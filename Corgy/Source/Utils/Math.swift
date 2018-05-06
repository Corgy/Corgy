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
        let xx = x.map { exp($0) }
        let sum = xx.reduce(0, +)
        let ret = x.map { exp($0) / sum}
        
        return ret
    }
    
    public static func gamma(_ linear : Float) -> Float {
        if(linear <= 0.03928) {
            return linear / 12.92
        } else {
           return pow((linear+0.055)/1.055, 2.4)
        }
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
