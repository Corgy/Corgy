//
//  Variable.swift
//  CorgyHost
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public typealias Layer = (_: Variable) -> Variable

public class Variable : CustomStringConvertible {
    public typealias DataType = Float
    var shape: [Int]
    private var count: Int
    public var value: [DataType]
    
    public init(_ dimensions: Int...) {
        shape = []
        count = 1
        for dimension in dimensions {
            shape.append(dimension)
            count *= dimension
        }
        value = [DataType](repeating: 0, count: count)
    }
    
    func validIndex(_ indices: [Int]) -> Bool {
        if indices.count != shape.count {
            return false
        }
        for (i, index) in indices.enumerated() {
            if index < 0 || index >= shape[i] {
                return false
            }
        }
        return true
    }
    
    func index(_ indices: [Int]) -> Int {
        assert(validIndex(indices))
        var ret = 0
        for i in 0..<shape.count {
            ret += indices[i] * shape[i]
        }
        return ret
    }
    
    subscript(indices: Int...) -> DataType {
        get {
            return value[index(indices)]
        }
        set {
            value[index(indices)] = newValue
        }
    }
    
    public var description: String {
        return "Shape: \(shape)\nvalue: \(value)\n"
    }
}
