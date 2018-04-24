//
//  Variable.swift
//  Corgy
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public typealias Layer = (_: Variable) -> Variable

public enum PoolType {
    case Average, Max
}

public class Variable : CustomStringConvertible {
    public typealias DataType = Float
    var shape: [Int]
    private var count: Int
    public var value: [DataType]
    private var indexAuxilary: [Int]
    
    private init() {
        shape = []
        value = []
        indexAuxilary = []
        count = 0
    }
    
    /// dimension(shape): (batchSize, channels, height, width)
    public convenience init(_ dimensions: Int...) {
        self.init(dimensions)
    }
    
    public init(_ dimensions: [Int]) {
        shape = []
        indexAuxilary = []
        count = 1
        for dimension in dimensions {
            shape.append(dimension)
            indexAuxilary.append(0)
            count *= dimension
        }
        indexAuxilary[shape.count - 1] = 1
        for i in (0..<shape.count-1).reversed() {
            indexAuxilary[i] = indexAuxilary[i + 1] * shape[i + 1]
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
            ret += indices[i] * indexAuxilary[i]
        }
        return ret
    }
    
    public subscript(indices: Int...) -> DataType {
        get {
            return value[index(indices)]
        }
        set {
            value[index(indices)] = newValue
        }
    }
    
    public var description: String {
        return "Shape: \(shape)\nvalue: \(value)\n\(indexAuxilary)"
    }
}

extension Variable {
    public func toString() -> String {
        var str = ""
        for s in shape {
            str.append(String(s) + " ")
        }
        str.append("\n")
        
        for f in value {
            str.append(String(f) + " ")
        }
        return str
    }
    static public func fromString(input: String) -> Variable {
        let lines = input.split(separator: "\n", maxSplits: 1)
        let head = lines[0]
        let shape = head.split(separator: " ").map { Int($0)! }
        let v = Variable(shape)
        let tail = lines[1]
        v.value = tail.replacingOccurrences(of: "\n", with: " ").split(separator: " ")
        .map { Float($0)! }
        return v
    }
}

extension Variable: Equatable {
    public static func ==(lhs: Variable, rhs: Variable) -> Bool {
        return lhs.count == rhs.count &&
                lhs.shape == rhs.shape &&
                lhs.indexAuxilary == rhs.indexAuxilary &&
                lhs.value == rhs.value
    }
}
