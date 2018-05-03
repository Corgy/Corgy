//
//  Variable.swift
//  Corgy
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public typealias Layer = (_: Variable) -> Variable

public class Variable : CustomStringConvertible {
    public typealias DataType = Float32
    private var shape: [Int]
    
    private var count: Int
    public var value: [DataType]
    private var indexAuxilary: [Int]
    
    public func setShape(_ shape:[Int]) {
        self.shape = shape
        indexAuxilary = Array(repeating: 0, count: shape.count)
        indexAuxilary[shape.count - 1] = 1
        for i in (0..<shape.count-1).reversed() {
            indexAuxilary[i] = indexAuxilary[i + 1] * shape[i + 1]
        }
    }
    
    public func getShape() -> [Int]{
        return self.shape
    }
    
    /// dimension(shape): (batchSize, channels, height, width)
    public convenience init(_ dimensions: Int...) {
        self.init(dimensions)
    }
    
    public init(_ dimensions: [Int]) {
        indexAuxilary = []
        count = 1
        value = []
        shape = []
        
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
//        assert(validIndex(indices))
        var ret = 0
        for i in 0..<shape.count {
            ret += indices[i] * indexAuxilary[i]
        }
        return ret
    }
    
    public subscript(indices: Int...) -> DataType {
        get {
//            assert(indices.count == shape.count)
            return value[index(indices)]
        }
        set {
//            assert(indices.count == shape.count)
            value[index(indices)] = newValue
        }
    }
    
    // Trim leading 1 in the shape
    public func trimDimension(atMost: Int = Int.max) {
        var shape = getShape()
        var atMost = atMost
        while shape[0] == 1 && shape.count > 1 && atMost > 0 {
            shape.remove(at: 0)
            atMost = atMost - 1
        }
        setShape(shape)
    }
    
    private func recursiveSet(toSet: Variable, indices: [CountableClosedRange<Int>], sub: [Int]) {
        var sub = sub
        if sub.count == indices.count {
            let origSub = sub
            for i in 0..<sub.count {
                sub[i] -= indices[i].lowerBound
            }
            toSet[sub] = self[origSub]
            return
        }
        
        let currIndex = sub.count
        print(currIndex)
        for num in [Int](indices[currIndex].lowerBound...indices[currIndex].upperBound) {
            sub.append(num)
            recursiveSet(toSet: toSet, indices: indices, sub: sub)
            sub.remove(at: sub.count - 1)
        }
    }
    
    public subscript(indices: [Int]) -> DataType {
        get {
//            assert(indices.count == shape.count)
            return value[index(indices)]
        }
        set {
            value[index(indices)] = newValue
        }
    }
    
    public subscript(indices: CountableClosedRange<Int>...) -> Variable {
        get {
//            assert(indices.count == shape.count)
            let lens = indices.map { $0.count }
            let ret = Variable(lens)
            
            recursiveSet(toSet: ret, indices: indices, sub: [])
            // Variable.trimDimension(ret)
            return ret
        }
        set {
            fatalError()
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
        .map { DataType($0)! }
        return v
    }
    
    public func printOneLayer(batch: Int = 0, channel: Int = 0) {
        let input = self
        let numRow: Int
        let numCol: Int
        if input.getShape().count == 2 {
            numRow = input.getShape()[0]
            numCol = input.getShape()[1]
        } else {
            numRow = input.getShape()[2]
            numCol = input.getShape()[3]
        }
        
        for i in 0..<numRow {
            for j in 0..<numCol {
                let data: DataType
                if input.getShape().count == 2 {
                    data = input[i,j]
                } else {
                    data = input[batch,channel,i,j]
                }
                if data >= 0 {
                    print(" ", terminator: "")
                }
                print(String(format: " %.4f", data), terminator: "")
            }
            print("")
        }
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
