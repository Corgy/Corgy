//
//  Variable.swift
//  Corgy
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

public class Variable : CustomStringConvertible {
    public typealias DataType = Float32
    internal(set) public var shape: [Int] {
        didSet {
            indexAuxilary = Array(repeating: 0, count: shape.count)
            indexAuxilary[shape.count - 1] = 1
            for i in (0..<shape.count-1).reversed() {
                indexAuxilary[i] = indexAuxilary[i + 1] * shape[i + 1]
            }
        }
    }
    
    /// count: number of elements this variable should be able to hold.
    ///
    /// `count == shape.reduce(1, *)` should be true
    private(set) public var count: Int
    
    /// number of elements underlying data structure can hold.
    /// because in order to use makeBuffer(bytesNoCopy:) methods in Metal,
    /// the pointer must be aligned with memory page, and the memory size
    /// must be divisible by pagesize, else makeBuffer(bytesNoCopy:) will
    /// return nil.
    ///
    /// Example:
    ///
    /// if pageSize is 2048 bytes, then the underlying memory size must be
    /// some integer times 2048 bytes. So if one element is of 4 bytes, then
    /// actualCount will be integer multiple of 512.
    ///
    /// when passing size of memory to makeBuffer(bytesNoCopy:) function,
    /// pass `actualCount * MemoryLayout<Variable.Datatype>.stride`
    /// rather than `count * MemoryLayout<Variable.Datatype>.stride`
    let actualCount: Int
    
    // TODO: make this internal
    public var value: UnsafeMutableBufferPointer<DataType>
    /// poiner and value always point at same place...
    /// because UnsafeMutableBufferPointer can be used easier
    /// as a collection, but many Metal and MPS function requires RawPointer...
    /// so both are provided as instance member.
    var pointer: UnsafeMutableRawPointer
    
    var indexAuxilary: [Int]
    
    /// dimension(shape): (batchSize, channels, height, width)
    public convenience init(_ dimensions: Int...) {
        self.init(dimensions)
    }
    
    public init(_ dimensions: [Int]) {
        indexAuxilary = []
        count = 1
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
        
        let pageSize = Int(getpagesize())
        let dataSize = count * MemoryLayout<DataType>.stride
        let pageCount = (dataSize + pageSize - 1) / pageSize
        actualCount = pageSize * pageCount / MemoryLayout<DataType>.stride
        
        var xvector: UnsafeMutableRawPointer? = nil
        posix_memalign(&xvector, pageSize, pageSize * pageCount)
        pointer = xvector!
        value = UnsafeMutableBufferPointer(start: xvector?.assumingMemoryBound(to: DataType.self), count: count)
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
    
    @inline(__always)
    func index(_ indices: [Int]) -> Int {
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
    
    
    // Trim leading 1 in the shape
    @discardableResult
    public func trimDimension(atMost: Int = Int.max) -> Variable {
        var atMost = atMost
        while shape[0] == 1 && shape.count > 1 && atMost > 0 {
            shape.remove(at: 0)
            atMost = atMost - 1
        }
        return self
    }
    
    private func recursiveSet(toSet: Variable, indices: [CountableClosedRange<Int>], sub: inout [Int]) {
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
            recursiveSet(toSet: toSet, indices: indices, sub: &sub)
            sub.remove(at: sub.count - 1)
        }
    }
    
    public subscript(indices: [Int]) -> DataType {
        get {
            return value[index(indices)]
        }
        set {
            value[index(indices)] = newValue
        }
    }
    
    public subscript(indices: CountableClosedRange<Int>...) -> Variable {
        get {
            let lens = indices.map { $0.count }
            let ret = Variable(lens)
            var sub: [Int] = []
            recursiveSet(toSet: ret, indices: indices, sub: &sub)
            // Variable.trimDimension(ret)
            return ret
        }
        set {
            fatalError()
        }
    }
    
    public var description: String {
        return toString()
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
        let tail = lines[1].replacingOccurrences(of: "\n", with: " ").split(separator: " ")
        for (i, e) in tail.enumerated() {
            v.value[i] = DataType(e)!
        }
        return v
    }
    
    public func printOneLayer(batch: Int = 0, channel: Int = 0) {
        let input = self
        let numRow: Int
        let numCol: Int
        if input.shape.count == 2 {
            numRow = input.shape[0]
            numCol = input.shape[1]
        } else {
            numRow = input.shape[2]
            numCol = input.shape[3]
        }
        
        for i in 0..<numRow {
            for j in 0..<numCol {
                let data: DataType
                if input.shape.count == 2 {
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
    
    public func printTransposed() {
        let input = self
        let numRow: Int
        let numCol: Int
        if input.shape.count == 2 {
            numRow = input.shape[0]
            numCol = input.shape[1]
        } else {
            numRow = input.shape[2]
            numCol = input.shape[3]
        }
        var cnt = 0
        for j in 0..<numCol {
            for i in 0..<numRow {
                for k in 0..<input.shape[1] {
                    cnt += 1
                    
                    if cnt > 1000{ break }
                    let data: DataType
                    if input.shape.count == 2 {
                        data = input[j,i]
                    } else {
                        data = input[0,k,j,i]
                    }
                    
//                    if data >= 0 {
//                        print(" ", terminator: "")
//                    }
                    print(String(format: " %.4f", data))
                    if (cnt % 16 == 0) {
                        print("")
                    }
                }
            }
        }
    }
}

extension Variable: Equatable {
    public static func ==(lhs: Variable, rhs: Variable) -> Bool {
        if !(lhs.count == rhs.count &&
            lhs.shape == rhs.shape &&
            lhs.indexAuxilary == rhs.indexAuxilary) {
            return false
        }
        for i in 0..<lhs.count {
            if lhs.value[i] != rhs.value[i] {
                return false
            }
        }
        return true
    }
}
