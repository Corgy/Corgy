//
//  Variable+File.swift
//  Corgy
//
//  Created by zbq on 5/1/18.
//

import Foundation

public extension Variable {
    fileprivate static let ToBeReplaced: [Character] = ["(", ")", ","]
    // TODO: how to directly read file into Variable's pointer
    /// return a Variable stored in binary file, shape of return Variable is described in shapeFile
    /// contents of shapeFile should be a string, format is space separated dimension.
    /// order: outChannel inChannel height width
    /// - parameter binaryFile: a path to binary file
    /// - parameter shapeFile: path to variable shape
    public static func of(binaryFile: String, shapeFile: String) -> Variable {
        let shapeString = try! String(contentsOfFile: shapeFile).replacingOccurrences(of: "[(),\n]+", with: "", options: .regularExpression, range: nil)
        let shape = shapeString.split(separator: " ").map { Int($0)! }
        let v = Variable(shape)
        guard let data = NSData(contentsOfFile: binaryFile) else {
            // TODO: add throws to this method
            fatalError()
        }
        data.getBytes(v.pointer, length: data.length)
        return v
    }
    
    // TODO: how to directly read file into Variable's pointer
    /// return a 1D Variable stored in binary file
    /// - parameter binaryFile: a path to binary file
    public static func of(binaryFile: String) -> Variable {
        guard let data = NSData(contentsOfFile: binaryFile) else {
            // TODO: add throws to this method
            fatalError()
        }
        let v = Variable(data.length / MemoryLayout<Variable.DataType>.stride)
        data.getBytes(v.pointer, length: data.length)
        return v
    }
    
    /// - parameter shapeFileName: if is nil, shape will not be write to file
    public func toFile(_ name: String, shapeFileName: String? = nil) {
        let data = NSData(bytes: pointer, length: actualCount * MemoryLayout<DataType>.stride)
        try! data.write(toFile: name)
    }
}
