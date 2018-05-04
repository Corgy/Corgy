//
//  VariableOperation.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal
import MetalPerformanceShaders

@available(OSX 10.13, *)
@available(iOS 10.0, *)
extension Variable {
    public func toMPSMatrix () -> MPSMatrix {
        let vshape = self.shape
        assert(vshape.count == 2)
        let vrow = vshape[0]
        let vcol = vshape[1]
        
        let vd = MPSMatrixDescriptor(rows: vrow, columns: vcol, rowBytes: MemoryLayout<DataType>.size * vcol, dataType: MPSDataType.float32)
        
        let vbuffer = Corgy.resource.device.makeBuffer(bytes: &self.value, length: MemoryLayout<DataType>.size * self.size, options: [])!
        
        let vmatrix = MPSMatrix(buffer: vbuffer, descriptor: vd)
        
        return vmatrix
    }
    
    public static func of(_ mpsMatrix: MPSMatrix) -> Variable {
        let ncol = mpsMatrix.columns
        let nrow = mpsMatrix.rows
        let ret = Variable(nrow, ncol)
        
        let rawPointer = mpsMatrix.data.contents()
        let opaque = OpaquePointer(rawPointer)
        let pointer = UnsafeMutablePointer<Variable.DataType>(opaque)
        
        for i in 0..<nrow {
            for j in 0..<ncol {
                ret[i, j]  = pointer[i * ncol + j]
            }
        }
        
        return ret
    }

}
infix operator ×
public func ×(_ v1: Variable, _ v2: Variable) -> Variable {
    let v1shape = v1.shape
    let v2shape = v2.shape
    
    assert(v1shape[1] == v2shape[0])
    
    let v1row = v1shape[0]
    let v1col = v1shape[1]
    let v2col = v2shape[1]
    
    var result = Variable(v1row, v2col)
    let resm = result.toMPSMatrix()
    
    let v1m = v1.toMPSMatrix()
    let v2m = v2.toMPSMatrix()
    
    let mul = MPSMatrixMultiplication(device: Corgy.resource.device,
                                      transposeLeft: false, transposeRight: false,
                                      resultRows: v1row, resultColumns: v2col,
                                      interiorColumns: v1col, alpha: 1, beta: 0)
    let commandBuffer = Corgy.resource.commandQueue.makeCommandBuffer()
    
    mul.encode(commandBuffer: commandBuffer!, leftMatrix: v1m, rightMatrix: v2m, resultMatrix: resm)
    
    commandBuffer!.commit()
    commandBuffer!.waitUntilCompleted()
    result = Variable.of(resm)
    
    return result
}
