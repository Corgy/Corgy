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
        
        // TODO: use bytes no copy
        let vbuffer = Corgy.resource.device.makeBuffer(bytes: &self.value, length: MemoryLayout<DataType>.size * self.size, options: [])!
        
        let vmatrix = MPSMatrix(buffer: vbuffer, descriptor: vd)
        
        return vmatrix
    }
    
    public static func of(_ mpsMatrix: MPSMatrix) -> Variable {
        let ncol = mpsMatrix.columns
        let nrow = mpsMatrix.rows
        let ret = Variable(nrow, ncol)
        
        let rawPointer = mpsMatrix.data.contents()
        
        memcpy(&ret.value, rawPointer, nrow * ncol * MemoryLayout<Variable.DataType>.stride)
        
        return ret
    }
    
    // Padding a variable with shape of (c, h, w)
    func padding(paddingWith: Int) -> Variable {
        let input = self
        let inputShape = input.shape
        assert(inputShape.count == 3)
        let c = inputShape[0]
        let h = inputShape[1]
        let w = inputShape[2]
        
        let output = Variable(c, h + 2 * paddingWith, w + 2 * paddingWith)
        
        for i in 0..<c {
            for j in 0..<h {
                for k in 0..<w {
                    output[i, j + paddingWith, k + paddingWith] = input[i, j, k]
                }
            }
        }
        
        return output
    }

}
infix operator ×
public func ×(_ v1: Variable, _ v2: Variable) -> Variable {
//    let start = currentMillsecond()
//
//    var t1, t2: CFTimeInterval
    
    let v1shape = v1.shape
    let v2shape = v2.shape
    
    assert(v1shape[1] == v2shape[0])
    
    let v1row = v1shape[0]
    let v1col = v1shape[1]
    let v2col = v2shape[1]
    
//    t1 = currentMillsecond()
    
    var result = Variable(v1row, v2col)
    
//    t2 = currentMillsecond()
//    print("New Variable: \(t2-t1)ms", terminator: ", ")
    
//    t1 = currentMillsecond()
    
    let resm = result.toMPSMatrix()
    
    let v1m = v1.toMPSMatrix()
    let v2m = v2.toMPSMatrix()
    
//    t2 = currentMillsecond()
//    print("To MPS Mat: \(t2-t1)ms", terminator: ", ")
    
//    t1 = currentMillsecond()
    let mul = MPSMatrixMultiplication(device: Corgy.resource.device,
                                      transposeLeft: false, transposeRight: false,
                                      resultRows: v1row, resultColumns: v2col,
                                      interiorColumns: v1col, alpha: 1, beta: 0)
    let commandBuffer = Corgy.resource.commandQueue.makeCommandBuffer()
    
    mul.encode(commandBuffer: commandBuffer!, leftMatrix: v1m, rightMatrix: v2m, resultMatrix: resm)
    
    commandBuffer!.commit()
    commandBuffer!.waitUntilCompleted()
    
//    t2 = currentMillsecond()
//    print("MPS Mat mul: \(t2-t1)ms", terminator: ", ")
    
    
//    t1 = currentMillsecond()
    
    result = Variable.of(resm)
    
//    t2 = currentMillsecond()
//    print("MPS to Variable: \(t2-t1)ms", terminator: ", ")
//
//    let end = currentMillsecond()
//    print("Total time: \(end-start) ms.")
    return result
}
