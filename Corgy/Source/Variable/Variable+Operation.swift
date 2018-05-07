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
    
    func makeBufferNoCopy() -> MTLBuffer {
        return Corgy.resource.device.makeBuffer(bytesNoCopy: pointer, length: MemoryLayout<DataType>.stride * actualCount, options: [], deallocator: nil)!
    }
    
    /// return a MPSMatrix based on self's data.
    ///
    /// returned MPSMatrix shares memory with self.
    /// - require: shape of self must be 2D
    public func toMPSMatrix () -> MPSMatrix {
        let vshape = self.shape
        assert(vshape.count == 2)
        let vrow = vshape[0]
        let vcol = vshape[1]
        
        let vd = MPSMatrixDescriptor(rows: vrow, columns: vcol, rowBytes: MemoryLayout<DataType>.stride * vcol, dataType: MPSDataType.float32)
        
        let vmatrix = MPSMatrix(buffer: makeBufferNoCopy(), descriptor: vd)
        
        return vmatrix
    }
    
    /// Padding a variable with shape of (c, h, w)
    /// only used in CPU version code
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
    
    var t1, t2: CFTimeInterval

    let v1shape = v1.shape
    let v2shape = v2.shape
    
    assert(v1shape[1] == v2shape[0])
    
    let v1row = v1shape[0]
    let v1col = v1shape[1]
    let v2col = v2shape[1]
    
    let result = Variable(v1row, v2col)
    
    t1 = currentMillsecond()
    
    let resm = result.toMPSMatrix()
    
    let v1m = v1.toMPSMatrix()
    let v2m = v2.toMPSMatrix()
    
    t2 = currentMillsecond()
//    print(String(format: "var to mps: %.4f", t2-t1), terminator: ",\t")
    
    let mul = MPSMatrixMultiplication(device: Corgy.resource.device,
                                      transposeLeft: false, transposeRight: false,
                                      resultRows: v1row, resultColumns: v2col,
                                      interiorColumns: v1col, alpha: 1, beta: 0)
    let commandBuffer = Corgy.resource.commandQueue.makeCommandBuffer()
    
    mul.encode(commandBuffer: commandBuffer!, leftMatrix: v1m, rightMatrix: v2m, resultMatrix: resm)
    
    commandBuffer!.commit()
    commandBuffer!.waitUntilCompleted()
    
//    print(String(format: "mps mul: %.4f", currentMillsecond()-t2), terminator: "")
    // doesn't need to create a variable out of resm since the underlying memory of resm is from
    // result, which is from makeBuffer(bytesNoCopy:). So return result directly is fine.
    return result
}
