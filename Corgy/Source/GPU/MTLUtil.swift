//
//  MTLUtil.swift
//  Corgy iOS
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Metal

// TODO: temperal placeholder struct, needs refractor
struct TEMP_PARAM {
    
}

/// submit work to queue of a neural network
///
/// Always pass both or none of `threadGroups` and `threadPerThreadGroup`. If
/// both not provided, a default will be computed.
///
/// If output is not provided, the computation result will be stored inplace
/// in input. Else the result will be stored in output.
///
/// - parameter threadGroups: number of thread groups in a grid
func submitWork(_ network: NeuralNetwork,
                       name function: String,
                       in input: Variable,
                       out output: Variable? = nil,
                       param: TEMP_PARAM? = nil,
                       threadGroups: MTLSize,
                       threadsPerThreadgroup: MTLSize) {
    let commandBuffer = network.commandQueue.makeCommandBuffer()!
    let f = network.library.makeFunction(name: function)!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    let pipelineState = try! network.device.makeComputePipelineState(function: f)
    encoder.setComputePipelineState(pipelineState)
    let length = input.value.count * MemoryLayout<Variable.DataType>.stride
    
    // TODO: use bytes no copy, which might cause a large amount of refractor
    let buffer = network.device.makeBuffer(bytes: input.value, length: length, options: [])!
    encoder.setBuffer(buffer, offset: 0, index: 0)
    let outputBuffer: MTLBuffer, outputLength: Int
    if output != nil {
        let l = output!.value.count * MemoryLayout<Variable.DataType>.stride
        let buf = network.device.makeBuffer(bytes: output!.value, length: l, options: [])
        (outputBuffer, outputLength) = (buf!, l)
        encoder.setBuffer(buf, offset: 0, index: 1)
    } else {
        (outputBuffer, outputLength) = (buffer, length)
    }
    
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)
    
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let content = NSData(bytesNoCopy: outputBuffer.contents(), length: outputLength, freeWhenDone: false)
    if output != nil {
        content.getBytes(&output!.value, length: length)
    } else {
        content.getBytes(&input.value, length: length)
    }
}

