//
//  MTLUtil.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Metal

let THREAD_PER_GROUP = 128

// TODO: temperal placeholder struct, needs refractor
struct TEMP_PARAM {
    
}

/// submit work to GPU work queue of a neural network and wait until finish.
///
/// - parameter input: Variables in input will be set to buffer of encoder
///             in order(from 1). Last one will be used as output.
func submitWork(_ resource: CorgyResource,
                       name function: String,
                       in input: Variable...,
                       param: TEMP_PARAM? = nil) {
    let output = input[input.count-1]
    // 坑: must create new command buffer and encoder for each compute task.
    let commandBuffer = resource.commandQueue.makeCommandBuffer()!
    let f = resource.library.makeFunction(name: function)!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    let pipelineState = try! resource.device.makeComputePipelineState(function: f)
    encoder.setComputePipelineState(pipelineState)
    
    var outputBuffer: MTLBuffer?, outputLength: Int?
    for (i, variable) in input.enumerated() {
        let length = variable.value.count * MemoryLayout<Variable.DataType>.stride
        // TODO: use bytes no copy, which might cause a large amount of refractor
        let buf = resource.device.makeBuffer(bytes: variable.value, length: length, options: [])
        encoder.setBuffer(buf, offset: 0, index: i)
        (outputBuffer, outputLength) = (buf, length)
    }
    
    let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.value.count), 1, 1)
    let threadGroups = MTLSizeMake((output.value.count + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
    
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
    
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // TODO: after makeBuffer(bytesNoCopy) is used, following code can be get rid of
    let content = NSData(bytesNoCopy: outputBuffer!.contents(), length: outputLength!, freeWhenDone: false)
    content.getBytes(&output.value, length: outputLength!)
}

