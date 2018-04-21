//
//  MTLUtil.swift
//  Corgy iOS
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Metal

// TODO: add parameter such as poolsize, stride, pool type,
//   and/or param in convolution layer. Exact design and
//   implementation to be discussed and continued.
struct TEMP_PARAM {
    
}

func submitWork(_ network: NeuralNetwork,
                name function: String,
                data: Variable...,
                param: TEMP_PARAM? = nil,
                cmpl: MTLCommandBufferHandler? = nil) {
    let f = network.library.makeFunction(name: function)!
    let pipelineState = try! network.device.makeComputePipelineState(function: f)
    network.computeEncoder.setComputePipelineState(pipelineState)
    for (i, d) in data.enumerated() {
        let buffer = network.device.makeBuffer(bytes: d.value, length: d.value.count * MemoryLayout<Variable.DataType>.stride, options: [])
        network.computeEncoder.setBuffer(buffer, offset: 0, index: i)
    }
    // TODO: these are fake numbers
    let threadGroupCount = MTLSizeMake(8, 8, 1)
    let threadGroups = MTLSizeMake(8, 8, 1)
    network.computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
    network.computeEncoder.endEncoding()
    network.commandBuffer.commit()
    if let handler = cmpl {
        network.commandBuffer.addCompletedHandler(handler)
    } else {
        network.commandBuffer.waitUntilCompleted()
    }
}
