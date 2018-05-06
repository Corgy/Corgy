//
//  SubmitWork.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal


extension Corgy {
    // TODO: temperal placeholder struct, needs refractor
    struct WorkParams {
        public var threadGroups : MTLSize!
        public var threadsPerThreadgroup : MTLSize!
        
        init(threadGroups: MTLSize!, threadsPerThreadgroup: MTLSize!) {
            self.threadGroups = threadGroups
            self.threadsPerThreadgroup = threadsPerThreadgroup
        }
    }
    
    static func makeBuffer<T: LayerParam>(_ param: T) -> MTLBuffer {
        var param = param
        return resource.device.makeBuffer(bytes: &param, length: MemoryLayout<T>.stride, options: [])!
    }
    
    static let resource = CorgyResource()
    
    /// submit work to GPU work queue of a neural network and wait until finish.
    ///
    /// - parameter input: Variables in input will be set to buffer of encoder
    ///             in order(from 0). Last one will be used as output.
    static func submitWork(name function: String,
                           in input: Variable...,
        param: WorkParams!,
        parameterBuffer: MTLBuffer? = nil) {
        
        // PITFALL: must create new command buffer and encoder for each compute task.
        let commandBuffer = resource.commandQueue.makeCommandBuffer()!
        let f = resource.library.makeFunction(name: function)!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = try! resource.device.makeComputePipelineState(function: f)
        encoder.setComputePipelineState(pipelineState)

        for (i, variable) in input.enumerated() {
            encoder.setBuffer(variable.makeBufferNoCopy(), offset: 0, index: i)
        }
        if let parameterBuf = parameterBuffer {
            encoder.setBuffer(parameterBuf, offset: 0, index: input.count)
        }
        let threadsPerThreadGroup = param.threadsPerThreadgroup
        let threadGroups = param.threadGroups
        
        encoder.dispatchThreadgroups(threadGroups!, threadsPerThreadgroup: threadsPerThreadGroup!)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
