//
//  MTLUtil.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Metal

// TODO: temperal placeholder struct, needs refractor
struct WorkParams {
    public var threadGroups : MTLSize!
    public var threadsPerThreadgroup : MTLSize!
    
    init(threadGroups: MTLSize!, threadsPerThreadgroup: MTLSize!) {
        self.threadGroups = threadGroups
        self.threadsPerThreadgroup = threadsPerThreadgroup
    }
}

extension Corgy {
    
    class CorgyResource {
        let device: MTLDevice!
        let library: MTLLibrary!
        let commandQueue: MTLCommandQueue!
        init() {
            device = MTLCreateSystemDefaultDevice()
            library = device.makeDefaultLibrary()
            commandQueue = device.makeCommandQueue()
        }
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
        let output = input[input.count-1]
        // PITFALL: must create new command buffer and encoder for each compute task.
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
        if let parameterBuf = parameterBuffer {
            encoder.setBuffer(parameterBuf, offset: 0, index: input.count)
        }
        
        let threadsPerThreadGroup = param.threadsPerThreadgroup
        let threadGroups = param.threadGroups
        
        encoder.dispatchThreadgroups(threadGroups!, threadsPerThreadgroup: threadsPerThreadGroup!)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // TODO: after makeBuffer(bytesNoCopy) is used, following code can be get rid of
        let content = NSData(bytesNoCopy: outputBuffer!.contents(), length: outputLength!, freeWhenDone: false)
        content.getBytes(&output.value, length: outputLength!)
    }
    
    // 垃圾语言。如果这个函数是这么声明的，就会错：
    // static func makeBuffer(param: LayerParam, length: Int) -> MTLBuffer
    // 如果是这么声明的，又是对的：
    // static func makeBuffer(param: PoolParam, length: Int) -> MTLBuffer
    // 仿佛param的声明类型会影响它的内存布局一样
    static func makeBuffer<T: LayerParam>(_ param: T) -> MTLBuffer {
        var param = param
        return resource.device.makeBuffer(bytes: &param, length: MemoryLayout<T>.stride, options: [])!
    }
    
}
