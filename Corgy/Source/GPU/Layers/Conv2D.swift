//
//  GPUConv2D.swift
//  Corgy
//
//  Created by HOPE on 4/26/18.
//

import Foundation
import Metal

public extension Corgy {
    public static func Conv2D(inChannels: Int,
                              outChannels: Int,
                              kernelSize: Int,
                              stride: Int = 1,
                              padding: Int = 0,
                              dilation: Int = 0,
                              groups: Int = 1,
                              weight: Variable,
                              bias: Variable = Variable(0)
        ) -> Layer {
        return { (_ input) in
            let output = Variable(input.getShape())
            return output
        }
    }
    
    fileprivate static func imageToMatrix(images: Variable, kernelSize: Int) -> Variable {
        assert(images.getShape().count == 3)
        let shape = images.getShape()
        let numChannel = shape[0]
        let width = shape[1]
        let height = shape[2]
        
        let sliceNumPerImage = (width - kernelSize + 1) * (height - kernelSize + 1)
        let sliceNum = sliceNumPerImage * numChannel
        
        let threadGroups = MTLSizeMake(getGroupNum(sliceNum, THREAD_PER_GROUP), 1, 1)
        let threadsPerThreadGroup = MTLSizeMake(THREAD_PER_GROUP, 1, 1)
        
        let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        
        let outputWidth = numChannel * kernelSize * kernelSize
        let outputHeight = sliceNumPerImage
        let output = Variable(outputHeight, outputWidth)
        
        submitWork(name: "imageToMatrix", in: images, output, param: param)
        
        return output
    }
}
