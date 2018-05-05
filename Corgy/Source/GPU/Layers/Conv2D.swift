//
//  GPUConv2D.swift
//  Corgy
//
//  Created by HOPE on 4/26/18.
//

import Foundation
import Metal

@available(OSX 10.13, *)
@available(iOS 10.0, *)
public extension Corgy {
    
    public static func Conv2D(inChannels: Int,
                              outChannels: Int,
                              kernelSize: Int,
                              stride: Int = 1,
                              padding: Int = 0,
                              dilation: Int = 1,
                              groups: Int = 1,
                              weight: Variable,
                              bias: Variable? = nil
        ) -> Layer {
        return { (_ input) in
            var inputShape = input.shape
            
            //FIXME: Just support one image
            assert(inputShape.count == 4 && inputShape[0] == 1)
            input.trimDimension(atMost: 1)
            
            inputShape = input.shape
            
            // FIXME: Serialized conversion, one performance bottleneck
            let m1 = imageToMatrix(image: input, kernelSize: kernelSize, padding: padding)
            let m2 = weightToMatrix(weight: weight)
            let res = m1 × m2
            
            let inputHeight = inputShape[1]
            let inputWidth  = inputShape[2]
            
            let outputHeight = inputHeight - kernelSize + 1
            let outputWidth = inputWidth - kernelSize + 1
            
            let output = Variable(outChannels, outputHeight, outputWidth)
            
            resultToVariable(input: res, output: output, bias: bias)

            // FIXME: preassume that number of image is 1
            var outputShape = output.shape
            outputShape.insert(1, at: 0)
            output.shape = outputShape
            
            return output
        }
    }
    
    /// convert a 3 dimension image with specified padding into a giant matrix.
    /// see [This slide](http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023) for reference.
    fileprivate static func imageToMatrix(image: Variable, kernelSize: Int, padding: Int) -> Variable {
        assert(image.shape.count == 3)
        let shape = image.shape
        let numChannel = shape[0]
        
        let paddedWidth = shape[2] + 2 * padding
        let paddedHeight = shape[1] + 2 * padding
        
        let kernelPerRow = paddedWidth - kernelSize + 1
        let kernelPerCol = paddedHeight - kernelSize + 1
        let sliceNumPerImage = kernelPerRow * kernelPerCol
        
        let outputWidth = numChannel * kernelSize * kernelSize
        let outputHeight = sliceNumPerImage
        let output = Variable(outputHeight, outputWidth)
        
        let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.size), 1, 1)
        let threadGroups = MTLSizeMake((output.size + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
        
        let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        
        let img2matParam = ImageToMatParam(inputParam: image.param, outputParam: output.param, kernelSize: kernelSize, padding: padding)
        let img2matParamBuffer = makeBuffer(img2matParam)
        
        submitWork(name: "ImageToMatrix", in: image, output, param: param, parameterBuffer: img2matParamBuffer)
        
        return output
    }
    
    fileprivate static func weightToMatrix(weight: Variable) -> Variable {
        let weightShape = weight.shape
        let inChannel = weightShape[1]
        
        let kernelSize = weightShape[3]
        let kernelSize²  = kernelSize * kernelSize
        
        let outChannel = weightShape[0]
        
        let outputWidth = outChannel
        let outputHeight = inChannel * kernelSize²
        
        let output = Variable(outputHeight, outputWidth)
        
        if output.size > 1000 {
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.size), 1, 1)
            let threadGroups = MTLSizeMake((output.size + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)

            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)

            let w2matParam = WeightToMatParam(inputParam: weight.param, outputParam: output.param)
            let w2matParamBuffer = makeBuffer(w2matParam)

            submitWork(name: "WeightToMatrix", in: weight, output, param: param, parameterBuffer: w2matParamBuffer)
        } else {
            for i in 0..<outputHeight {
                for j in 0..<outputWidth {
                    let h = i % kernelSize² / kernelSize
                    let w = i % kernelSize² % kernelSize
                    output[i, j] = weight[j, i / kernelSize², h, w]
                }
            }
        }
    
        return output
    }
    /// - parameter input: input Variable, result of previous matrix multiply
    /// - parameter bias: bias
    /// - parameter output: output Variable
    fileprivate static func resultToVariable(input: Variable, output: Variable, bias: Variable? = nil) {
        let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.size), 1, 1)
        let threadGroups = MTLSizeMake((output.size + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
        
        let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        
        if bias == nil {
            let mat2varParam = MatToVariableWithoutBias(inputParam: input.param, outputParam: output.param)
            let mat2varParamBuffer = makeBuffer(mat2varParam)
            
            submitWork(name: "Mat2VarWithoutBias", in: input, output, param: param, parameterBuffer: mat2varParamBuffer)
            return
        } else {
            let mat2varParam = MatToVariableWithBias(inputParam: input.param, biasParam: bias!.param, outputParam: output.param)
            let mat2varParamBuffer = makeBuffer(mat2varParam)
            submitWork(name: "Mat2VarWithBias", in: input, bias!, output, param: param, parameterBuffer: mat2varParamBuffer)
            return
        }
    }
}
