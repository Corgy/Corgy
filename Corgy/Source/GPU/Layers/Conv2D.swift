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
    // Padding a variable with shape of (c, h, w)
    public static func padding(_ input: Variable, paddingWith: Int) -> Variable {
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
            var input = input
            
            //FIXME: Just support one image
            assert(inputShape.count == 4 && inputShape[0] == 1)
            input.trimDimension(atMost: 1)
            
            // TODO: use soft padding(don't create a new Variable)
            input = Corgy.padding(input, paddingWith: padding)
            
            inputShape = input.shape
            
            // FIXME: Serialized conversion, one performance bottleneck
            let t1 = timing()
            let m1 = imageToMatrix(image: input, kernelSize: kernelSize)
            let t2 = timing()
            let m2 = weightToMatrix(weight: weight)
            let t3 = timing()
            let res = m1 × m2
            let intvl = timing() - t3;
//            print("\tConv matrix op: \(t2-t1), \(t3-t2), \(intvl) milliseconds")
            
            let inputHeight = inputShape[1]
            let inputWidth  = inputShape[2]
            
            let outputHeight = inputHeight - kernelSize + 1
            let outputWidth = inputWidth - kernelSize + 1
            
            let output = Variable(outChannels, outputHeight, outputWidth)
            
//            timing("\tresult to variable: ") {
            
                resultToVariable(input: res, output: output, bias: bias)

                // FIXME: preassume that number of image is 1
                var outputShape = output.shape
                outputShape.insert(1, at: 0)
                output.shape = outputShape
//            }
            
            return output
        }
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
    
    fileprivate static func imageToMatrix(image: Variable, kernelSize: Int) -> Variable {
        assert(image.shape.count == 3)
        let shape = image.shape
        let numChannel = shape[0]
        let width = shape[2]
        let height = shape[1]
        
        let kernelPerRow = width - kernelSize + 1
        let kernelPerCol = height - kernelSize + 1
        let sliceNumPerImage = kernelPerRow * kernelPerCol
        
        let outputWidth = numChannel * kernelSize * kernelSize
        let outputHeight = sliceNumPerImage
        let output = Variable(outputHeight, outputWidth)
        
        let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, output.size), 1, 1)
        let threadGroups = MTLSizeMake((output.size + THREAD_PER_GROUP - 1) / THREAD_PER_GROUP, 1, 1)
        
        let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
        
        let img2matParam = ImageToMatParam(inputParam: image.param, outputParam: output.param, kernelSize: kernelSize)
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
    
}
