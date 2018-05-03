//
//  GPUConv2D.swift
//  Corgy
//
//  Created by HOPE on 4/26/18.
//

import Foundation
import Metal
import QuartzCore

@available(OSX 10.13, *)
@available(iOS 10.0, *)
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
            var inputShape = input.getShape()
            //FIXME: Just support one image
            assert(inputShape.count == 4 && inputShape[0] == 1)
            input.trimDimension(atMost: 1)
            inputShape = input.getShape()
            
            let t1 = CACurrentMediaTime()
            let m1 = imageToMatrix(image: input, kernelSize: kernelSize)
            let t2 = CACurrentMediaTime()
            let m2 = weightToMatrix(weight: weight, image: input)
            let t3 = CACurrentMediaTime()
            let res = Corgy.matrixMultiply(m1, m2)
            let intvl = CACurrentMediaTime() - t3;
            print("\tConv matrix op: \(t2-t1), \(t3-t2), \(intvl) seconds")
            
            let inputHeight = inputShape[1]
            let inputWidth  = inputShape[2]
            
            let outputHeight = inputHeight - kernelSize + 1
            let outputWidth = inputWidth - kernelSize + 1
            
            let output = Variable(outChannels, outputHeight, outputWidth)
            timing("\tGPU conv: ") {
                for c in 0..<outChannels {
                    for h in 0..<outputHeight {
                        for w in 0..<outputWidth {
                            output[c, h, w] = res[outputWidth * h + w, c]
                            
                        }
                    }
                }
                
                if bias.value.count > 0 {
                    for c in 0..<outChannels {
                        for h in 0..<outputHeight {
                            for w in 0..<outputWidth {
                                output[c, h, w] += bias[c]
                                
                            }
                        }
                    }
                }
                // print(res)
                
                // FIXME: preassume that number of image is 1
                var outputShape = output.getShape()
                outputShape.insert(1, at: 0)
                output.setShape(outputShape)
            }
        
            return output
        }
    }
    
    fileprivate static func imageToMatrix(image: Variable, kernelSize: Int) -> Variable {
        assert(image.getShape().count == 3)
        let shape = image.getShape()
        let numChannel = shape[0]
        let width = shape[1]
        let height = shape[2]
        
        let kernelSize²  = kernelSize * kernelSize
        let kernelPerRow = width - kernelSize + 1
        let kernelPerCol = height - kernelSize + 1
        let sliceNumPerImage = kernelPerRow * kernelPerCol
        
        let outputWidth = numChannel * kernelSize * kernelSize
        let outputHeight = sliceNumPerImage
        let output = Variable(outputHeight, outputWidth)
        
        // TODO: Launch Kernel for make these giant squares
        for i in 0..<outputHeight {
            for j in 0..<outputWidth {
                let channel = j / kernelSize²
                
                let num = j % kernelSize²
                let row = i / kernelPerRow + num / kernelSize
                let col = i % kernelPerRow + num % kernelSize
                
                output[i, j] = image[channel, row, col]
            }
        }
        
        return output
    }
    
    fileprivate static func weightToMatrix(weight: Variable, image: Variable) -> Variable {
        assert(image.getShape().count == 3)
        let weightShape = weight.getShape()
        let inChannel = weightShape[1]
        
        let kernelSize = weightShape[3]
        let kernelSize²  = kernelSize * kernelSize
        
        let outChannel = weightShape[0]
        
        let outputWidth = outChannel
        let outputHeight = inChannel * kernelSize²
        
        let output = Variable(outputHeight, outputWidth)
        
        for i in 0..<outputHeight {
            for j in 0..<outputWidth {
                let h = i % kernelSize² / kernelSize
                let w = i % kernelSize² % kernelSize
                output[i, j] = weight[j, i / kernelSize², h, w]
            }
        }
        
        return output
    }
}
