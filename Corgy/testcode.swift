//
//  testcode.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Corgy

@available(OSX 10.13, *)
func test () {
    GPUTest.testConv2D()
}

func testCPU() {
    let network = ModelImporter.loadMNISTCNN("MNIST_CNN")
    let image = Image(named: Image.Name("four"))!
    let input = Variable.of(grayScaleImage: image)
    let output = network.forward(input)
    print(output)
}

@available(OSX 10.13, *)
enum GPUTest {
    static func testReLU() {
        let relu = Corgy.ReLU()
        let input = Variable(32,16)
        for i in 0..<input.value.count {
            input.value[i] = Float(i)
            if i % 2 == 0 {
                input.value[i] = -input.value[i]
            }
        }
        print("Inplace ReLU: \ninput:\n")
        print(input)
        _ = relu(input)
        print("\noutput:\n")
        print(input)
        print("\n")
    }
    static func testNeg() {
        let input = Variable(32,16)
        for i in 0..<input.value.count {
            input.value[i] = Float(i)
        }
        
        let neg2 = Corgy.Neg2()
        print("Neg Not inplace: \ninput:\n")
        print(input)
        print("\noutput:\n")
        print(neg2(input))
        print("\n")
        
        let neg = Corgy.Neg()
        
        print("Inplace Neg: \ninput:\n")
        print(input)
        _ = neg(input)
        print("\noutput:\n")
        print(input)
        print("\n")
    }
    static func testPoolMax() {
        let input = Variable(1, 1, 32, 16)
        for i in 0..<input.value.count {
            input.value[i] = Float(i)
        }
        let poolMax = Corgy.Pool(poolSize: (2, 2), poolType: .Max)
        print("Max pooling layer: \ninput:\n")
        input.printOneLayer(batch: 0, channel: 0)
        print("\noutput: \n")
        poolMax(input).printOneLayer(batch: 0, channel: 0)
        print("\n")
    }
    static func testPoolAvg() {
        let input = Variable(1, 1, 32, 16)
        for i in 0..<input.value.count {
            input.value[i] = Float(i)
        }
        let poolAve = Corgy.Pool(poolSize: (2, 2), poolType: .Average)
        print("Avg pooling layer: \ninput:\n")
        input.printOneLayer(batch: 0, channel: 0)
        print("\noutput: \n")
        poolAve(input).printOneLayer(batch: 0, channel: 0)
        print("\n")
    }
    
    static func testDropout() {
        let input = Variable(1, 1, 32, 16)
        for i in 0..<input.value.count {
            input.value[i] = Float(i)
        }
        let poolAve = Corgy.Dropout(p: 0.9)
        print("Dropout layer: \ninput:\n")
        input.printOneLayer(batch: 0, channel: 0)
        print("\noutput: \n")
        poolAve(input).printOneLayer(batch: 0, channel: 0)
        print("\n")
    }
    
    static func testConv2D() {
        let weight = Variable(1, 2, 2, 2)
        for i in 0..<weight.value.count {
            weight.value[i] = Float(i)
        }
        
        let bias   = Variable(2)
        bias.value = [1, 2]
        
        let conv2d = Corgy.Conv2D(inChannels:  1,
                                  outChannels: 2,
                                  kernelSize:  2,
                                  weight: weight,
                                  bias: bias
        )
        
        let input = Variable(1, 4, 4)
        for i in 0..<input.value.count {
            input.value[i] = Float(i)
        }
        
        let output = conv2d(input)
        
        print(output)
    }
    
    @available(OSX 10.13, *)
    static func testMultiply() {
        let v1 = Variable(4, 4)
        for i in 0..<v1.value.count {
            v1.value[i] = Float(i);
        }
        
        let v2 = Variable(4, 4)
        for i in 0..<v2.value.count {
            v2.value[i] = Float(i);
        }
        
        print(Corgy.matrixMultiply(v1, v2))
    }
}
