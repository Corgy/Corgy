//
//  testcode.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Corgy
import QuartzCore

@available(OSX 10.13, *)
@available(iOS 10.0, *)
func test () {
    async {
        
//        let imageName = "four"
//        let imageName = "four_colored"
        let imageName = "car"
        #if os(iOS)
        let image = Image(named: imageName)!
        #elseif os(OSX)
        let image = Image(named: Image.Name(imageName))!
        #endif
        // GPUTest.MNIST(image: image)
        testYolo(image: image, computeOn: .GPU)
    }
}

@available(OSX 10.13, *)
@available(iOS 10.0, *)
enum CPUTest {
    static func MNIST(image: Image) {
        testMNIST(image: image, computeOn: .CPU)
    }
    
    static func testConv2D() {
        let weight = Variable(2, 1, 3, 3)
        for i in 0..<weight.count {
            weight.value[i] = Variable.DataType(i)
        }
        
        let bias   = Variable(2)
        bias.value[0] = Float(1)
        bias.value[1] = Float(2)
        
        let conv2d = CPU.Conv2D(inChannels:  1,
                                  outChannels: 2,
                                  kernelSize:  3,
                                  padding: 1,
                                  weight: weight,
                                  bias: bias
        )
        
        let input = Variable(1, 1, 4, 4)
        for i in 0..<input.count {
            input.value[i] = Variable.DataType(i)
        }
        
        let output = conv2d(input)
        
        print(output)
    }
}


@available(OSX 10.13, *)
@available(iOS 10.0, *)
func testMNIST(image: Image, computeOn: ComputeOn) {
    let network = ModelImporter.loadMNISTCNN("MNIST_CNN", computeOn: computeOn)
    timing("\(computeOn)") {
        let input = Variable.of(grayScaleImage: image)//), to: (28, 28))
        let output = network.forward(input)
        print(output)
    }
}

@available(OSX 10.13, *)
@available(iOS 10.0, *)
func testYolo(image: Image, computeOn: ComputeOn) {
    let network = ModelImporter.importYolo(computeOn: computeOn)
    let input = Variable.of(image: image, to: (416, 416))
    let output = network.forward(input)
    let boxes = ModelImporter.getResult(input: output)
    boxes.sorted(by: { (a, b) -> Bool in
        return a.score > b.score
    }).forEach { print($0) }
}

@available(OSX 10.13, *)
@available(iOS 10.0, *)
enum GPUTest {
    static func testReLU() {
        let relu = Corgy.ReLU
        let input = Variable(32,33)
        for i in 0..<input.count {
            input.value[i] = Variable.DataType(i)
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
    
    static func testPoolMax() {
        let input = Variable(1, 1, 32, 16)
        for i in 0..<input.count {
            input.value[i] = Variable.DataType(i)
        }
        let poolMax = Corgy.Pool(poolSize: (2, 2), poolType: .Max)
        print("Max pooling layer: \ninput:\n")
        input.printOneLayer(batch: 0, channel: 0)
        let output = poolMax(input)
        print("\noutput: \n")
        output.printOneLayer(batch: 0, channel: 0)
    }
    
    static func testPoolAvg() {
        let input = Variable(1, 1, 32, 16)
        for i in 0..<input.count {
            input.value[i] = Variable.DataType(i)
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
        for i in 0..<input.count {
            input.value[i] = Variable.DataType(i)
        }
        let poolAve = Corgy.Dropout(p: 0.9)
        print("Dropout layer: \ninput:\n")
        input.printOneLayer(batch: 0, channel: 0)
        print("\noutput: \n")
        poolAve(input).printOneLayer(batch: 0, channel: 0)
        print("\n")
    }
    
    static func testConv2D() {
        let weight = Variable(2, 1, 3, 3)
        for i in 0..<weight.count {
            weight.value[i] = Variable.DataType(i)
        }
    
        let bias   = Variable(2)
        bias.value[0] = Float(1)
        bias.value[1] = Float(2)

        let conv2d = Corgy.Conv2D(inChannels:  1,
                                  outChannels: 2,
                                  kernelSize:  3,
                                  padding: 1,
                                  weight: weight,
                                  bias: bias
        )

        let input = Variable(1, 1, 4, 4)
        for i in 0..<input.count {
            input.value[i] = Variable.DataType(i)
        }

        let output = conv2d(input)

        print(output)
    }
    
    @available(OSX 10.13, *)
    @available(iOS 10.0, *)
    static func testMultiply() {
        let v1 = Variable(4, 4)
        for i in 0..<v1.count {
            v1.value[i] = Variable.DataType(i);
        }
        
        let v2 = Variable(4, 4)
        for i in 0..<v2.count {
            v2.value[i] = Variable.DataType(i);
        }
        
        print(v1 × v2)
    }
    
    static func testVariabel() {
        let v = Variable(3, 3, 3)
        for i in 0..<v.count {
            v.value[i] = Variable.DataType(i)
        }
        
        let x = v[1...1, 1...2, 1...2]
        
        print(x)
    }
    
    static func MNIST(image: Image) {
        testMNIST(image: image, computeOn: .GPU)
    }
    
    static func testLeakyReLU() {
        let leakyReLU = Corgy.LeakyReLU()
        let input = Variable(1,1,1,5)
        
        for i in 0..<5 {
            if i % 2 == 0 {
                input.value[i] = Float(-i)
            } else {
                input.value[i] = Float(i)
            }
        }
        
        let output = leakyReLU(input)
        print(output)
    }
}
