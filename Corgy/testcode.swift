//
//  testcode.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Corgy

func test () {
    testGPU()
}

func testCPU() {
    let network = ModelImporter.loadMNISTCNN("MNIST_CNN")
    let image = Image(named: Image.Name("four"))!
    let input = Variable.of(grayScaleImage: image)
    let output = network.forward(input)
    print(output)
}

func testGPU() {
    let nn = NeuralNetwork()
    
    var input = Variable(32,16)
    for i in 0..<input.value.count {
        input.value[i] = Float(i)
    }
    
    let neg2 = Corgy.Neg2(network: nn)
    print("Neg Not inplace: \ninput:\n")
    print(input)
    print("\noutput:\n")
    print(neg2(input))
    print("\n")
    
    let neg = Corgy.Neg(network: nn)
    
    print("Inplace Neg: \ninput:\n")
    print(input)
    _ = neg(input)
    print("\noutput:\n")
    print(input)
    print("\n")
    
    let relu = Corgy.ReLU(network: nn)
    input = Variable(32,16)
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
