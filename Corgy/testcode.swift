//
//  testcode.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Corgy

func test() {
    print("hello world")
    let nn = NeuralNetwork()
    
    let input = Variable(3,4)
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
    neg(input)
    print("\noutput:\n")
    print(input)
    print("\n")
    
    
}
