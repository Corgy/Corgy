//
//  NeuralNetwork.swift
//  Corgy
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
import Metal

public class NeuralNetwork {
    var layers : [Layer]
    
    public init() {
        self.layers = []
    }
    
    public func forward(_ x: Variable) ->Variable {
        var output = x
        for layer in layers {
//            print("Input: ")
//            output.printOneLayer()
            output = layer(output)
//            print("One layer!")
        }
        return output
    }
    
    @discardableResult public func add(_ layer: @escaping Layer) -> NeuralNetwork {
        self.layers.append(layer)
        return self
    }
}

