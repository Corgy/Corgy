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
    
    public func forward(_ x: Variable) -> Variable {
        var output = x
        var totalTime: CFTimeInterval = 0
        for (i, layer) in layers.enumerated() {
//            print("one layer, input: \(output.shape)", terminator: "")
            let t = timing() {
                output = layer(output)
            }
//            print("output: \(output.shape)\ttime: \(t*1000) ms")
            totalTime += t
        }
        print("Total time: \(totalTime)")
        return output
    }
    
    @discardableResult public func add(_ layer: @escaping Layer) -> NeuralNetwork {
        self.layers.append(layer)
        return self
    }
}

