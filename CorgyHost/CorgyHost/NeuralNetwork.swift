//
//  NeuralNetwork.swift
//  CorgyHost
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation

class NeuralNetwork {
    var layers : [Layer]
    
    public init() {
        self.layers = []
    }
    
    public func forward(x: Variable) ->Variable {
        var output = x
        for layer in layers {
            output = layer(output)
        }
        return output
    }
    
    public func add(layer: @escaping Layer) -> NeuralNetwork {
        self.layers.append(layer)
        return self
    }
}

