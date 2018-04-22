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
    // TODO: replace ! with Non-optional type and do proper error return
    let device: MTLDevice!
    let library: MTLLibrary!
    let commandQueue: MTLCommandQueue!
    
    public init() {
        self.layers = []
        device = MTLCreateSystemDefaultDevice()
        library = device.makeDefaultLibrary()
        commandQueue = device.makeCommandQueue()
    }
    
    public func forward(_ x: Variable) ->Variable {
        var output = x
        for layer in layers {
            output = layer(output)
        }
        return output
    }
    
    @discardableResult public func add(_ layer: @escaping Layer) -> NeuralNetwork {
        self.layers.append(layer)
        return self
    }
}

