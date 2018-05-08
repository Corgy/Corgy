//
//  NeuralNetwork.swift
//  Corgy
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import Foundation
import Metal

/// TODO: There must be a better way to do this but I don't have
/// time, tomorrow is deadline so I'll stick with this ugly solution
public let CorgyApproachingLastLayer = "CorgyApproachingLastLayer"

public class NeuralNetwork {
    var layers : [Layer]
    
    public init() {
        self.layers = []
    }
    
    public func forward(_ x: Variable) -> Variable {
        var output = x
        for (i, layer) in layers.enumerated() {
            output = layer(output)
            if i == layers.count - 2 {
                let notification = NSNotification.Name(rawValue: CorgyApproachingLastLayer)
                NotificationCenter.default.post(name: notification, object: nil)
            }
        }
        return output
    }
    
    @discardableResult public func add(_ layer: @escaping Layer) -> NeuralNetwork {
        self.layers.append(layer)
        return self
    }
}

