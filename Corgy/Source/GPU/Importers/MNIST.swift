//
//  ModelImporter.swift
//  Corgy
//
//  Created by HOPE on 4/23/18.
//

import Foundation

public class ModelImporter {
    public static func loadMNISTCNN(_ path: String) -> NeuralNetwork {
        let ret = NeuralNetwork()
        let kernelSize = 5

        var layers: [Variable] = []
        let names = ["conv1_weight", "conv1_bias", "conv2_weight", "conv2_bias", "fc_weight", "fc_bias"]

        for name in names {
            if let modelPath = Bundle.main.path(forResource: "MNIST_CNN_" + name, ofType: "txt", inDirectory: "Models") {
                let contentOfFile = try! String(contentsOfFile: modelPath)
                layers.append(Variable.fromString(input: contentOfFile))
            }
        }

        ret.add(CPU.Conv2D(inChannels: 1, outChannels: 16, kernelSize: kernelSize, weight: layers[0], bias: layers[1]))
            .add(CPU.ReLU)
            .add(CPU.Pool2D(poolSize: (2, 2), poolType: .Max))
            .add(CPU.Conv2D(inChannels: 16, outChannels: 32, kernelSize: kernelSize, weight: layers[2], bias: layers[3]))
            .add(CPU.ReLU)
            .add(CPU.Pool2D(poolSize: (2, 2), poolType: .Max))
            .add(CPU.flatten)
            .add(CPU.FullConnected(weight: layers[4], bias: layers[5]))
        
        return ret
    }
}
