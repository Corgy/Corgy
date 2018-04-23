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
        let conv_1_weight = Variable(1, 16, kernelSize, kernelSize)
        let conv_1_bias   = Variable(16)
        let conv_2_weight = Variable(16, 32, kernelSize, kernelSize)
        let conv_2_bias   = Variable(32)
        var fc_weight     = Variable(4 * 4 * 32, 10)
        let fc_bias       = Variable(10)
        
        if let modelPath = Bundle.main.path(forResource: "MNIST_CNN", ofType: "txt", inDirectory: "Models") {
            print(modelPath)
            do {
                let contentOfFile = try String(contentsOfFile: modelPath)
                let scanner = Scanner(string: contentOfFile)
                
                for out in 0..<16 {
                    for i in 0..<kernelSize {
                        for j in 0..<kernelSize {
                            scanner.scanFloat(&conv_1_weight[0, out, i, j])
                        }
                    }
                }
                
                for out in 0..<16 {
                    scanner.scanFloat(&conv_1_bias[out])
                }
                
                for outchannel in 0..<32 {
                    for inchannel in 0..<16 {
                        for i in 0..<kernelSize {
                            for j in 0..<kernelSize {
                                scanner.scanFloat(&conv_2_weight[inchannel, outchannel, i, j])
                            }
                        }
                    }
                }
                
                for channel in 0..<32 {
                    scanner.scanFloat(&conv_2_bias[channel])
                }
                
                
                for i in 0..<512/6 + 1 {
                    for k in 0..<10 {
                        for j in i * 6 ..< min((i+1)*6, 512) {
                            scanner.scanFloat(&fc_weight[j, k])
                        }
                    }
                }
                
                fc_weight = CPU.flatten(fc_weight)
                print(fc_weight)
                for i in 0..<10 {
                    scanner.scanFloat(&fc_bias[i])
                }
                
                ret.add(CPU.Conv2D(inChannels: 1, outChannels: 16, kernelSize: 5, weight: conv_1_weight, bias: conv_1_bias))
                    .add(CPU.ReLU)
                    .add(CPU.Pool2D(poolSize: (2, 2), poolType: .Max))
                    .add(CPU.Conv2D(inChannels: 16, outChannels: 32, kernelSize: 5, weight: conv_1_weight, bias: conv_1_bias))
                    .add(CPU.ReLU)
                    .add(CPU.Pool2D(poolSize: (2, 2), poolType: .Max))
                    .add(CPU.flatten)
                    .add(CPU.FullConnected(weight: fc_weight, bias: fc_bias))
                
            } catch {
                print("Failed to read text")
            }
        }
        return ret
    }
}
