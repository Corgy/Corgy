//
//  TinyYolo.swift
//  Corgy
//
//  Created by zbq on 5/1/18.
//

import Foundation

public extension ModelImporter {
    public static func importYolo(computeOn: ComputeOn) -> NeuralNetwork {
        let ret = NeuralNetwork()
        
        switch computeOn {
        case .CPU:
            ret
                .add(convLayer(1, computeOn: .CPU)).add(CPU.ReLU)
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(2, computeOn: .CPU)).add(CPU.ReLU)
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(3, computeOn: .CPU)).add(CPU.ReLU)
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(4, computeOn: .CPU)).add(CPU.ReLU)
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(5, computeOn: .CPU)).add(CPU.ReLU)
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(6, computeOn: .CPU)).add(CPU.ReLU)
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(7, computeOn: .CPU)).add(CPU.ReLU)
                .add(convLayer(8, computeOn: .CPU)).add(CPU.ReLU)
                .add(convLayer(9, computeOn: .CPU, kernelSize: 1))
        default:
            ret
                .add(convLayer(1)).add(Corgy.ReLU)
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(2)).add(Corgy.ReLU)
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(3)).add(Corgy.ReLU)
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(4)).add(Corgy.ReLU)
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(5)).add(Corgy.ReLU)
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(6)).add(Corgy.ReLU)
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(7)).add(Corgy.ReLU)
                .add(convLayer(8)).add(Corgy.ReLU)
                .add(convLayer(9, kernelSize: 1))
        }
        
        return ret
    }
    
    fileprivate static let ConvLayerFileNamePrefix = "Corgy_conv"
    fileprivate static func convLayer(_ n: Int, computeOn: ComputeOn = .GPU, kernelSize: Int = 3) -> Layer {
        let weightFileName = "\(ConvLayerFileNamePrefix)\(n)_W"
        let weightShapeFileName = "\(ConvLayerFileNamePrefix)\(n)_W_shape"
        let biasFileName = "\(ConvLayerFileNamePrefix)\(n)_b"
        let weightFilePath = Bundle.main.path(forResource: weightFileName, ofType: "bin", inDirectory: "Models/yolo")!
        let shapePath = Bundle.main.path(forResource: weightShapeFileName, ofType: nil, inDirectory: "Models/yolo")!
        let biasPath = Bundle.main.path(forResource: biasFileName, ofType: "bin", inDirectory: "Models/yolo")!
        let weight = Variable.of(binaryFile: weightFilePath, shapeFile: shapePath)
        let bias = Variable.of(binaryFile: biasPath)
        switch computeOn {
        case .CPU:
            return CPU.Conv2D(inChannels: weight.shape[1], outChannels: weight.shape[0], kernelSize: 3, weight: weight, bias: bias)
        default:
            return Corgy.Conv2D(inChannels: weight.shape[1], outChannels: weight.shape[0], kernelSize: 3, weight: weight, bias: bias)
        }
    }
}
