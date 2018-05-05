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
                .add(convLayer(1, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(2, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(3, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(4, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(5, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(CPU.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(6, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(CPU.Pool(poolSize: (1, 1), stride: (1, 1), poolType: .Max))
                .add(convLayer(7, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(convLayer(8, computeOn: .CPU)).add(CPU.LeakyReLU())
                .add(convLayer(9, computeOn: .CPU, kernelSize: 1)).add(CPU.ReLU)
        case .GPU:
            ret
                .add(convLayer(1, padding: 1)).add(Corgy.LeakyReLU())
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(2, padding: 1)).add(Corgy.LeakyReLU())
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(3, padding: 1)).add(Corgy.LeakyReLU())
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(4, padding: 1)).add(Corgy.LeakyReLU())
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(5, padding: 1)).add(Corgy.LeakyReLU())
                .add(Corgy.Pool(poolSize: (2, 2), poolType: .Max))
                .add(convLayer(6, padding: 1)).add(Corgy.LeakyReLU())
                .add(Corgy.Pool(poolSize: (1, 1), stride: (1, 1), poolType: .Max))
                .add(convLayer(7, padding: 1)).add(Corgy.LeakyReLU())
                .add(convLayer(8, padding: 1)).add(Corgy.LeakyReLU())
                .add(convLayer(9, kernelSize: 1)).add(Corgy.ReLU)
        }
        
        return ret
    }
    
    fileprivate static let ConvLayerFileNamePrefix = "corgy_voc_conv"
    fileprivate static let ConvLayerFileDirectory = "Models/yolo-voc"
    fileprivate static func convLayer(_ n: Int,
                                      computeOn: ComputeOn = .GPU,
                                      kernelSize: Int = 3,
                                      padding: Int = 0) -> Layer {
        let weightFileName = "\(ConvLayerFileNamePrefix)\(n)_W"
        let weightShapeFileName = "\(ConvLayerFileNamePrefix)\(n)_W_shape"
        let biasFileName = "\(ConvLayerFileNamePrefix)\(n)_b"
        let weightFilePath = Bundle.main.path(forResource: weightFileName, ofType: "bin", inDirectory: ConvLayerFileDirectory)!
        let shapePath = Bundle.main.path(forResource: weightShapeFileName, ofType: nil, inDirectory: ConvLayerFileDirectory)!
        let biasPath = Bundle.main.path(forResource: biasFileName, ofType: "bin", inDirectory: ConvLayerFileDirectory)!
        let weight = Variable.of(binaryFile: weightFilePath, shapeFile: shapePath)
        let bias = Variable.of(binaryFile: biasPath)
        switch computeOn {
        case .CPU:
            return CPU.Conv2D(inChannels: weight.shape[1],
                              outChannels: weight.shape[0],
                              kernelSize: kernelSize,
                              padding: padding,
                              weight: weight,
                              bias: bias)
        case .GPU:
            return Corgy.Conv2D(inChannels: weight.shape[1],
                                outChannels: weight.shape[0],
                                kernelSize: kernelSize,
                                padding: padding,
                                weight: weight,
                                bias: bias)
        }
    }
    
    fileprivate static let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    fileprivate static let confidenceThreshold: Float = 0.3
    
    /// input is the output of YOLO network, must be of shape [1, 125, 13, 13}
    fileprivate static let getResult: Layer = { (_ input: Variable) in
        assert(input.size == 125*13*13)
        
        let numCellX = 13
        let numCellY = 13
        let numBox = 5
        let numClasses = 20
        
        struct Box {
            let x, y, w, h, score: Variable.DataType
            let klassIndex: Int
        }
        
        /// get box info at box b at cell (cx, cy)
        /// b must be between 0 and 4
        /// cx and cy must be between 0 and 12
        func get(cx: Int, cy: Int, b: Int) -> Box {
            var klasses: [Variable.DataType] = Array(repeating: 0, count: numClasses)
            for i in 0..<20 {
                klasses[i] = input[0, b * 5 + 5 + i, cy, cx]
            }
            klasses = Math.softMax(klasses)
            let (klassIndex, klass) = klasses.argmax()
            return Box(x: input[0, b * 5, cy, cx],          // TODO
                       y: input[0, b * 5 + 1, cy, cx],
                       w: input[0, b * 5 + 2, cy, cx],
                       h: input[0, b * 5 + 3, cy, cx],
                       score: klass * Math.sigmoid(input[0, b * 5 + 4, cy, cx]),
                       klassIndex: klassIndex)
        }
        var boxes: [Box] = []
        for cy in 0..<numCellY {
            for cx in 0..<numCellX {
                for b in 0..<numBox {
                    let box = get(cx: cx, cy: cy, b: b)
                    if box.score > confidenceThreshold {
                        boxes.append(box)
                    }
                }
            }
        }
        
    }
}
