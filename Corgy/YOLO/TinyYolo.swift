//
//  TinyYolo.swift
//  Corgy
//
//  Created by zbq on 5/1/18.
//

import Foundation
import Corgy
#if os(iOS)
import UIKit
#elseif os(OSX)
import AppKit
#endif


public extension ModelImporter {
    public struct Box {
        public let x, y, w, h, score: Variable.DataType
        public let klassIndex: Int
        public func rect(in frame: CGRect) -> CGRect {
            let scaledX = CGFloat(x / 416.0) * frame.size.width
            let scaledY = CGFloat(y / 416.0) * frame.size.height
            let scaledW = CGFloat(w / 416.0) * frame.size.width
            let scaledH = CGFloat(h / 416.0) * frame.size.height
            return CGRect(x: scaledX, y: scaledY, width: scaledW, height: scaledH)
        }
    }
    
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
                .add(convLayer(9, computeOn: .CPU, kernelSize: 1))
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
                .add(convLayer(9, kernelSize: 1))
        }
        
        return ret
    }
    
    
    /*
     * The difference between t-yolo and yolo-voc is that,
     * weights in t-yolo is pretransformed into matrix as describted [here](http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023)
     * But to use pretransformed weights, original shape of weights must also be provided.
     * convLayer distinguish this by check if there is original shape file in folder.
     * To get transformed weights from original weights, use the weight_transform.py script
     * provided in this repo.
     */
    fileprivate static let ConvLayerFileNamePrefix = "t_corgy_voc_conv" // "corgy_voc_conv"
    fileprivate static let ConvLayerFileDirectory = "Models/t-yolo"     // "Models/yolo-voc"
    
    fileprivate static func convLayer(_ n: Int,
                                      computeOn: ComputeOn = .GPU,
                                      kernelSize: Int = 3,
                                      padding: Int = 0) -> Layer {
        let weightFileName = "\(ConvLayerFileNamePrefix)\(n)_W"
        /// transformed 2d weight shape
        let weightShapeFileName = "\(ConvLayerFileNamePrefix)\(n)_W_shape"
        /// original 4d weight shape
        let weightRealShapeFileName = "\(ConvLayerFileNamePrefix)\(n)_W_real_shape"
        let biasFileName = "\(ConvLayerFileNamePrefix)\(n)_b"
        let weightFilePath = Bundle.main.path(forResource: weightFileName, ofType: "bin", inDirectory: ConvLayerFileDirectory)!
        let shapePath = Bundle.main.path(forResource: weightShapeFileName, ofType: nil, inDirectory: ConvLayerFileDirectory)!
        let realShapePath = Bundle.main.path(forResource: weightRealShapeFileName, ofType: nil, inDirectory: ConvLayerFileDirectory)
        let biasPath = Bundle.main.path(forResource: biasFileName, ofType: "bin", inDirectory: ConvLayerFileDirectory)!
        let weight = Variable.of(binaryFile: weightFilePath, shapeFile: shapePath)
        var weightShape = weight.shape
        var weightTransformed = false
        if realShapePath != nil {
            let shapeString = try! String(contentsOfFile: realShapePath!).replacingOccurrences(of: "[(),\n]+", with: "", options: .regularExpression, range: nil)
            weightShape = shapeString.split(separator: " ").map { Int($0)! }
            weightTransformed = true
        }
        let bias = Variable.of(binaryFile: biasPath)
        switch computeOn {
        case .CPU:
            return CPU.Conv2D(inChannels: weightShape[1],
                              outChannels: weightShape[0],
                              kernelSize: kernelSize,
                              padding: padding,
                              weight: weight,
                              bias: bias)
        case .GPU:
            return Corgy.Conv2D(inChannels: weightShape[1],
                                outChannels: weightShape[0],
                                kernelSize: kernelSize,
                                padding: padding,
                                weight: weight,
                                bias: bias,
                                weightTransformed: weightTransformed)
        }
    }
    
    fileprivate static let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    fileprivate static let confidenceThreshold: Float = 0.1
    
    public static let voc_labels = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    
    /// input is the output of YOLO network, must be of shape [1, 125, 13, 13}
    public static func getResult(input: Variable) -> [Box] {
//        assert(input.count == 125*13*13)
        let numCellX = 13
        let numCellY = 13
        let layerSize = 169 // 13 * 13
        let numBox = 5
        let numClasses = 20
        let inputWidth  = 416
        let cellSize = Float(inputWidth) / Float(numCellY)
        
        /// get box info at box b at cell (cx, cy)
        /// b must be between 0 and 4
        /// cx and cy must be between 0 and 12
        func get(cx: Int, cy: Int, b: Int) -> Box {
            
            let oy = Math.sigmoid(input[0, b * 25, cy, cx])
            let ox = Math.sigmoid(input[0, b * 25 + 1, cy, cx])
            let ow = input[0, b * 25 + 2, cy, cx]
            let oh = input[0, b * 25 + 3, cy, cx]
            
            var klasses: [Variable.DataType] = Array(repeating: 0, count: numClasses)
            for i in 0..<20 {
                klasses[i] = input[0, b * 25 + 5 + i, cy, cx]
            }
            klasses = Math.softMax(klasses)
            let (klassIndex, klass) = klasses.argmax()
            
            // the real pixel of the center of infenrenced box
            let realy = (Float(cy) + oy) * cellSize
            let realx = (Float(cx) + ox) * cellSize
            
            let realW = exp(ow) * anchors[2 * b] * cellSize
            let realH = exp(oh) * anchors[2 * b + 1] * cellSize
            
            return Box(x: realx - realW / 2,
                       y: realy - realH / 2,
                       w: realW,
                       h: realH,
                       score: klass * Math.sigmoid(input[0, b * 25 + 4, cy, cx]),
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
        
        return boxes
    }
    
    public static func nonMaxSuppression(boxes: [Box], limit: Int, threshold: Float) -> [Box] {
        let sortedIndices = boxes.indices.sorted {
            boxes[$0].score > boxes[$1].score
        }
        var selected: [Box] = []
        var active = [Bool](repeating: true, count: boxes.count)
        var numActive = active.count
        
        outer : for i in 0..<boxes.count {
            if active[i] {
                let boxA = boxes[sortedIndices[i]]
                selected.append(boxA)
                
                if selected.count >= limit { break }
                
                for j in i+1..<boxes.count {
                    if active[j] {
                        let boxB = boxes[sortedIndices[j]]
                        let rectA = CGRect(x: Double(boxA.x), y: Double(boxA.y), width: Double(boxA.w), height: Double(boxA.h))
                        let rectB = CGRect(x: Double(boxB.x), y: Double(boxB.y), width: Double(boxB.w), height: Double(boxB.h))
                        
                        if IOU(a: rectA, b: rectB) > threshold {
                            active[j] = false
                            numActive -= 1
                            if numActive <= 0 { break outer }
                        }
                    }
                }
            }
        }
        return selected
    }
    
    public static func IOU(a: CGRect, b: CGRect) -> Float {
        let areaA = a.width * a.height
        if areaA <= 0 { return 0 }
        
        let areaB = b.width * b.height
        if areaB <= 0 { return 0 }
        
        let intersectionMinX = max(a.minX, b.minX)
        let intersectionMinY = max(a.minY, b.minY)
        let intersectionMaxX = min(a.maxX, b.maxX)
        let intersectionMaxY = min(a.maxY, b.maxY)
        let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
            max(intersectionMaxX - intersectionMinX, 0)
        return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }
}
