//
//  YOLO.swift
//  CorgyiOS
//
//  Created by buqian zheng on 5/7/18.
//

import Foundation
import Corgy

class YOLO {
    static let sharedInstance = YOLO()
    
    private let network: NeuralNetwork?
    private let semaphore: DispatchSemaphore
    private init() {
        network = ModelImporter.importYolo(computeOn: .GPU)
        semaphore = DispatchSemaphore(value: 1)
    }
    
    /// this method will block until predict is finished.
    /// try always to all in background thread
    func predict(input: Variable) -> [ModelImporter.Box] {
        var output: Variable?
        timing("YOLO") {
            semaphore.wait()
            output = network!.forward(input)
            semaphore.signal()
        }
        let boxes = ModelImporter.getResult(input: output!)
        return ModelImporter.nonMaxSuppression(boxes: boxes, limit: 3, threshold: 0.01)
    }
}
