//
//  CorgyResource.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal

extension Corgy {
    class CorgyResource {
        let device: MTLDevice!
        let library: MTLLibrary!
        let commandQueue: MTLCommandQueue!
        init() {
            device = MTLCreateSystemDefaultDevice()
            let frameworkBundle = Bundle(for: NeuralNetwork.self)
            guard let defaultLibrary = try? device.makeDefaultLibrary(bundle: frameworkBundle) else {
                fatalError("Could not load default library from specified bundle")
            }
            library = defaultLibrary
            commandQueue = device.makeCommandQueue()
        }
    }
}
