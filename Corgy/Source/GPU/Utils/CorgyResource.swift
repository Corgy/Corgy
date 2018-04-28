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
            library = device.makeDefaultLibrary()
            commandQueue = device.makeCommandQueue()
        }
    }
}
