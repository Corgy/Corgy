//
//  ParamType.swift
//  Corgy iOS
//
//  Created by buqian zheng on 4/26/18.
//

import Foundation

// to make makeBuffer easier
protocol LayerParam { }

struct VariableParam {
    let size: Int32
    let batch: Int32
    let sizePerBatch: Int32
    let channel: Int32
    let sizePerChannel: Int32
    let width: Int32
    let height: Int32
    init(batch: Int32, channel: Int32, width: Int32, height: Int32) {
        self.batch = batch
        self.channel = channel
        self.width = width
        self.height = height
        self.sizePerChannel = width * height
        self.sizePerBatch = self.sizePerChannel * channel
        self.size = self.sizePerBatch * batch
    }
}

extension Variable {
    func getParam() -> VariableParam {
        return VariableParam(batch: Int32(self.getShape()[0]),
                             channel: Int32(self.getShape()[1]),
                             width: Int32(self.getShape()[3]),
                             height: Int32(self.getShape()[2]))
    }
}

struct PoolParam: LayerParam {
    let inputParam: VariableParam
    let outputParam: VariableParam
    let poolSizeX: Int32
    let poolSizeY: Int32
    // TODO: stride, dilation, padding to be supported
    init(inputParam: VariableParam, outputParam: VariableParam, poolSizeX: Int, poolSizeY: Int) {
        self.inputParam = inputParam
        self.outputParam = outputParam
        self.poolSizeX = Int32(poolSizeX)
        self.poolSizeY = Int32(poolSizeY)
    }
}

struct DropoutParam: LayerParam {
    let inputParam: VariableParam
    let p: Float32
    let seed: UInt32
    init(inputParam: VariableParam, p: Double, seed: UInt32) {
        self.inputParam = inputParam
        self.p = Float32(p)
        self.seed = seed
    }
}

struct LeakyReLUParam: LayerParam {
    let inputParam: VariableParam
    let negativeScope: Float
    
    init(inputParam: VariableParam, negativeScope: Float) {
        self.inputParam = inputParam
        self.negativeScope = negativeScope
    }
}

