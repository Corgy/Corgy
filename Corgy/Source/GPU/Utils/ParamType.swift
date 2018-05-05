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
    var param: VariableParam {
        if shape.count == 4 {
            return VariableParam(batch: Int32(self.shape[0]),
                                 channel: Int32(self.shape[1]),
                                 width: Int32(self.shape[3]),
                                 height: Int32(self.shape[2]))
        } else if shape.count == 3 {
            return VariableParam(batch: 1,
                                 channel: Int32(self.shape[0]),
                                 width: Int32(self.shape[2]),
                                 height: Int32(self.shape[1]))
        } else if shape.count == 2 {
            return VariableParam(batch: 1,
                                 channel: 1,
                                 width: Int32(self.shape[1]),
                                 height: Int32(self.shape[0]))
        } else if shape.count == 1 {
            return VariableParam(batch: 1,
                                 channel: 1,
                                 width: Int32(self.shape[0]),
                                 height: 1)
        } else {
            // TODO: handle this
            fatalError()
        }
    }
}

struct PoolParam: LayerParam {
    let inputParam: VariableParam
    let outputParam: VariableParam
    let poolSizeW: Int32
    let poolSizeH: Int32
    let strideW: Int32
    let strideH: Int32
    let padW: Int32
    let padH: Int32
    // TODO: stride, dilation, padding to be supported
    init(inputParam: VariableParam, outputParam: VariableParam, poolSizeW: Int, poolSizeH: Int, strideW: Int, strideH: Int, padW: Int, padH: Int) {
        self.inputParam = inputParam
        self.outputParam = outputParam
        self.poolSizeW = Int32(poolSizeW)
        self.poolSizeH = Int32(poolSizeH)
        self.strideW = Int32(strideW)
        self.strideH = Int32(strideH)
        self.padW = Int32(padW)
        self.padH = Int32(padH)
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
    let negativeScope: Float32
    
    init(inputParam: VariableParam, negativeScope: Float) {
        self.inputParam = inputParam
        self.negativeScope = Float32(negativeScope)
    }
}

struct ImageToMatParam: LayerParam {
    let inputParam: VariableParam
    let outputParam: VariableParam
    let kernelSize: Int32
    let kernelSizeSquared: Int32
    
    let kernelPerRow: Int32
    let kernelPerCol: Int32
    let padding: Int32
    /// - parameter inputParam: must be of 3 dimension
    /// - parameter outputParam: must be of 2 dimension
    init(inputParam: VariableParam, outputParam: VariableParam, kernelSize: Int, padding: Int) {
        self.inputParam = inputParam
        self.outputParam = outputParam
        self.kernelSize = Int32(kernelSize)
        self.kernelSizeSquared = self.kernelSize * self.kernelSize
        self.kernelPerRow = inputParam.width - Int32(kernelSize) + 1
        self.kernelPerCol = inputParam.height - Int32(kernelSize) + 1
        self.padding = Int32(padding)
    }
}

struct WeightToMatParam: LayerParam {
    let inputParam: VariableParam
    let outputParam: VariableParam
}

/// convert the result of big matrix multiply and bias to original multi-channel Variable
struct MatToVariableWithBias: LayerParam {
    let inputParam: VariableParam
    let biasParam: VariableParam
    let outputParam: VariableParam
}
/// convert the result of big matrix multiply to original multi-channel Variable
struct MatToVariableWithoutBias: LayerParam {
    let inputParam: VariableParam
    let outputParam: VariableParam
}
