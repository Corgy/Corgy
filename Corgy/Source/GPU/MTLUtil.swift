//
//  MTLUtil.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Metal

let THREAD_PER_GROUP = 128

// TODO: temperal placeholder struct, needs refractor
struct TEMP_PARAM {
    
}

/// submit work to GPU work queue of a neural network and wait until finish.
///
/// - parameter input: Variables in input will be set to buffer of encoder
///             in order(from 1). Last one will be used as output.


