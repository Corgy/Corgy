//
//  Miscellaneous.swift
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

import Foundation
import Metal
extension Corgy {
    static func getGroupNum(_ eleNum: Int, _ eleNumPerGroup: Int) -> Int {
        return  (eleNum + eleNumPerGroup - 1) / eleNumPerGroup
    }
    
    public static func Neg() -> Layer {
        return { ( _ input) in
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.size), 1, 1)
            let threadGroups = MTLSizeMake(getGroupNum(input.size, THREAD_PER_GROUP), 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            submitWork(name: "testNeg", in: input, param: param)
            return input
        }
    }
    
    
    public static func Neg2() -> Layer {
        return { ( _ input) in
            let threadsPerThreadGroup = MTLSizeMake(min(THREAD_PER_GROUP, input.size), 1, 1)
            let threadGroups = MTLSizeMake(getGroupNum(input.size, THREAD_PER_GROUP), 1, 1)
            
            let param = WorkParams(threadGroups: threadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
            
            let output = Variable(input.shape)
            submitWork(name: "testNeg2", in: input, output, param: param)
            return output
        }
    }
}


