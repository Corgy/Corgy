//
//  Timing.swift
//  Corgy iOS
//
//  Created by buqian zheng on 4/28/18.
//

import Foundation
import QuartzCore

@discardableResult
public func timing(_ message: String? = nil, _ block: () -> Void) -> CFTimeInterval {
    let startTime = CACurrentMediaTime()
    block()
    let interval = CACurrentMediaTime() - startTime
    if message != nil {
        print("\(message!) used \(interval * 1000) miliseconds.")
    }
    return interval
}

/// return CACurrentMediaTime() * 1000, which means in milliseconds
public func timing() -> CFTimeInterval {
    return CACurrentMediaTime() * 1000
}
