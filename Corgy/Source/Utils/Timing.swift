//
//  Timing.swift
//  Corgy iOS
//
//  Created by buqian zheng on 4/28/18.
//

import Foundation
import QuartzCore

func timing(_ message: String? = nil, _ block: () -> Void) -> CFTimeInterval {
    let startTime = CACurrentMediaTime()
    block()
    let interval = CACurrentMediaTime() - startTime
    if message != nil {
        print("\(message!) used \(interval) seconds.")
    }
    return interval
}
