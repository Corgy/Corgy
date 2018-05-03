//
//  Async.swift
//  Corgy
//
//  Created by buqian zheng on 5/2/18.
//

import Foundation


public func async(_ block: @escaping () -> Void) {
    DispatchQueue.global().async {
        block()
    }
}
