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

public func async_main(_ block: @escaping () -> Void) {
    DispatchQueue.main.async {
        block()
    }
}
