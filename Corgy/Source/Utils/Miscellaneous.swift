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
}


