//
//  Virable+Image.swift
//  Corgy
//
//  Created by buqian zheng on 4/23/18.
//

import Foundation

#if os(iOS)
    import UIKit
    public typealias Image = UIImage
#elseif os(OSX)
    import Cocoa
    public typealias Image = NSImage
#endif

fileprivate func cgImageOf(image: Image) -> CGImage {
    #if os(iOS)
        return image.cgImage!
    #elseif os(OSX)
        var rect = CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)!
    #endif
}

public extension Variable {
    public static func of(image: Image) -> Variable {
        fatalError()
    }
    public static func of(grayScaleImage image: Image) -> Variable {
        let cgImage = cgImageOf(image: image)
        let width = cgImage.width
        let height = cgImage.height
        let bitsPerComponent = cgImage.bitsPerComponent
        let bytesPerRow = cgImage.bytesPerRow
        let totalBytes = height * bytesPerRow
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelValues = [UInt8](repeating: 0, count: totalBytes)
        
        let contextRef = CGContext(data: &pixelValues, width: width, height: height, bitsPerComponent: bitsPerComponent,
                                   bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
        contextRef?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let v = Variable(1, 1, height, width)
        v.value = pixelValues.map { Float($0) / 255 }
        return v
    }
}
