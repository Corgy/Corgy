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
    
    /// create a (1, 3, height, width) Variable out of an image
    /// - parameter size: if provided, image will be resized to size.width and size.height
    public static func of(image: Image, to size: (Int, Int)? = nil) -> Variable {
        let cgImage = cgImageOf(image: image)
        return Variable.of(cgImage: cgImage, to: size)
    }
    public static func of(cgImage: CGImage, to size: (Int, Int)? = nil) -> Variable {
        let (width, height) = size ?? (cgImage.width, cgImage.height)
        let bitsPerComponent = cgImage.bitsPerComponent
        let bytesPerRow = cgImage.bytesPerRow
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: height * width * 4)
        let contexRef = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        contexRef!.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let v = Variable(1, 3, height, width)
        let channelSize = height * width
        for i in 0..<(pixels.count/4) {
            // ith pixel
            v.value[i] = DataType(pixels[i * 4]) / 255.0
            v.value[channelSize + i] = DataType(pixels[i * 4 + 1]) / 255.0
            v.value[channelSize * 2 + i] = DataType(pixels[i * 4 + 2]) / 255.0
        }
        return v
    }
    /// create a (1, 1, height, width) Variable out of an grayscale image
    /// - parameter image: a grayscale image. it must be a grayscale. we didn't add code
    ///                   to convert a colored image to a grayscale image.
    /// - parameter size: if provided, image will be resized to size.width and size.height
    public static func of(grayScaleImage image: Image, to size: (Int, Int)? = nil) -> Variable {
        let cgImage = cgImageOf(image: image)
        
        let (width, height) = size ?? (cgImage.width, cgImage.height)
        let bitsPerComponent = cgImage.bitsPerComponent
        let bytesPerRow = cgImage.bytesPerRow
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixelValues = [UInt8](repeating: 0, count: height * width)
        let v = Variable(1, 1, height, width)
        let contextRef = CGContext(data: &pixelValues, width: width, height: height, bitsPerComponent: bitsPerComponent,
                                   bytesPerRow: bytesPerRow / 2, space: colorSpace, bitmapInfo: 0)
        contextRef!.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        for (i, p) in pixelValues.enumerated() {
            v.value[i] = 1 - DataType(p) / 255
        }
        return v
    }
}

