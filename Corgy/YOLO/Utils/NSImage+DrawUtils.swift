//
//  NSImage+DrawUtils.swift
//  CorgyMac
//
//  Created by HOPE on 5/7/18.
//

import Cocoa
import Foundation
import Corgy
import QuartzCore

extension NSImage {
    public func drawRect(_ rect: CGRect) -> NSImage? {
        self.lockFocus()
        let rectangle = NSBezierPath(rect: rect)
        
        NSColor.red.set()
        rectangle.lineWidth = 5
        rectangle.stroke()
        
        self.unlockFocus()
        return self
    }
    
    public func writePNG(toURL url: URL) {
        guard let data = tiffRepresentation,
            let rep = NSBitmapImageRep(data: data),
            let imgData = rep.representation(using: .png, properties: [.compressionFactor : NSNumber(floatLiteral: 1.0)]) else {
                
                Swift.print("\(self.self) Error Function '\(#function)' Line: \(#line) No tiff rep found for image writing to \(url)")
                return
        }
        
        do {
            try imgData.write(to: url)
        }catch let error {
            Swift.print("\(self.self) Error Function '\(#function)' Line: \(#line) \(error.localizedDescription)")
        }
    }
}
