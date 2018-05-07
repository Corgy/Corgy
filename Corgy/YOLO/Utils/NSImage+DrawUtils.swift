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
    @discardableResult public func drawRect(_ rect: CGRect) -> NSImage? {
        self.lockFocus()
        let rectangle = NSBezierPath(rect: rect)
        
        NSColor.red.set()
        rectangle.lineWidth = 5
        rectangle.stroke()
        
        self.unlockFocus()
        return self
    }
    
    public func drawBox(_ box: ModelImporter.Box) -> NSImage? {
        let text = ModelImporter.voc_labels[box.klassIndex] + " " + String(box.score)
        let im:NSImage = drawRect(CGRect(x: Double(box.x), y: Double(box.y), width: Double(box.w), height: Double(box.h)))!
        
        let font = NSFont.boldSystemFont(ofSize: 18)
        let imageRect = CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height)
        
        let textRect = CGRect(x: Double(box.x), y: Double(box.y), width: 200.0, height: 20.0)
        let textStyle = NSMutableParagraphStyle.default.mutableCopy() as! NSMutableParagraphStyle
        let textFontAttributes = [
            NSAttributedStringKey.font: font,
            NSAttributedStringKey.foregroundColor: NSColor.white,
            NSAttributedStringKey.paragraphStyle: textStyle
        ]
        
        let rep:NSBitmapImageRep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(self.size.width), pixelsHigh: Int(self.size.height), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: NSColorSpaceName.calibratedRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        
        im.addRepresentation(rep)
        
        im.lockFocus()
        
        let nstext = NSString(string: text)
        im.draw(in: imageRect)
        nstext.draw(in: textRect, withAttributes: textFontAttributes)
        
        im.unlockFocus()
        
        return im
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
