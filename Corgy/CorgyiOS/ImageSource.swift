//
//  ImageSource.swift
//  CorgyiOS
//
//  Created by buqian zheng on 5/7/18.
//

import Foundation
import AVFoundation
import UIKit
import Corgy

protocol ImageSourceDelegate {
    func captured(_ variable: Variable)
}

class ImageSource: NSObject {
    
    var isRunning: Bool {
        return captureSession.isRunning
    }
    
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    
    let queue = DispatchQueue(label: "org.Corgy.camera-queue")
    
    var delegate: ImageSourceDelegate?
    
    let fps = 2
    var lastTime = CMTime()
    
    public override init() {
        super.init()
        let notification = NSNotification.Name(rawValue: CorgyApproachingLastLayer)
        NotificationCenter.default.addObserver(self, selector: #selector(ImageSource.didReceiveNotification), name: notification, object: nil)
    }
    
    var shouldPrepareNext = true
    let prepareLock: DispatchSemaphore = DispatchSemaphore(value: 1)
    @objc private func didReceiveNotification() {
        shouldPrepareNext = true
    }
    
    func start() {
        if !captureSession.isRunning {
            captureSession.startRunning()
            shouldPrepareNext = true
        }
    }
    func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
            shouldPrepareNext = false
        }
    }
    
    func setup() -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480
        
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            return false
        }
        
        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            return false
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspect
        previewLayer?.connection?.videoOrientation = .portrait
        
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        videoOutput.connection(with: AVMediaType.video)?.videoOrientation = .portrait
        captureSession.commitConfiguration()
        
        return true
    }
    
}

extension ImageSource {
    /// - parameter completion: completion handler will run on imageQueue,
    ///         switch back to original queue if necessary
    func resize(_ image: UIImage, to newSize: CGSize, completion: ((UIImage) -> ())? = nil) {
        DispatchQueue.global().async {
            UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
            image.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
            let img = UIGraphicsGetImageFromCurrentImageContext()!
            UIGraphicsEndImageContext()
            if let completion = completion {
                completion(img)
            }
        }
    }
}

extension ImageSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        if !shouldPrepareNext {
            return
        }
        prepareLock.wait()
        if !shouldPrepareNext {
            prepareLock.signal()
            return
        }
        shouldPrepareNext = false
        prepareLock.signal()
        
        if let delegate = delegate {
            DispatchQueue.global().async {
                let imageBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
                let ciimage : CIImage = CIImage(cvPixelBuffer: imageBuffer)
                let image : UIImage = UIImage.of(ciImage: ciimage)
                self.resize(image, to: CGSize(width: 416, height: 416)) { (image: UIImage) in
                    let variable = Variable.of(image: image, to: (416, 416))
                    delegate.captured(variable)
                }
            }
        }
    }
}

fileprivate extension UIImage {
    static func of(ciImage: CIImage) -> UIImage {
        let context: CIContext = CIContext.init(options: nil)
        let cgImage: CGImage = context.createCGImage(ciImage, from: ciImage.extent)!
        let image: UIImage = UIImage.init(cgImage: cgImage)
        return image
    }
}


