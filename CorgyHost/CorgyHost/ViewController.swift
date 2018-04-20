//
//  ViewController.swift
//  CorgyHost
//
//  Created by buqian zheng on 4/18/18.
//  Copyright © 2018 buqian zheng. All rights reserved.
//

import UIKit
import AVFoundation
import Corgy

class ViewController: UIViewController {
    
    var session: AVCaptureSession?
    var photoOutput: AVCapturePhotoOutput?
    var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    
    @IBOutlet weak var previewView: UIView!
    
    @IBAction func takePhoto(_ sender: Any) {
        let input = Variable(1, 1, 4, 4)
        
        for y in 0..<4 {
            for x in 0..<4 {
                input[0, 0, y, x] = Float(arc4random() % 100)
            }
        }
        print(input)
        let maxPool = CPU.pool2dGen(poolSize: (2, 2), poolType: .Max)
        let avgPool = CPU.pool2dGen(poolSize: (2, 2), poolType: .Average)
        print(maxPool(input))
        print(avgPool(input))
    }
    
    override func viewWillAppear(_ animated: Bool) {
//        super.viewWillAppear(animated)
//        session = AVCaptureSession()
//        session!.sessionPreset = AVCaptureSession.Preset.photo
//        let backCamera = AVCaptureDevice.default(for: AVMediaType.video)
//        var error: NSError?
//        var input: AVCaptureDeviceInput!
//        do {
//            input = try AVCaptureDeviceInput(device: backCamera!)
//        } catch let error1 as NSError {
//            error = error1
//            input = nil
//            print(error!.localizedDescription)
//        }
//        if error == nil && session!.canAddInput(input) {
//            session!.addInput(input)
//            photoOutput = AVCapturePhotoOutput()
//            if session!.canAddOutput(photoOutput!) {
//                session!.addOutput(photoOutput!)
//                videoPreviewLayer = AVCaptureVideoPreviewLayer(session: session!)
//                videoPreviewLayer!.videoGravity = AVLayerVideoGravity.resizeAspect
//                videoPreviewLayer!.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
//                previewView.layer.addSublayer(videoPreviewLayer!)
//                session!.startRunning()
//            }
//        }
    }
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
//        videoPreviewLayer!.frame = previewView.bounds
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()

    }


}

