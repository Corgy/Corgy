//
//  CameraViewController.swift
//  CorgyiOS
//
//  Created by buqian zheng on 5/7/18.
//

import UIKit
import AVFoundation
import Corgy

class CameraViewController: UIViewController {
    
    @IBOutlet weak var cameraView: UIImageView!
    let imageSource = ImageSource()
    let yolo: YOLO = YOLO.sharedInstance
    var colors: [UIColor] = []
    let maxBoxes: Int = 5
    var boundingBoxes: [BoundingBox] = []
    override func viewDidLoad() {
        super.viewDidLoad()
        if !imageSource.setup() {
            print("Error on setup image source.")
            return
        }
        imageSource.delegate = self
        cameraView.layer.addSublayer(imageSource.previewLayer!)
        imageSource.previewLayer?.frame = cameraView.bounds
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
        for _ in 0..<maxBoxes {
            let box = BoundingBox()
            box.addToLayer(cameraView.layer)
            boundingBoxes.append(box)
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    override func viewWillDisappear(_ animated: Bool) {
        if imageSource.isRunning {
            imageSource.stop()
        }
    }
    
    @IBAction func switchTapped(_ sender: UIButton) {
        if imageSource.isRunning {
            imageSource.stop()
            sender.titleLabel?.text = "Tap to Stop!!!"
        } else {
            imageSource.start()
            sender.titleLabel?.text = "Tap to Start!!!"
        }
    }

}

extension CameraViewController: ImageSourceDelegate {
    func captured(_ variable: Variable) {
        let boxes = yolo.predict(input: variable)
        DispatchQueue.main.async {
            for i in 0..<self.boundingBoxes.count {
                if i >= boxes.count {
                    self.boundingBoxes[i].hide()
                    continue
                }
                let box = boxes[i]
                let label = String(format: "%@ %.1f", classes[box.klassIndex], box.score * 100)
                let color = self.colors[box.klassIndex]
                self.boundingBoxes[i].show(frame: box.rect(in: self.cameraView.bounds), label: label, color: color)
            }
        }
    }
}
