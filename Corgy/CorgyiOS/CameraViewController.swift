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
    override func viewDidLoad() {
        super.viewDidLoad()
        if !imageSource.setup() {
            print("Error on setup image source.")
            return
        }
        imageSource.delegate = self
        cameraView.layer.addSublayer(imageSource.previewLayer!)
        imageSource.previewLayer?.frame = cameraView.bounds
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
    
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
var count = 0
extension CameraViewController: ImageSourceDelegate {
    func captured(_ variable: Variable) {
        print("\(count): ", terminator: "")
        print(yolo.predict(input: variable))
    }
}
