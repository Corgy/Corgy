//
//  AppDelegate.swift
//  CorgyMac
//
//  Created by buqian zheng on 4/20/18.
//

import Cocoa
import Corgy

@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {
    
    @IBOutlet weak var window: NSWindow!
    
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        print("hello world")
        let input = Variable(1, 1, 4, 4)
        
        for y in 0..<4 {
            for x in 0..<4 {
                input[0, 0, y, x] = Float(arc4random() % 100)
            }
        }
        print(input)
        let maxPool = CPU.Pool2D(poolSize: (2, 2), poolType: .Max)
        let avgPool = CPU.Pool2D(poolSize: (2, 2), poolType: .Average)
        print(maxPool(input))
        print(avgPool(input))
        // Insert code here to initialize your application
    }
    
    func applicationWillTerminate(_ aNotification: Notification) {
        // Insert code here to tear down your application
    }
    
    
}

