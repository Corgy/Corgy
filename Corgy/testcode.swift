//
//  testcode.swift
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

import Foundation
import Corgy

func test () {
    let network = ModelImporter.loadMNISTCNN("MNIST_CNN")
    let image = Image(named: Image.Name("five"))!
    let input = Variable.of(grayScaleImage: image)
    let output = network.forward(input)
    print(output)
}