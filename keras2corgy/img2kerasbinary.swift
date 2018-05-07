import Foundation
import Corgy

let image = Image(contentsOfFile: "dog.jpg")!

let input = Variable.of(image: image)

print(input.shape)

let output = Variable(416,416,3)

for i in 0..<416 {
    for j in 0..<416 {
        output[i, j, 0] = input[0, 0, i, j]
        output[i, j, 1] = input[0, 1, i, j]
        output[i, j, 2] = input[0, 2, i, j]
    }
}

let data = NSData(bytes: output.pointer, length: output.actualCount * 4)

try! data.write(toFile: "/Users/zbq/Desktop/shouldbecorrect/imagedata.bin")