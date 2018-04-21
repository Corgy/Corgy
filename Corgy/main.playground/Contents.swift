import Cocoa
import Corgy

var str = "Hello, playground"

let kernelWeight = Variable(1, 2, 2, 2)
for i in 0..<kernelWeight.value.count {
    kernelWeight.value[i] = Float(i)
}

let conv2d = CPU.Conv2D(inChannels: 1, outChannels: 2, kernelSize: 2, weight: kernelWeight)

let input = Variable(1, 1, 4, 4)
for i in 0..<input.value.count {
    input.value[i] = Float(i)
}
print(conv2d(input))

