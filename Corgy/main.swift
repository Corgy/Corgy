import Corgy

print("Hello World!")

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