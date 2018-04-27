import Cocoa

protocol C {
    
}

struct A:C {
    let a: Int32
}
struct B:C {
    let a: Int32
    let b: Int32
}

var a: C = A(a: 1)
var b: C = B(a: 1, b: 2)

print("\(MemoryLayout<A>.stride) \(MemoryLayout<B>.stride) \(MemoryLayout<Float32>.stride)")
//print("\(MemoryLayout<a>.stride) \(MemoryLayout<b>.stride)")

