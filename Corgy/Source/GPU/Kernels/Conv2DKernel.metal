//
//  Conv2DKernel.metal
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

#include <metal_stdlib>
using namespace metal;
#include "KernelParamType.h"
//
//kernel void imageToMatrix(const device float *input [[buffer(0)]],
//                          device float *output [[buffer(1)]],
//                          uint2 gsize [[threadgroup_per_gridd]],
//                          uint2 gid[[threadgroup_position_in_grid]],
//                          uint2 tsize[[threads_per_threadgroup]],
//                          uint2 tid[[thread_position_in_threadgroup]]) {
//    uint id = (tsize.x * tsize.y) * (gsize.x * gid.y + gid.x) + tsize.x * tid.y + tid.x;
//}
