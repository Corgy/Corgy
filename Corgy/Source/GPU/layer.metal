//
//  layer.metal
//  Corgy
//
//  Created by buqian zheng on 4/21/18.
//

#include <metal_stdlib>
using namespace metal;

struct TEMP_PARAM {
    
};

kernel void ReLU(device float *input [[ buffer(0) ]],
                 uint id [[thread_position_in_grid]]
                 ) {
    if (input[id] < 0) input[id] = 0;
}

kernel void Pool2DAVG(const device float *input [[ buffer(0) ]],
                      device float *output [[ buffer(1) ]],
                      uint id [[thread_position_in_grid]]
                      ) {
    output[id] = -input[id];
}

kernel void Pool2DMAX(const device float *input [[ buffer(0) ]],
                      device float *output [[ buffer(1) ]],
                      uint id [[thread_position_in_grid]]
                      ) {
    output[id] = -input[id];
}

// Example of a kernel that reads an input and writes to an output
kernel void testNeg2(const device float *input [[ buffer(0) ]],
                     device float *output [[ buffer(1) ]],
                     uint2 gsize [[threadgroups_per_grid]],       // number of thread groups
                     uint2 gid [[threadgroup_position_in_grid]], // coordinate of current thread group
                     uint2 tsize [[threads_per_threadgroup]],     // number of thread in one thread group
                     uint2 tid [[thread_position_in_threadgroup]]// coordinate of current thread
                     ) {
    uint id = (tsize.x * tsize.y) * (gsize.x * gid.y + gid.x) + tsize.x * tid.y + tid.x;
    output[id] = -input[id];
}

// Example of an inplace kernel
kernel void testNeg(device float *input [[ buffer(0) ]],
                    uint2 gsize [[threadgroups_per_grid]],       // number of thread groups
                    uint2 gid [[threadgroup_position_in_grid]], // coordinate of current thread group
                    uint2 tsize [[threads_per_threadgroup]],     // number of thread in one thread group
                    uint2 tid [[thread_position_in_threadgroup]]// coordinate of current thread
                    ) {
    uint id = (tsize.x * tsize.y) * (gsize.x * gid.y + gid.x) + tsize.x * tid.y + tid.x;
    input[id] = -input[id];
}

kernel void imageToMatrix(const device float *input [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          uint2 gsize [[threadgroup_per_gridd]],
                          uint2 gid[[threadgroup_position_in_grid]],
                          uint2 tsize[[threads_per_threadgroup]],
                          uint2 tid[[thread_position_in_threadgroup]]) {
    uint id = (tsize.x * tsize.y) * (gsize.x * gid.y + gid.x) + tsize.x * tid.y + tid.x;
    
    
}
