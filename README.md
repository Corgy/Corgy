# Corgy - A Neural Network Framework for macOS and iOS parallelized by Metal

By [Buqian Zheng](https://github.com/zhengbuqian)(buqianz) and [Yongkang Huang](https://github.com/MrDwZ)(yongkan1)

We implemented Corgy, a neural network framework in Swift and Metal. Corgy can be embedded into both macOS and iOS applications and be used to construct trained neural networks and evaluate them with ease. We achieved 60x to 150x speedup on different devices with different GPUs.

<div style="visibility: hidden;display: none;">
Run MNIST on 15' Macbook pro 1000 times: GPU takes ~9 seconds, CPU takes ~26 minutes.
</div>

Corgy is not designed for network training due to limited computation power and battery life on iOS devices.

## Background

Metal 2 framework is an interface provided by Apple that provides near-direct access to the graphics processing unit (GPU) on iPhone/iPad and Mac. Besides graphics, Metal 2 incorporated a bunch of libraries that provide excellent paralyzed support to the necessary linear algebra operations and signal processing functions that capable of running in various kind of Apple devices. These libraries made it possible for us to build a well implemented GPU-accelerated deep learning models on the iOS devices based on the trained model given by other frameworks.<sup id="a1">[1](#f1)</sup> 

Generally speaking, the _inference_ stage of a trained neural network is very computation intensive, especially for those models that have a considerably large number of layers or applied in the scenarios that are necessary to process high-resolution images.  It's worth notice that there is a **tremendous amount of matrix computation(eg. convolution)** which is appropriate to apply parallelized operation to optimize the performance.

![](img/alexnet.png)
A sample Neural Network(AlexNet) with multiple layers<sup id="a2">[2](#f2)</sup>

### Challenge

## Approach

We are using Swift programming language and Metal 2 framework provided by Apple. Our target machines are all devices running macOS and iOS, such as iMac, Macbook, iPhone and iPad. After compilation our project will output 2 frameworks, 1 for macOS and 1 for iOS. 

The parallel abstraction of Metal 2 is very much alike CUDA: when dispatching compute pass to GPU, programmers will first write kernel functions that will be executed by each threads, then specify number of thread group(aka. block in CUDA) in grid and number of threads in each thread group, Metal will execute kernels over this grid. Inside each thread group there is smaller unit called SIMD group, means a bunch of threads that share same SIMD instructions. But under our implementation there is no need to consider this.

In our implementation, we mapped one element in the output tensor of current layer to one thread: each thread computes and updates exactly one element of the output, and the input will be readonly, so we don't need to worry about synchronization between threads.

Under this mapping, threads with continuous ids might read input data from different memory locations but will always write to continuous memory locations. So there won't be scatter operations when writing to memory.

临时写一点思路：
1. Change convolution to giant matrix multiplication
2. The design and implementation of Variable class
3. We tried to map one thread to several elements, from 2 to 16 elements per thread, the performance is almost the same, but adds much complexity to our project, so we discarded this approach.
4. At first we are using a Swift array as the underlying data structure of Variable class and this causes us a big problem. When submitting work to GPU, there are two methods to allocate a memory buffer that can be used by the GPU: `makeBuffer(bytes:)` and `makeBuffer(bytesNoCopy:)`. As their names indicate, the first method will allocate a new memory buffer and copy given data into that buffer, while the second methods will use the given data directly. It's obvious that we'll want to use the second method to avoid the memory copy overhead to achieve better performance, but it turns out the second method always returns `nil` meaning fail to create buffer and no error message is provided. So we temporarily stick with the first method. After several days, our framework was optimized to a phase this copy is taking a majority of time so we'll have to solve this. We found that the memory used to create buffer must be page aligned in memory in order to use the `makeBuffer(bytesNoCopy:)` method, but after researches and multiple tries, we realized that there is no way in Swift to create an array that is guaranteed to be page aligned. So we have to completely refractor Variable class to use pointer as underlying data structure. 

We didn't manage to finish the model compression part as mentioned in project proposal.

## Result

Overall our implementation is good, but not good enough. 

*All CPU versions mentioned below are single-threaded CPU code without SIMD optimization. Only a compiler optimization at level `-Ofast` is applied.*

Comparing with the naive CPU version implementation without parallelism, our GPU version is **~60x faster on a mobile device**(iPhone 6s with a 6 core PowerVR GT7600 GPU), **~160x faster on a laptop**(Macbook pro with integrated Intel Iris pro GPU with 48 Execution Units).

Experiment network attribute:

MNIST:

- input size: 1 * 28 * 28
- number of weights: ~0.1M
- number of convolution layers: 2
- number of fully connected layer: 1

YOLO:

- input size: 3 * 416 * 416
- number of weights: ~17M
- number of convolution layers: 9
- number of fully connected layer: 0

Measurement result: 

|iPhone 6s|MNIST|Tiny YOLO|
|---|---|---|
|CPU|1500ms|753s|
|GPU|0.025s|0.77s|
|speed up|~60x|~1000x|

|Macbook pro|MNIST|Tiny YOLO|
|---|---|---|
|CPU||729s... why so slow??...|
|GPU||0.028s|
|speed up|||

*Because it takes less time to run GPU version, so we take the average time of 100 runs. The running time of CPU version is the result of running once.*

TODO: how did the program differ when problem size changes?
Based on above benchmark we can see that as the problem size increases, 

Why we say our speedup is not good enough? Because when comparing with Apple official implementation of `MPSCNNConvolution`, we are only about one third as fast, which means there is still lots of optimization space. This comparison is based on an open source implementation of YOLO on iPhone using official `MPSCNNConvolution`. This implementation can recognize ~5 images per second while our implementation can only achieve ~1.3 images per second.

And due to a limited time, we weren't able to create a better baseline version and CPU parallelized version to do the benchmark.

What limited your speedup? 

1. An important performance limitation is in the convolution layer. In order to turn convolution computation into matrix multiplication, we need to change the structure of weight matrix. In some last convolution layer of Tiny YOLO, there is nearly 10 million weights and it takes a huge amount of time to rearrange the weight matrix even with GPU accelaration. This should be a memory bound limitation since when applying this rearrangement, each thread only moves one element from one place to another, there is no computation except for find out where this element should go.
**A possible solution: save pre-arrange weights in file since every time before use weights need to be rearranged.**
TODO: provide data here, how much time exactly this step is causing?

2. In some of the compute kernels, there are `if`s and `for`s which might cause divergence, leading to poor SIMD utilization.

Deeper Analysis: break down of execution time of different phase. What phase uses more time? Is there room to improve?

Overall, We believe that our choice of Metal framework to do Neural Network accelaration on iOS and macOS devices is sound, especially for iOS devices. With less cores, even with SIMD instructions, a well tuned CPU version is less likely to get similiar performance as GPU version.

## Division of work

Equal work are done by both team members.

## References

<b id="f1">1</b> https://developer.apple.com/metal/ [↩](#a1)

<b id="f2">2</b> https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf [↩](#a2)

<b id="f3">3</b> http://pytorch.org [↩](#a3)

<b id="f4">4</b> https://github.com/BVLC/caffe [↩](#a4)

<b id="f5">5</b> https://developer.apple.com/documentation/metal/compute_processing/about_threads_and_threadgroups [↩](#a5)
