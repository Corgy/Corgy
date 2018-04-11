# Name to be determined - a neural network framework parallelized by Metal 2

By [Buqian Zheng](https://github.com/zhengbuqian)(buqianz) and [Yongkang Huang](https://github.com/MrDwZ)(yongkan1)

We are going to implement a neural network parallelized by Metal 2 so we can run popular NN models on iOS and macOS devices very fast.

## Background

Metal 2 framework is an interface provided by Apple that provides near-direct access to the graphics processing unit (GPU) on iPhone/iPad and Mac. Besides just graphics, Metal 2 provides deep support for GPU-accelerated machine learning<sup id="a1">[1](#f1)</sup> which makes it suitable to implement a parallelized neural network.

Surely the _evaluation_ of a trained neural network is very computation intensive, especially for those models that has many layers dealing with high resolution images.  It's worth note that there is a **huge amount of matrix computation** in this process which can be greatly parallelized. By taking advantage of GPU accelarate ability provided by Metal 2, we can speed up this process greatly.

![](img/alexnet.png)
A sample Neural Network(AlexNet) with multiple layers<sup id="a2">[2](#f2)</sup>

## The Challenge, TODO

Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?

Describe constraints: What are the properties of the system that make mapping the workload to it challenging?

## Resources

This project will be implemented from scratch. We'll be using iPhone, iPad and Macbook pro as the platform on which the framework will run. However we will perhaps refrence source code of PyTorch<sup id="a3">[3](#f3)</sup> and Caffe<sup id="a4">[4](#f4)</sup> framework about the implementation of a Neural Framework.

## Goals and Deliverables

### Plan to achieve

Our goal in this project is to implement a parallel Neural Network framework that runs on Apple's devices with iOS and macOS. We will explore the parallel ability of the GPU on those devices. Due to the limitation of power and computation resources of mobile devices, this framework will mainly focus on the evaluation of trained neural network.

With this framework, programmers can:

1. Simplify the process of migrating general neural network model trained by mainstream open-source deeplearning frameworks such as PyTorch and Keras to iOS and macOS devices.
2. Leverage the power of the GPU computing resources of iOS and macOS devices by utilizing built-in frameworks such as Metal 2.
3. Make the evaluation of neural network mobile-friendly, such as compress model to reduce application package size, or try to use less energy so the battery can last longer.

This framework will add support for common layers such as `FullyConnected`, `ReLU`, `Dense`, `MaxPooling`, `AveragePooling`,`Convolution`, `SoftMax` layers.

We will make a demo application that runs on iPhone as the host app of a popular trained neural network model Tiny YOLO to do object recognition. We hope that we can process 5 to 10 images captured by camera on iPhone which is almost realtime.

### Hope to Achieve

If we have time, we'll also plan to use Accelarate framework, CPU SIMD API on iOS/macOS, to implement a CPU version. Then we will do benchmark comparing the performance of the GPU version and the CPU version.

## Platform Choice

We will choose iOS and macOS as hosting OS. Metal 2 framework and Accelarate framework will be used to implementation our framework. We will choose Swift as our primary language.

## Schedule

Checkpoint 1: April 18, Checkpoint 1: April 27, Due: May 7

|Time|Task|Status|
|---|---|---|
||||

### References

<b id="f1">1</b> https://developer.apple.com/metal/ [↩](#a1)

<b id="f2">2</b> https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf [↩](#a2)

<b id="f3">3</b> http://pytorch.org [↩](#a3)

<b id="f4">4</b> https://github.com/BVLC/caffe [↩](#a4)
