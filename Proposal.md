# Name to be determined - A Neural Network Framework Parallelized by Metal 2

By [Buqian Zheng](https://github.com/zhengbuqian)(buqianz) and [Yongkang Huang](https://github.com/MrDwZ)(yongkan1)

We are going to implement a framework that solving the folowing problems:
1. Simplifying the process of migrating general neural network model obtained from major open souerce deep learning frameworks (e.g. PyTorch, Keras) to iOS devices.
2. Leveraging the power of the computing resources of iOS devices by utilizing the tools including Metal Performance Shader and Accelerate to run the imported model.
3. Providing mobile-friendly features (e.g data compression, energy efficient) in the best effort.

## Background

Metal 2 framework is an interface provided by Apple that provides near-direct access to the graphics processing unit (GPU) on iPhone/iPad and Mac. Besides just graphics, Metal 2 incorporated a bunch of libraries that provide excellent parallized support to the necessary linear algebra operations and signal processing functions capable of running in various kind of devices. These libraries made it's possible for us to build a well implemented GPU-accelerated deep learning models on the iOS devices<sup id="a1">[1](#f1)</sup> 

Surely the _inference_ stage of a trained neural network is very computation intensive, especially for those models that has many layers dealing with high resolution images.  It's worth note that there is a **huge amount of matrix computation** in this process which can be greatly parallelized. By taking advantage of GPU accelarate ability provided by Metal 2, we can speed up this process greatly.

![](img/alexnet.png)
A sample Neural Network(AlexNet) with multiple layers<sup id="a2">[2](#f2)</sup>

## The Challenge

Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
Describe constraints: What are the properties of the system that make mapping the workload to it challenging?

**What are the dependencies**
The neural net work get data dependencies between different layers. Usually, the deep neural network that widely accepted today consists of consideralbly large number of layers and each of them has different kiind of workloads. For instance, for the convolutional neural network(CNN), the major building block could be the convolution layer, which's main work load id to executes the matrix dot between the convolution kernel and the image in the layer; the pooling layer which perfomes the max value or the average value extraction of a given region of image, and the fully connected layer, which performes tremendous amount of maxtrix multiplications and activate function computations.

**What are its memory access characteristics?**
The memory access characteristics of the neural network is decided by how specifically we present the data model and the stratagies to dynamically load the weight from permanant storage. Generally speaking, the convolution and pooling operation requires a square region of the image presentaion, if we store the image in a row major manner, for large images the locallity is not ideal due to that the different row of image is not stored in concecutive region of memories.

In addition to the image representation storage, the storage of the weight connect each layer is also crucial to the performance. Basically, we want the weight represention stored in the pemanant storage to be compact, especially for the mobile devices that is sensitive to the storage capability constrains. Also, when doing a batch of calculations, we want the expanded weights in memory not only to have the good locality but also friendly to the cahce coherence policy of both the X86 and ARM platform to gain expected outstanding  performance.

**Is there a high communication to computation ratio?**
The computation is actually based on a shared memory model that we do not require mutiple devices to execute the trained model, so the computation intensity should be high and the communication to computation ratio should be relativly low. The exact complexity will be calculated after we have done the implementation.

**What are the properties of the system that make mapping the workload to it challenging?**
Actually, due to the various of the Apple products that across both x86 and ARM structures and the difference of architectures as well as the supported techologies and number of processing units of the different generations of the same kind of product. It could be tough to have a general rules to fine tune the performance that is commonly accepted by all the platforms.

## Resources
This project will be implemented from scratch. We'll be using iPhone, iPad and Macbook pro as the platform on which the framework will run. However we will perhaps refrence source code of PyTorch<sup id="a3">[3](#f3)</sup> and Caffe<sup id="a4">[4](#f4)</sup> framework about the implementation of a Neural Framework.

## Goals and Deliverables

## Platform Choice

We will choose iOS and macOS as hosting OS. Metal 2 framework and Accelarate framework will be used to implementation our framework. We will choose Swift as our primary language. The specific related hardware platform currently decided incorporating:
ARM:
* iPhone 6s
  * Processor: Apple A9 (2 Cores CPU with ARMv8-A Architecture and 6 cores GPU)
  * 2 GB LPDDR4 RAM
* iPad Air

X86:
* MacBook Pro
* MacBook Air
  * Intel Core i5-4250U CPU (2 cores with hyper-threading)
  * 4GB DDR3 RAM

## Schedule
a
|Time|Task|Status|
|---|---|---|
|Project checkpoint I|Implement Model converter and the first version of GPU based implementation ||
|Project checkpoint II|Implement fine tuned model implementation and basic benchmark||
|Before Due|Implement a demo object detection application and the thorough benchmarks accrosing multiple devices ||

### References

<b id="f1">1</b> https://developer.apple.com/metal/ [↩](#a1)

<b id="f2">2</b> https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf [↩](#a2)

<b id="f3">3</b> http://pytorch.org [↩](#a3)

<b id="f4">4</b> https://github.com/BVLC/caffe [↩](#a4)


