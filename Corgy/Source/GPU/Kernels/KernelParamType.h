//
//  KernelParamType.h
//  Corgy
//
//  Created by HOPE on 4/27/18.
//

#ifndef KernelParamType_h
#define KernelParamType_h

struct VariableParam {
    int size;
    int batch;
    int sizePerBatch;
    int channel;
    int sizePerChannel;
    int width;
    int height;
};

struct PoolParam {
    VariableParam inputParam;
    VariableParam outputParam;
    int poolSizeW;
    int poolSizeH;
    int strideW;
    int strideH;
    int padW;
    int padH;
};

struct LeakyReLUParam {
    VariableParam inputParam;
    float negativeScope;
};

struct DropoutParam {
    VariableParam inputParam;
    float p;
    unsigned seed;
};

struct ImageToMatParam {
    VariableParam inputParam;
    VariableParam outputParam;
    int kernelSize;
    int kernelSizeSquared;
    int kernelPerRow;
    int kernelPerCol;
};

struct WeightToMatParam {
    VariableParam inputParam;
    VariableParam outputParam;
};

struct MatToVariableWithBias {
    VariableParam inputParam;
    VariableParam biasParam;
    VariableParam outputParam;
};

struct MatToVariableWithoutBias {
    VariableParam inputParam;
    VariableParam outputParam;
};

#endif /* KernelParamType_h */
