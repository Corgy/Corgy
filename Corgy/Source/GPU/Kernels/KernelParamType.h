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
    int poolSizeX;
    int poolSizeY;
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

#endif /* KernelParamType_h */
