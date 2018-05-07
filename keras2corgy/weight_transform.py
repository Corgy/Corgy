import numpy as np

folder = 'yolo-voc'

prefix = "corgy_voc_conv"

for i in range(9):
    wfile = folder + '/' + prefix + str(i+1) + "_W.bin"
    shapefile = folder + '/' + prefix + str(i+1) + "_W_shape"
    shapestring = open(shapefile, 'r').readline()
    shape = [int(x) for x in shapestring.replace('\n', '').split(" ")]
    weight = np.fromfile(open(wfile, 'r'), dtype=np.float32)
    weight = np.reshape(weight, tuple(shape))

    outchannel = shape[0]
    inchannel = shape[1]
    kernelsize = shape[3]
    kernelsizeSq = kernelsize ** 2

    outwidth = outchannel
    outheight = inchannel * kernelsizeSq

    tranformed = np.zeros(shape=(outheight, outwidth), dtype=np.float32)
    for k in range(outheight):
        for j in range(outwidth):
            h = int(k % kernelsizeSq / kernelsize)
            w = int(k % kernelsizeSq % kernelsize)
            index = k // kernelsizeSq
            tranformed[k][j] = weight[j][index][h][w]

    tranformed.tofile(folder + '/t_' + prefix + str(i+1) + "_W.bin")

    shapestring = "%d %d" % (outheight, outwidth)
    originalshapestring = "%d %d %d %d" % tuple(shape)
    open(folder + '/t_' + prefix + str(i+1) + "_W_shape", 'w').write(shapestring)
    open(folder + '/t_' + prefix + str(i+1) + "_W_real_shape", 'w').write(originalshapestring)


    