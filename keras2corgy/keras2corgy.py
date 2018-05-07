# Most of this code is credited to [hollance](https://github.com/hollance).
# I did some minor changes to suit my need.

import os
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

# generated using YAD2K
model_path = "model_data/yolov2-tiny-voc.h5"

# Load the model that was exported by YAD2K.
model = load_model(model_path)
# model.summary()

model_nobn = Sequential()
model_nobn.add(Conv2D(16, (3, 3), padding="same", input_shape=(416, 416, 3)))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(32, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(64, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(128, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(256, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D())
model_nobn.add(Conv2D(512, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(MaxPooling2D(strides=(1, 1), padding="same"))
model_nobn.add(Conv2D(1024, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(Conv2D(1024, (3, 3), padding="same"))
model_nobn.add(LeakyReLU(alpha=0.1))
model_nobn.add(Conv2D(125, (1, 1), padding="same", activation='linear'))
model_nobn.summary()

def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    conv_weights = conv_layer.get_weights()[0]

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]
    
    epsilon = 1e-3
    new_weights = conv_weights * gamma / np.sqrt(variance + epsilon)
    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)
    return new_weights, new_bias

W_nobn = []
W_nobn.extend(fold_batch_norm(model.layers[1], model.layers[2]))
W_nobn.extend(fold_batch_norm(model.layers[5], model.layers[6]))
W_nobn.extend(fold_batch_norm(model.layers[9], model.layers[10]))
W_nobn.extend(fold_batch_norm(model.layers[13], model.layers[14]))
W_nobn.extend(fold_batch_norm(model.layers[17], model.layers[18]))
W_nobn.extend(fold_batch_norm(model.layers[21], model.layers[22]))
W_nobn.extend(fold_batch_norm(model.layers[25], model.layers[26]))
W_nobn.extend(fold_batch_norm(model.layers[28], model.layers[29]))
W_nobn.extend(model.layers[31].get_weights())
model_nobn.set_weights(W_nobn)

# Make a prediction using the original model and also using the model that
# has batch normalization removed, and check that the differences between
# the two predictions are small enough. They seem to be smaller than 1e-4,
# which is good enough for us, since we'll be using 16-bit floats anyway.

print("Comparing models...")

# order: [ batch, height, width, inputChannel ]
image_data = np.fromfile(open('model_data/imagedata.bin', 'r'), dtype=np.float32).reshape(1,416,416,3)
# order: [ batch, inputChannel, height, width ]
corgy_image_data = image_data.transpose(0,3,1,2)

features_nobn = model_nobn.predict(image_data)

# output in Corgy should be 
corgy_features_nobn = features_nobn.transpose(0,3,1,2)

max_error = 0
for i in range(features.shape[1]):
    for j in range(features.shape[2]):
        for k in range(features.shape[3]):
            diff = np.abs(features[0, i, j, k] - features_nobn[0, i, j, k])
            max_error = max(max_error, diff)
            if diff > 1e-4:
                print(i, j, k, ":", features[0, i, j, k], features_nobn[0, i, j, k], diff)

print("Largest error:", max_error)

# Convert the weights and biases to Metal format.

print("\nConverting parameters...")

dst_path = "Parameters"
W = model_nobn.get_weights()
for i, w in enumerate(W):
    j = i // 2 + 1
    print(w.shape)
    if i % 2 == 0:
        # weight order in keras is [ height, width, inputChannel, outputChannel ]
        w.transpose(3, 2, 0, 1).tofile(os.path.join(dst_path, "new_corgy_voc_conv%d_W.bin" % j))
    else:
        w.tofile(os.path.join(dst_path, "new_corgy_voc_conv%d_b.bin" % j))

print("Done!")