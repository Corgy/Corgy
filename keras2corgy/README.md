Code files in this folder are mostly for debug use.

[Image to keras binary](/img2kerasbinary.swift): given an image of size 416 * 416, transform it into keras format binary file. Input numpy array for a keras YOLO should be 1 * 416 * 416 * 3. Dimension order is [ batch, height, width, inputChannel ]. Corgy `Variable` image data has order [ batch, inputChannel, height, width]. You can see this difference in both img2kerasbinary.swift and keras2corgy.py.

[Transform weight from keras to Corgy](/keras2corgy.py): convert weights in a keras model to binary files that can be directly loaded by Corgy. Weight arrays in keras have order [ height, width, inputChannel, outputChannel ]. Weight arrays in Corgy have order [ outputChannel, inputChannel, height, width ]. So a simple `.transpose(3, 2, 0, 1)` on numpy array is sufficient.

Besides, keras2corgy.py also unfolds batch normalization. For reference please see this [post](http://machinethink.net/blog/object-detection-with-yolo/).

Usage:

1. download tiny yolo cfg and weights file from [official website](https://pjreddie.com/darknet/yolo/).
2. generate keras usable h5 file use [YAD2K](https://github.com/allanzelener/YAD2K).
3. in keras2corgy.py change model path to h5 file format.
4. run keras2corgy.py

Most code in keras2corgy.py is credited to [hollance](https://github.com/hollance), [original code](https://github.com/hollance/Forge/blob/master/Examples/YOLO/yolo2metal.py).
