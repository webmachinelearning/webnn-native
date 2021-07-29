# SqueezeNet Example

This example showcases the SqueezeNet-based image classification by WebNN API.

This example leverages the network topology of [SqueezeNet 1.0](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz) from TFLite models with "nhwc" layout and [SqueezeNet 1.1](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) from ONNX models with "nchw" layout. It loads an image as the input by [stb](https://github.com/nothings/stb) library, and loads squeezenet1.0_nhwc weights/biases from .npy [files](https://github.com/webmachinelearning/test-data/tree/main/models/squeezenet1.0_nhwc/weights) or squeezenet1.1_nchw weights/biases from .npy [files](https://github.com/webmachinelearning/test-data/tree/main/models/squeezenet1.1_nchw/weights) by [cnpy](https://github.com/rogersce/cnpy) library.

## Usage

```sh
> out/Release/SqueezeNet -h

Example Options:
    -h                      Print this message.
    -i "<path>"             Required. Path to an image.
    -m "<path>"             Required. Path to the .npy files with trained weights/biases.
    -l "<layout>"           Optional. Specify the layout: "nchw" or "nhwc". The default value is "nchw".
    -n "<integer>"          Optional. Number of iterations. The default value is 1, and should not be less than 1.
    -d "<device>"           Optional. Specify a target device: "cpu" or "gpu" or "default" to infer on. The default value is "default".
```

## Example Output

```sh
> out/Release/SqueezeNet -i examples/images/test.jpg -l nchw -m node/third_party/webnn-polyfill/test-data/models/squeezenet1.1_nchw/weights/
Info: Compilation Time: 42.1368 ms
Info: Execution Time: 2.45855 ms

Prediction Result:
#   Probability   Label
0   99.23%        lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
1   0.31%         polecat, fitch, foulmart, foumart, Mustela putorius
2   0.26%         weasel

Info: Done.
```

```sh
> out/Release/SqueezeNet -i examples/images/test.jpg -l nhwc -m node/third_party/webnn-polyfill/test-data/models/squeezenet1.0_nhwc/weights/
Info: Compilation Time: 58.464 ms
Info: Execution Time: 4.79983 ms

Prediction Result:
#   Probability   Label
0   94.60%        lesser panda
1   1.69%         teddy
2   1.19%         giant panda

Info: Done.
```
