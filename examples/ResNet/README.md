# ResNet Example

This example showcases the ResNet-based image classification by WebNN API.

This example leverages the network topology of [ResNet101 V2](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz) from TFLite models with "nhwc" layout and [ResNet50 V2](https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v2-7.tar.gz) from ONNX models with "nchw" layout. It loads an image as the input by [stb](https://github.com/nothings/stb) library, and loads resnet101v2_nhwc weights/biases from .npy [files](https://github.com/webmachinelearning/test-data/tree/main/models/resnet101v2_nhwc/weights) or resnet50v2_nchw weights/biases from .npy [files](https://github.com/webmachinelearning/test-data/tree/main/models/resnet50v2_nchw/weights) by [cnpy](https://github.com/rogersce/cnpy) library.

## Usage

```sh
> out/Release/ResNet -h

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
> out/Release/ResNet -i examples/images/test.jpg -l nchw -m node/third_party/webnn-polyfill/test-data/models/resnet50v2_nchw/weights/
Info: Compilation Time: 212.473 ms
Info: Execution Time: 25.396 ms

Prediction Result:
#   Probability   Label
0   97.09%        lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
1   2.53%         polecat, fitch, foulmart, foumart, Mustela putorius
2   0.25%         weasel

Info: Done.
```

```sh
> out/Release/ResNet -i examples/images/test.jpg -l nhwc -m node/third_party/webnn-polyfill/test-data/models/resnet101v2_nhwc/weights/
Info: Compilation Time: 530.819 ms
Info: Execution Time: 65.3282 ms

Prediction Result:
#   Probability   Label
0   99.46%        lesser panda
1   0.13%         polecat
2   0.01%         Pomeranian

Info: Done.
```
