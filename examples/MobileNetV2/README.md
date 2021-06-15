# MobileNet V2 Example

This example showcases the MobileNet V2-based image classification by WebNN API.

This example leverages the network topology of [MobileNet V2](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) from TFLite models with "nhwc" layout and [MobileNet V2](https://github.com/onnx/models/tree/master/vision/classification/mobilenet) from ONNX models with "nchw" layout. It loads an image as the input by [stb](https://github.com/nothings/stb) library, and loads weights/biases from .npy [files](/examples/MobileNetV2/weights) by [cnpy](https://github.com/rogersce/cnpy) library.

## Usage

```sh
> out/Release/MobileNetV2 [OPTION]

Options:
    -h                      Print this message.
    -i "<path>"             Required. Path to an image.
    -m "<path>"             Required. Path to the .npy files with trained weights/biases.
    -l "<layout>"           Optional. Specify the layout: "nchw" or "nhwc". The default value is "nchw".
    -n "<integer>"          Optional. Number of iterations. The default value is 1, and should not be less than 1.

```

## Example Output

```sh
> out/Release/MobileNetV2 -i examples/images/test.jpg -l nhwc -m node/third_party/webnn-polyfill/test-data/models/mobilenetv2_nhwc/weights/
Info: Compilation Time: 180.225 ms
Info: Execution Time: 7.667 ms

Prediction Result:
#   Probability   Label
0   93.83%        lesser panda
1   0.55%         polecat
2   0.18%         giant panda

Info: Done.
```

```sh
> out/Release/MobileNetV2 -i examples/images/test.jpg -l nchw -m node/third_party/webnn-polyfill/test-data/models/mobilenetv2_nchw/weights/
Info: Compilation Time: 140.49 ms
Info: Execution Time: 6.23081 ms

Prediction Result:
#   Probability   Label
0   99.54%        lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
1   0.23%         polecat, fitch, foulmart, foumart, Mustela putorius
2   0.10%         weasel

Info: Done.
```
