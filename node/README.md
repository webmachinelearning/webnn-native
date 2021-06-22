# WebNN-native Binding for Node.js

*WebNN-native Binding for Node.js is a Node.js [C++ addon](https://nodejs.org/api/addons.html). The implementation is based on WebNN-native C++ API and exposes [WebNN](https://webmachinelearning.github.io/webnn/) JavaScript API.


## Prerequisites

Install [Node.js](https://nodejs.org/).

For Windows, install [Visual Studio 2019](https://visualstudio.microsoft.com/vs/).

For Linux, install `build-essential` package.

**Verified configurations:**
  * Node.js 14 LTS
  * Windows 10
  * Ubuntu Linux 16.04

## Build WebNN-native

You need to build WebNN native as a C/C++ library by following [WebNN-native README](https://github.com/webmachinelearning/webnn-native#readme), please make sure the end2end tests can work as expected.

**Note:** In case you have multiple python installations, you might want to use the --script-executable gn flag to instruct gn to use the python 2.x installation.

## Build and Run

### Install Node.js modules

Install node-gyp and other Node.js modules that are required to build the Node.js addon. The `webnn_native_lib_path` must be configured so that WebNN native header files and libraries can be found. The path must be a relative path to the `node` folder. For example, execute the following command:

```shell script
> npm install --webnn_native_lib_path="../out/Release"
```

### Build

The Node.js addon can also be built after install. For example, execute the following command:

```shell script
> npm run build --webnn_native_lib_path="../out/Release"
```

## Test

```shell script
> npm test
```

## Example

 * [LeNet in Electron.js](examples/electron/lenet/README.md)
 * [LeNet in Node.js](examples/console/lenet/README.md)
 * [ImageClassification in Electron.js](examples/electron/image_classification/README.md)
