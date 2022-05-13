// Copyright 2021 The WebNN-native Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "examples/SqueezeNet/SqueezeNet.h"

#include "common/Log.h"

SqueezeNet::SqueezeNet() : ExampleBase() {
}

bool SqueezeNet::ParseAndCheckExampleOptions(int argc, const char* argv[]) {
    if (!ExampleBase::ParseAndCheckExampleOptions(argc, argv)) {
        return false;
    }

    bool nchw = mLayout == "nchw" ? true : false;
    mLabelPath = nchw ? "examples/labels/labels1000.txt" : "examples/labels/labels1001.txt";
    mModelHeight = 224;
    mModelWidth = 224;
    mModelChannels = 3;
    mNormalization = nchw ? true : false;
    nchw ? mMean = {0.485, 0.456, 0.406} : mMean = {127.5, 127.5, 127.5};
    nchw ? mStd = {0.229, 0.224, 0.225} : mStd = {127.5, 127.5, 127.5};
    mOutputShape = nchw ? std::vector<int32_t>({1, 1000}) : std::vector<int32_t>({1, 1001});
    return true;
}

const wnn::Operand SqueezeNet::BuildConstantFromNpy(const wnn::GraphBuilder& builder,
                                                    const std::string& path) {
    const cnpy::NpyArray data = cnpy::npy_load(path);
    mConstants.push_back(data.data_holder);
    return utils::BuildConstant(builder, data.shape, data.data<float>(), data.num_bytes());
}

const wnn::Operand SqueezeNet::BuildConv(const wnn::GraphBuilder& builder,
                                         const wnn::Operand& input,
                                         const std::string& name,
                                         utils::Conv2dOptions* options) {
    std::string suffix = mLayout == "nchw" ? "_weight.npy" : "_kernel.npy";
    const std::string weightsPath = mWeightsPath + name + suffix;
    const wnn::Operand convWeights = BuildConstantFromNpy(builder, weightsPath);
    std::string biasSuffix = mLayout == "nchw" ? "_bias.npy" : "_Conv2D_bias.npy";
    const std::string biasPath = mWeightsPath + name + biasSuffix;
    const wnn::Operand convBias = BuildConstantFromNpy(builder, biasPath);
    if (!mFused) {
        const wnn::Conv2dOptions* conv2dOptions = options != nullptr ? options->AsPtr() : nullptr;
        std::vector<int32_t> newShape;
        mLayout == "nchw" ? newShape = {1, -1, 1, 1} : newShape = {1, 1, 1, -1};
        const wnn::Operand reshapedBias =
            builder.Reshape(convBias, newShape.data(), newShape.size());
        const wnn::Operand conv = builder.Conv2d(input, convWeights, conv2dOptions);
        const wnn::Operand add = builder.Add(conv, reshapedBias);
        return builder.Relu(add);
    } else {
        utils::Conv2dOptions fusedOptions;
        if (options != nullptr) {
            fusedOptions = *options;
        }
        fusedOptions.bias = convBias;
        fusedOptions.activation = builder.ReluOperator();
        return builder.Conv2d(input, convWeights, fusedOptions.AsPtr());
    }
}

const wnn::Operand SqueezeNet::BuildFire(const wnn::GraphBuilder& builder,
                                         const wnn::Operand& input,
                                         const std::string& convName,
                                         const std::string& conv1x1Name,
                                         const std::string& conv3x3Name) {
    utils::Conv2dOptions convOptions;
    if (!(mLayout == "nchw")) {
        convOptions.inputLayout = wnn::InputOperandLayout::Nhwc;
        convOptions.filterLayout = wnn::Conv2dFilterOperandLayout::Ohwi;
    }
    const wnn::Operand conv = BuildConv(builder, input, convName, &convOptions);
    const wnn::Operand conv1x1 = BuildConv(builder, conv, conv1x1Name, &convOptions);
    convOptions.padding = {1, 1, 1, 1};
    const wnn::Operand conv3x3 = BuildConv(builder, conv, conv3x3Name, &convOptions);
    std::vector<wnn::Operand> inputsOperand = {conv1x1, conv3x3};
    uint32_t axis = mLayout == "nchw" ? 1 : 3;
    return builder.Concat(inputsOperand.size(), inputsOperand.data(), axis);
}

const wnn::Operand SqueezeNet::LoadNchw(const wnn::GraphBuilder& builder, bool softmax) {
    mWeightsPath = mWeightsPath + "squeezenet0_";
    const wnn::Operand input = utils::BuildInput(builder, "input", {1, 3, 224, 224});

    utils::Conv2dOptions conv0Options;
    conv0Options.strides = {2, 2};
    const wnn::Operand conv0 = BuildConv(builder, input, "conv0", &conv0Options);

    utils::Pool2dOptions pool0Options;
    pool0Options.windowDimensions = {3, 3};
    pool0Options.strides = {2, 2};
    const wnn::Operand pool0 = builder.MaxPool2d(conv0, pool0Options.AsPtr());

    const wnn::Operand fire0 = BuildFire(builder, pool0, "conv1", "conv2", "conv3");
    const wnn::Operand fire1 = BuildFire(builder, fire0, "conv4", "conv5", "conv6");
    utils::Pool2dOptions pool1Options = pool0Options;
    const wnn::Operand pool1 = builder.MaxPool2d(fire1, pool1Options.AsPtr());
    const wnn::Operand fire2 = BuildFire(builder, pool1, "conv7", "conv8", "conv9");
    const wnn::Operand fire3 = BuildFire(builder, fire2, "conv10", "conv11", "conv12");
    utils::Pool2dOptions pool2Options = pool0Options;
    const wnn::Operand pool2 = builder.MaxPool2d(fire3, pool2Options.AsPtr());
    const wnn::Operand fire4 = BuildFire(builder, pool2, "conv13", "conv14", "conv15");
    const wnn::Operand fire5 = BuildFire(builder, fire4, "conv16", "conv17", "conv18");
    const wnn::Operand fire6 = BuildFire(builder, fire5, "conv19", "conv20", "conv21");
    const wnn::Operand fire7 = BuildFire(builder, fire6, "conv22", "conv23", "conv24");

    const wnn::Operand conv25 = BuildConv(builder, fire7, "conv25");
    utils::Pool2dOptions pool3Options;
    pool3Options.windowDimensions = {13, 13};
    pool3Options.strides = {13, 13};
    const wnn::Operand pool3 = builder.AveragePool2d(conv25, pool3Options.AsPtr());
    const std::vector<int32_t> newShape = {1, -1};
    const wnn::Operand reshape0 = builder.Reshape(pool3, newShape.data(), newShape.size());
    const wnn::Operand output = softmax ? builder.Softmax(reshape0) : reshape0;
    return output;
}

const wnn::Operand SqueezeNet::LoadNhwc(const wnn::GraphBuilder& builder, bool softmax) {
    mWeightsPath = mWeightsPath;
    const wnn::Operand input = utils::BuildInput(builder, "input", {1, 224, 224, 3});
    utils::Conv2dOptions conv1Options;
    conv1Options.strides = {2, 2};
    conv1Options.autoPad = wnn::AutoPad::SameUpper;
    conv1Options.inputLayout = wnn::InputOperandLayout::Nhwc;
    conv1Options.filterLayout = wnn::Conv2dFilterOperandLayout::Ohwi;
    const wnn::Operand conv1 = BuildConv(builder, input, "conv1", &conv1Options);
    utils::Pool2dOptions maxPool1Options;
    maxPool1Options.windowDimensions = {3, 3};
    maxPool1Options.strides = {2, 2};
    maxPool1Options.layout = wnn::InputOperandLayout::Nhwc;
    const wnn::Operand maxpool1 = builder.MaxPool2d(conv1, maxPool1Options.AsPtr());
    const wnn::Operand fire2 =
        BuildFire(builder, maxpool1, "fire2_squeeze", "fire2_e1x1", "fire2_e3x3");
    const wnn::Operand fire3 =
        BuildFire(builder, fire2, "fire3_squeeze", "fire3_e1x1", "fire3_e3x3");
    const wnn::Operand fire4 =
        BuildFire(builder, fire3, "fire4_squeeze", "fire4_e1x1", "fire4_e3x3");
    utils::Pool2dOptions maxPool4Options = maxPool1Options;
    const wnn::Operand maxpool4 = builder.MaxPool2d(fire4, maxPool1Options.AsPtr());
    const wnn::Operand fire5 =
        BuildFire(builder, maxpool4, "fire5_squeeze", "fire5_e1x1", "fire5_e3x3");
    const wnn::Operand fire6 =
        BuildFire(builder, fire5, "fire6_squeeze", "fire6_e1x1", "fire6_e3x3");
    const wnn::Operand fire7 =
        BuildFire(builder, fire6, "fire7_squeeze", "fire7_e1x1", "fire7_e3x3");
    const wnn::Operand fire8 =
        BuildFire(builder, fire7, "fire8_squeeze", "fire8_e1x1", "fire8_e3x3");
    utils::Pool2dOptions maxPool8Options = maxPool1Options;
    const wnn::Operand maxpool8 = builder.MaxPool2d(fire8, maxPool8Options.AsPtr());
    const wnn::Operand fire9 =
        BuildFire(builder, maxpool8, "fire9_squeeze", "fire9_e1x1", "fire9_e3x3");
    utils::Conv2dOptions conv10Options;
    conv10Options.inputLayout = wnn::InputOperandLayout::Nhwc;
    conv10Options.filterLayout = wnn::Conv2dFilterOperandLayout::Ohwi;
    const wnn::Operand conv10 = BuildConv(builder, fire9, "conv10", &conv10Options);
    utils::Pool2dOptions avgPoolOptions;
    avgPoolOptions.windowDimensions = {13, 13};
    avgPoolOptions.layout = wnn::InputOperandLayout::Nhwc;
    const wnn::Operand averagePool2d = builder.AveragePool2d(conv10, avgPoolOptions.AsPtr());
    const std::vector<int32_t> newShape = {1, -1};
    const wnn::Operand reshape = builder.Reshape(averagePool2d, newShape.data(), newShape.size());
    const wnn::Operand output = softmax ? builder.Softmax(reshape) : reshape;
    return output;
}
