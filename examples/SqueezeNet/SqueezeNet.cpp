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

#include <algorithm>

#include "common/Log.h"

SqueezeNet::SqueezeNet(bool nchw) {
    mNchw = nchw;
    mContext = CreateCppContext();
    mContext.SetUncapturedErrorCallback(
        [](MLErrorType type, char const* message, void* userData) {
            if (type != MLErrorType_NoError) {
                dawn::ErrorLog() << "Error type is " << type << ", message is " << message;
            }
        },
        this);
}

const ml::Operand SqueezeNet::BuildConstantFromNpy(const ml::GraphBuilder& builder,
                                                   const std::string& path) {
    const cnpy::NpyArray data = cnpy::npy_load(path);
    mConstants.push_back(data.data_holder);
    return utils::BuildConstant(builder, data.shape, data.data<float>(), data.num_bytes());
}

const ml::Operand SqueezeNet::BuildConv(const ml::GraphBuilder& builder,
                                        const ml::Operand& input,
                                        const std::string& name,
                                        utils::Conv2dOptions* options) {
    std::string suffix = mNchw ? "_weight.npy" : "_kernel.npy";
    const std::string weightsPath = mDataPath + name + suffix;
    const ml::Operand convWeights = BuildConstantFromNpy(builder, weightsPath);
    std::string biasSuffix = mNchw ? "_bias.npy" : "_Conv2D_bias.npy";
    const std::string biasPath = mDataPath + name + biasSuffix;
    const ml::Operand convBias = BuildConstantFromNpy(builder, biasPath);
    std::vector<int32_t> newShape;
    mNchw ? newShape = {1, -1, 1, 1} : newShape = {1, 1, 1, -1};
    const ml::Operand reshapedBias = builder.Reshape(convBias, newShape.data(), newShape.size());
    const ml::Conv2dOptions* conv2dOptions = options != nullptr ? options->AsPtr() : nullptr;
    const ml::Operand conv = builder.Conv2d(input, convWeights, conv2dOptions);
    const ml::Operand add = builder.Add(conv, reshapedBias);
    return builder.Relu(add);
}

const ml::Operand SqueezeNet::BuildFire(const ml::GraphBuilder& builder,
                                        const ml::Operand& input,
                                        const std::string& convName,
                                        const std::string& conv1x1Name,
                                        const std::string& conv3x3Name) {
    utils::Conv2dOptions convOptions;
    if (!mNchw) {
        convOptions.inputLayout = ml::InputOperandLayout::Nhwc;
        convOptions.filterLayout = ml::FilterOperandLayout::Ohwi;
    }
    const ml::Operand conv = BuildConv(builder, input, convName, &convOptions);
    const ml::Operand conv1x1 = BuildConv(builder, conv, conv1x1Name, &convOptions);
    convOptions.padding = {1, 1, 1, 1};
    const ml::Operand conv3x3 = BuildConv(builder, conv, conv3x3Name, &convOptions);
    std::vector<ml::Operand> inputsOperand = {conv1x1, conv3x3};
    uint32_t axis = mNchw ? 1 : 3;
    return builder.Concat(inputsOperand.size(), inputsOperand.data(), axis);
}

bool SqueezeNet::LoadNCHW(const std::string& weightsPath, bool softmax) {
    mDataPath = weightsPath + "squeezenet0_";
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(mContext);
    const ml::Operand input = utils::BuildInput(builder, "input", {1, 3, 224, 224});

    utils::Conv2dOptions conv0Options;
    conv0Options.strides = {2, 2};
    const ml::Operand conv0 = BuildConv(builder, input, "conv0", &conv0Options);

    utils::Pool2dOptions pool0Options;
    pool0Options.windowDimensions = {3, 3};
    pool0Options.strides = {2, 2};
    const ml::Operand pool0 = builder.MaxPool2d(conv0, pool0Options.AsPtr());

    const ml::Operand fire0 = BuildFire(builder, pool0, "conv1", "conv2", "conv3");
    const ml::Operand fire1 = BuildFire(builder, fire0, "conv4", "conv5", "conv6");
    utils::Pool2dOptions pool1Options = pool0Options;
    const ml::Operand pool1 = builder.MaxPool2d(fire1, pool1Options.AsPtr());
    const ml::Operand fire2 = BuildFire(builder, pool1, "conv7", "conv8", "conv9");
    const ml::Operand fire3 = BuildFire(builder, fire2, "conv10", "conv11", "conv12");
    utils::Pool2dOptions pool2Options = pool0Options;
    const ml::Operand pool2 = builder.MaxPool2d(fire3, pool2Options.AsPtr());
    const ml::Operand fire4 = BuildFire(builder, pool2, "conv13", "conv14", "conv15");
    const ml::Operand fire5 = BuildFire(builder, fire4, "conv16", "conv17", "conv18");
    const ml::Operand fire6 = BuildFire(builder, fire5, "conv19", "conv20", "conv21");
    const ml::Operand fire7 = BuildFire(builder, fire6, "conv22", "conv23", "conv24");

    const ml::Operand conv25 = BuildConv(builder, fire7, "conv25");
    utils::Pool2dOptions pool3Options;
    pool3Options.windowDimensions = {13, 13};
    pool3Options.strides = {13, 13};
    const ml::Operand pool3 = builder.AveragePool2d(conv25, pool3Options.AsPtr());
    const std::vector<int32_t> newShape = {1, -1};
    const ml::Operand reshape0 = builder.Reshape(pool3, newShape.data(), newShape.size());
    const ml::Operand output = softmax ? builder.Softmax(reshape0) : reshape0;
    mGraph = utils::AwaitBuild(builder, {{"output", output}});
    if (!mGraph) {
        dawn::ErrorLog() << "Failed to create graph.";
        return false;
    }
    mConstants.clear();

    return true;
}

bool SqueezeNet::LoadNHWC(const std::string& weightsPath, bool softmax) {
    mDataPath = weightsPath;
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(mContext);
    const ml::Operand input = utils::BuildInput(builder, "input", {1, 224, 224, 3});
    utils::Conv2dOptions conv1Options;
    conv1Options.strides = {2, 2};
    conv1Options.autoPad = ml::AutoPad::SameUpper;
    conv1Options.inputLayout = ml::InputOperandLayout::Nhwc;
    conv1Options.filterLayout = ml::FilterOperandLayout::Ohwi;
    const ml::Operand conv1 = BuildConv(builder, input, "conv1", &conv1Options);
    utils::Pool2dOptions maxPool1Options;
    maxPool1Options.windowDimensions = {3, 3};
    maxPool1Options.strides = {2, 2};
    maxPool1Options.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand maxpool1 = builder.MaxPool2d(conv1, maxPool1Options.AsPtr());
    const ml::Operand fire2 =
        BuildFire(builder, maxpool1, "fire2_squeeze", "fire2_e1x1", "fire2_e3x3");
    const ml::Operand fire3 =
        BuildFire(builder, fire2, "fire3_squeeze", "fire3_e1x1", "fire3_e3x3");
    const ml::Operand fire4 =
        BuildFire(builder, fire3, "fire4_squeeze", "fire4_e1x1", "fire4_e3x3");
    utils::Pool2dOptions maxPool4Options = maxPool1Options;
    const ml::Operand maxpool4 = builder.MaxPool2d(fire4, maxPool1Options.AsPtr());
    const ml::Operand fire5 =
        BuildFire(builder, maxpool4, "fire5_squeeze", "fire5_e1x1", "fire5_e3x3");
    const ml::Operand fire6 =
        BuildFire(builder, fire5, "fire6_squeeze", "fire6_e1x1", "fire6_e3x3");
    const ml::Operand fire7 =
        BuildFire(builder, fire6, "fire7_squeeze", "fire7_e1x1", "fire7_e3x3");
    const ml::Operand fire8 =
        BuildFire(builder, fire7, "fire8_squeeze", "fire8_e1x1", "fire8_e3x3");
    utils::Pool2dOptions maxPool8Options = maxPool1Options;
    const ml::Operand maxpool8 = builder.MaxPool2d(fire8, maxPool8Options.AsPtr());
    const ml::Operand fire9 =
        BuildFire(builder, maxpool8, "fire9_squeeze", "fire9_e1x1", "fire9_e3x3");
    utils::Conv2dOptions conv10Options;
    conv10Options.inputLayout = ml::InputOperandLayout::Nhwc;
    conv10Options.filterLayout = ml::FilterOperandLayout::Ohwi;
    const ml::Operand conv10 = BuildConv(builder, fire9, "conv10", &conv10Options);
    utils::Pool2dOptions avgPoolOptions;
    avgPoolOptions.windowDimensions = {13, 13};
    avgPoolOptions.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand averagePool2d = builder.AveragePool2d(conv10, avgPoolOptions.AsPtr());
    const std::vector<int32_t> newShape = {1, -1};
    const ml::Operand reshape = builder.Reshape(averagePool2d, newShape.data(), newShape.size());
    const ml::Operand output = softmax ? builder.Softmax(reshape) : reshape;
    mGraph = utils::AwaitBuild(builder, {{"output", output}});
    if (!mGraph) {
        dawn::ErrorLog() << "Failed to create graph.";
        return false;
    }
    mConstants.clear();

    return true;
}

ml::Result SqueezeNet::Compute(const void* inputData, size_t inputLength) {
    if (!mGraph) {
        dawn::ErrorLog() << "Graph is not ready.";
        return ml::Result();
    }
    mResults = utils::AwaitCompute(mGraph, {{"input", {inputData, inputLength}}});
    if (mResults.GetHandle() == nullptr) {
        return ml::Result();
    }
    return mResults.Get("output");
}
