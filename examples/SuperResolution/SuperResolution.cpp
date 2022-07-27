// Copyright 2022 The WebNN-native Authors
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

#include "examples/SuperResolution/SuperResolution.h"

SuperResolution::SuperResolution() : ExampleBase() {
}

const wnn::Operand SuperResolution::BuildConstantFromNpy(const wnn::GraphBuilder& builder,
                                                         const std::string& path) {
    const cnpy::NpyArray data = cnpy::npy_load(path);
    mConstants.push_back(data.data_holder);
    return utils::BuildConstant(builder, data.shape, data.data<float>(), data.num_bytes());
}

const wnn::Operand SuperResolution::BuildConv(const wnn::GraphBuilder& builder,
                                              const wnn::Operand& input,
                                              int32_t convIndex,
                                              bool relu,
                                              utils::Conv2dOptions* options,
                                              const std::string& biasName) {
    std::string prefix = mLayout == "nchw" ? mWeightsPath + "conv" : mWeightsPath + "Const_";
    std::string suffix = mLayout == "nchw" ? "_weight.npy" : ".npy";
    const std::string weightsPath = prefix + std::to_string(convIndex) + suffix;
    const wnn::Operand convWeights = BuildConstantFromNpy(builder, weightsPath);

    // TODO: Figure out correct "channels last" path suffix.
    prefix = mLayout == "nchw" ? mWeightsPath + "conv" : mWeightsPath + "super_resolution_";
    if (mLayout == "nchw") {
        prefix.append(std::to_string(convIndex));
    }

    const std::string biasPath = prefix + biasName + "_bias.npy";
    const wnn::Operand convBias = BuildConstantFromNpy(builder, biasPath);

    const wnn::Conv2dOptions* conv2dOptions = options != nullptr ? options->AsPtr() : nullptr;
    const wnn::Operand conv2d = builder.Conv2d(input, convWeights, conv2dOptions);

    if (!mFused) {
        if (relu) {
            return builder.Relu(conv2d);
        }
        return conv2d;
    }

    // Fused
    utils::Conv2dOptions fusedOptions;
    if (options != nullptr) {
        fusedOptions = *options;
    }
    fusedOptions.bias = convBias;

    if (relu) {
        fusedOptions.activation = builder.ReluOperator();
    }

    return builder.Conv2d(input, convWeights, fusedOptions.AsPtr());
}

const wnn::Operand SuperResolution::LoadNchw(const wnn::GraphBuilder& builder, bool softmax) {
    const wnn::Operand input = utils::BuildInput(builder, "input", {1, 1, 224, 224});

    utils::Conv2dOptions conv1Options;
    conv1Options.strides = {1, 1};
    conv1Options.padding = {2, 2, 2, 2};
    conv1Options.dilations = {1, 1};
    const wnn::Operand conv1 =
        BuildConv(builder, input, /*convIndex*/ 1, /*relu*/ true, &conv1Options);

    utils::Conv2dOptions conv2Options;
    conv2Options.strides = {1, 1};
    conv2Options.padding = {1, 1, 1, 1};
    conv2Options.dilations = {1, 1};
    const wnn::Operand conv2 =
        BuildConv(builder, conv1, /*convIndex*/ 2, /*relu*/ true, &conv2Options);

    utils::Conv2dOptions conv3Options;
    conv3Options.strides = {1, 1};
    conv3Options.padding = {1, 1, 1, 1};
    conv3Options.dilations = {1, 1};
    const wnn::Operand conv3 =
        BuildConv(builder, conv2, /*convIndex*/ 3, /*relu*/ true, &conv3Options);

    utils::Conv2dOptions conv4Options;
    conv4Options.strides = {1, 1};
    conv4Options.padding = {1, 1, 1, 1};
    conv4Options.dilations = {1, 1};
    const wnn::Operand conv4 =
        BuildConv(builder, conv3, /*convIndex*/ 4, /*relu*/ false, &conv4Options);

    const std::vector<int32_t> newShape1 = {-1, 1, 3, 3, 224, 224};
    const wnn::Operand reshape1 = builder.Reshape(conv4, newShape1.data(), newShape1.size());

    wnn::TransposeOptions transpose1Options;
    std::vector<int32_t> permutation = {0, 1, 4, 2, 5, 3};
    transpose1Options.permutation = permutation.data();
    transpose1Options.permutationCount = permutation.size();
    const wnn::Operand transpose1 = builder.Transpose(reshape1, &transpose1Options);

    const std::vector<int32_t> newShape2 = {-1, 1, 672, 672};
    return builder.Reshape(transpose1, newShape2.data(), newShape2.size());
}