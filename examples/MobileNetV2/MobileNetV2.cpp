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

#include "examples/MobileNetV2/MobileNetV2.h"

#include <algorithm>

#include "common/Log.h"

MobileNetV2::MobileNetV2(bool nchw) : mNCHW(nchw) {
    mContext = CreateCppContext();
    mContext.SetUncapturedErrorCallback(
        [](MLErrorType type, char const* message, void* userData) {
            if (type != MLErrorType_NoError) {
                dawn::ErrorLog() << "Error type is " << type << ", message is " << message;
            }
        },
        this);
}

const ml::Operand MobileNetV2::BuildConstantFromNpy(const ml::GraphBuilder& builder,
                                                    const std::string& path) {
    const cnpy::NpyArray data = cnpy::npy_load(path);
    mConstants.push_back(data.data_holder);
    return utils::BuildConstant(builder, data.shape, data.data<float>(), data.num_bytes());
}

const ml::Operand MobileNetV2::BuildConv(const ml::GraphBuilder& builder,
                                         const ml::Operand& input,
                                         int32_t convIndex,
                                         bool fused,
                                         utils::Conv2dOptions* options,
                                         const std::string& biasName) {
    std::string prefix = mNCHW ? mDataPath + "conv_" : mDataPath + "Const_";
    std::string suffix = mNCHW ? "_weight.npy" : ".npy";
    const std::string weightsPath = prefix + std::to_string(convIndex) + suffix;
    const ml::Operand convWeights = BuildConstantFromNpy(builder, weightsPath);

    prefix = mNCHW ? mDataPath + "conv_" : mDataPath + "MobilenetV2_";
    if (mNCHW) {
        prefix.append(std::to_string(convIndex));
    }
    const std::string biasPath = prefix + biasName + "_bias.npy";
    const ml::Operand convBias = BuildConstantFromNpy(builder, biasPath);

    std::vector<int32_t> newShape =
        mNCHW ? std::vector<int32_t>({1, -1, 1, 1}) : std::vector<int32_t>({1, 1, 1, -1});
    const ml::Operand reshapedBias = builder.Reshape(convBias, newShape.data(), newShape.size());

    const ml::Conv2dOptions* conv2dOptions = options != nullptr ? options->AsPtr() : nullptr;
    const ml::Operand conv = builder.Conv2d(input, convWeights, conv2dOptions);
    const ml::Operand add = builder.Add(conv, reshapedBias);
    if (fused) {
        ml::ClampOptions clampOptions;
        float min = 0;
        auto minConstant = std::make_shared<std::vector<char>>(sizeof(float));
        std::memcpy(minConstant->data(), &min, sizeof(float));
        mConstants.push_back(minConstant);
        float max = 6;
        auto maxConstant = std::make_shared<std::vector<char>>(sizeof(float));
        std::memcpy(maxConstant->data(), &max, sizeof(float));
        mConstants.push_back(maxConstant);
        clampOptions.minValue =
            utils::BuildConstant(builder, {}, minConstant->data(), sizeof(float));
        clampOptions.maxValue =
            utils::BuildConstant(builder, {}, maxConstant->data(), sizeof(float));
        const ml::Operand clamp = builder.Clamp(add, &clampOptions);
        return clamp;
    } else {
        return add;
    }
}

const ml::Operand MobileNetV2::BuildConvBatchNorm(const ml::GraphBuilder& builder,
                                                  const ml::Operand& input,
                                                  int32_t nameIndex,
                                                  utils::Conv2dOptions* options,
                                                  int32_t subNameIndex) {
    const std::string subName =
        subNameIndex != -1 ? "_linearbottleneck" + std::to_string(subNameIndex) : "";
    std::string prefix = mDataPath + "mobilenetv20_features" + subName;
    const std::string weightsPath = prefix + "_conv" + std::to_string(nameIndex) + "_weight.npy";
    const ml::Operand convWeights = BuildConstantFromNpy(builder, weightsPath);
    prefix.append("_batchnorm" + std::to_string(nameIndex));
    const std::string meanPath = prefix + "_running_mean.npy";
    const ml::Operand mean = BuildConstantFromNpy(builder, meanPath);
    const std::string variancePath = prefix + "_running_var.npy";
    const ml::Operand variance = BuildConstantFromNpy(builder, variancePath);
    const ml::Conv2dOptions* conv2dOptions = options != nullptr ? options->AsPtr() : nullptr;
    const ml::Operand conv = builder.Conv2d(input, convWeights, conv2dOptions);
    const std::string scalePath = prefix + "_gamma.npy";
    const ml::Operand scale = BuildConstantFromNpy(builder, scalePath);
    const std::string biasPath = prefix + "_beta.npy";
    const ml::Operand bias = BuildConstantFromNpy(builder, biasPath);
    ml::BatchNormOptions batchNormOptions;
    batchNormOptions.scale = scale;
    batchNormOptions.bias = bias;
    return builder.BatchNorm(conv, mean, variance, &batchNormOptions);
}

const ml::Operand MobileNetV2::BuildGemm(const ml::GraphBuilder& builder,
                                         const ml::Operand& input,
                                         int32_t gemmIndex) {
    std::string suffix = mNCHW ? "_weight.npy" : "_kernel.npy";
    const std::string weightsPath = mDataPath + "gemm_" + std::to_string(gemmIndex) + suffix;
    const ml::Operand gemmWeights = BuildConstantFromNpy(builder, weightsPath);
    const std::string biasPath = mDataPath + "gemm_" + std::to_string(gemmIndex) + "_bias.npy";
    const ml::Operand gemmBias = BuildConstantFromNpy(builder, biasPath);
    ml::GemmOptions gemmOptions;
    gemmOptions.c = gemmBias;
    gemmOptions.bTranspose = true;
    return builder.Gemm(input, gemmWeights, &gemmOptions);
}

const ml::Operand MobileNetV2::BuildFire(const ml::GraphBuilder& builder,
                                         const ml::Operand& input,
                                         const std::vector<int32_t>& convIndexes,
                                         int32_t groups,
                                         bool strides,
                                         bool shouldAdd) {
    utils::Conv2dOptions convOptions;
    if (!mNCHW) {
        convOptions.inputLayout = ml::InputOperandLayout::Nhwc;
        convOptions.filterLayout = ml::FilterOperandLayout::Hwio;
    }
    const ml::Operand conv1x1 = BuildConv(builder, input, convIndexes[0], true, &convOptions);
    convOptions.padding = {1, 1, 1, 1};
    convOptions.groups = groups;
    if (strides) {
        convOptions.strides = {2, 2};
    }
    const ml::Operand conv3x3 = BuildConv(builder, conv1x1, convIndexes[1], true, &convOptions);
    const ml::Operand conv1x1NotClamp = BuildConv(builder, conv3x3, convIndexes[2], false);
    return shouldAdd ? builder.Add(input, conv1x1NotClamp) : conv1x1NotClamp;
}

const ml::Operand MobileNetV2::BuildBatchNormFire(const ml::GraphBuilder& builder,
                                                  const ml::Operand& input,
                                                  int32_t subNameIndex,
                                                  utils::Conv2dOptions* options) {
    const ml::Operand batchNorm0 = BuildConvBatchNorm(builder, input, 0, nullptr, subNameIndex);
    const ml::Operand batchNorm1 =
        BuildConvBatchNorm(builder, builder.Relu(batchNorm0), 1, options, subNameIndex);
    return BuildConvBatchNorm(builder, builder.Relu(batchNorm1), 2, nullptr, subNameIndex);
}

const ml::Operand MobileNetV2::BuildLinearBottleneck(const ml::GraphBuilder& builder,
                                                     const ml::Operand& input,
                                                     const std::vector<int32_t>& convIndexes,
                                                     int32_t biasIndex,
                                                     utils::Conv2dOptions* dwiseOptions,
                                                     bool shouldAdd) {
    utils::Conv2dOptions convOptions;
    convOptions.autoPad = ml::AutoPad::SameUpper;
    convOptions.inputLayout = ml::InputOperandLayout::Nhwc;
    convOptions.filterLayout = ml::FilterOperandLayout::Ohwi;

    const std::string biasPrefix = "expanded_conv_" + std::to_string(biasIndex);

    const ml::Operand conv1x1 = BuildConv(builder, input, convIndexes[0], true, &convOptions,
                                          biasPrefix + "_expand_Conv2D");
    dwiseOptions->autoPad = ml::AutoPad::SameUpper;
    dwiseOptions->inputLayout = ml::InputOperandLayout::Nhwc;
    dwiseOptions->filterLayout = ml::FilterOperandLayout::Ihwo;
    const ml::Operand conv3x3 = BuildConv(builder, conv1x1, convIndexes[1], true, dwiseOptions,
                                          biasPrefix + "_depthwise_depthwise");

    const ml::Operand conv1x1NotClamp = BuildConv(builder, conv3x3, convIndexes[2], false,
                                                  &convOptions, biasPrefix + "_project_Conv2D");
    return shouldAdd ? builder.Add(input, conv1x1NotClamp) : conv1x1NotClamp;
}

const ml::Operand MobileNetV2::BuildFireMore(const ml::GraphBuilder& builder,
                                             const ml::Operand& input,
                                             const std::vector<int32_t>& convIndexes,
                                             const std::vector<int32_t> groups,
                                             bool strides) {
    const std::vector<int32_t> convList1(convIndexes.begin(), convIndexes.begin() + 3);
    const ml::Operand fire1 = BuildFire(builder, input, convList1, groups[0], strides, false);
    const std::vector<int32_t> convList2(convIndexes.begin() + 3, convIndexes.begin() + 6);
    const ml::Operand fire2 = BuildFire(builder, fire1, convList2, groups[1]);
    if (convIndexes.size() >= 9) {
        const std::vector<int32_t> convList3(convIndexes.begin() + 6, convIndexes.begin() + 9);
        const ml::Operand fire3 = BuildFire(builder, fire2, convList3, groups[1]);
        if (convIndexes.size() == 12) {
            const std::vector<int32_t> convList4(convIndexes.begin() + 9, convIndexes.begin() + 12);
            return BuildFire(builder, fire3, convList4, groups[1]);
        } else {
            return fire3;
        }
    } else {
        return fire2;
    }
}

bool MobileNetV2::LoadNCHW(const std::string& weightsPath, bool softmax) {
    mDataPath = weightsPath;
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(mContext);
    const ml::Operand input = utils::BuildInput(builder, "input", {1, 3, 224, 224});

    utils::Conv2dOptions conv0Options;
    conv0Options.strides = {2, 2};
    conv0Options.padding = {1, 1, 1, 1};
    const ml::Operand conv0 = BuildConv(builder, input, 0, true, &conv0Options);

    utils::Conv2dOptions conv2Options;
    conv2Options.groups = 32;
    conv2Options.padding = {1, 1, 1, 1};
    const ml::Operand conv2 = BuildConv(builder, conv0, 2, true, &conv2Options);
    const ml::Operand conv4 = BuildConv(builder, conv2, 4, false);
    const ml::Operand add15 = BuildFireMore(builder, conv4, {5, 7, 9, 10, 12, 14}, {96, 144});
    const ml::Operand add32 =
        BuildFireMore(builder, add15, {16, 18, 20, 21, 23, 25, 27, 29, 31}, {144, 192});
    const ml::Operand add55 =
        BuildFireMore(builder, add32, {33, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54}, {192, 384});
    const ml::Operand add72 =
        BuildFireMore(builder, add55, {56, 58, 60, 61, 63, 65, 67, 69, 71}, {384, 576}, false);
    const ml::Operand add89 =
        BuildFireMore(builder, add72, {73, 75, 77, 78, 80, 82, 84, 86, 88}, {576, 960});
    const ml::Operand conv94 = BuildFire(builder, add89, {90, 92, 94}, 960, false, false);
    const ml::Operand conv95 = BuildConv(builder, conv94, 95, true);
    const ml::Operand pool97 = builder.AveragePool2d(conv95);
    const std::vector<int32_t> newShape = {1, -1};
    const ml::Operand reshape103 = builder.Reshape(pool97, newShape.data(), newShape.size());
    const ml::Operand gemm104 = BuildGemm(builder, reshape103, 104);
    const ml::Operand output = softmax ? builder.Softmax(gemm104) : gemm104;
    mGraph = utils::AwaitBuild(builder, {{"output", output}});
    if (!mGraph) {
        dawn::ErrorLog() << "Failed to create graph.";
        return false;
    }
    mConstants.clear();

    return true;
}

bool MobileNetV2::LoadNHWC(const std::string& weightsPath, bool softmax) {
    mDataPath = weightsPath;
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(mContext);
    const ml::Operand input = utils::BuildInput(builder, "input", {1, 224, 224, 3});

    utils::Conv2dOptions conv0Options;
    conv0Options.strides = {2, 2};
    conv0Options.autoPad = ml::AutoPad::SameUpper;
    conv0Options.inputLayout = ml::InputOperandLayout::Nhwc;
    conv0Options.filterLayout = ml::FilterOperandLayout::Ohwi;
    const ml::Operand conv0 = BuildConv(builder, input, 90, true, &conv0Options, "Conv_Conv2D");

    utils::Conv2dOptions conv1Options;
    conv1Options.groups = 32;
    conv1Options.autoPad = ml::AutoPad::SameUpper;
    conv1Options.inputLayout = ml::InputOperandLayout::Nhwc;
    conv1Options.filterLayout = ml::FilterOperandLayout::Ihwo;
    const ml::Operand conv1 =
        BuildConv(builder, conv0, 238, true, &conv1Options, "expanded_conv_depthwise_depthwise");

    utils::Conv2dOptions conv2Options;
    conv2Options.autoPad = ml::AutoPad::SameUpper;
    conv2Options.inputLayout = ml::InputOperandLayout::Nhwc;
    conv2Options.filterLayout = ml::FilterOperandLayout::Ohwi;
    const ml::Operand conv2 =
        BuildConv(builder, conv1, 167, false, &conv2Options, "expanded_conv_project_Conv2D");

    utils::Conv2dOptions dwiseConv0Options;
    dwiseConv0Options.groups = 96;
    dwiseConv0Options.strides = {2, 2};
    const ml::Operand bottleneck0 =
        BuildLinearBottleneck(builder, conv2, {165, 99, 73}, 1, &dwiseConv0Options, false);

    utils::Conv2dOptions dwiseConv1Options;
    dwiseConv1Options.groups = 144;
    const ml::Operand bottleneck1 =
        BuildLinearBottleneck(builder, bottleneck0, {3, 119, 115}, 2, &dwiseConv1Options);

    utils::Conv2dOptions dwiseConv2Options;
    dwiseConv2Options.groups = 144;
    dwiseConv2Options.strides = {2, 2};
    const ml::Operand bottleneck2 =
        BuildLinearBottleneck(builder, bottleneck1, {255, 216, 157}, 3, &dwiseConv2Options, false);

    utils::Conv2dOptions dwiseConv3Options;
    dwiseConv3Options.groups = 192;
    const ml::Operand bottleneck3 =
        BuildLinearBottleneck(builder, bottleneck2, {227, 221, 193}, 4, &dwiseConv3Options);

    utils::Conv2dOptions dwiseConv4Options = dwiseConv3Options;
    const ml::Operand bottleneck4 =
        BuildLinearBottleneck(builder, bottleneck3, {243, 102, 215}, 5, &dwiseConv4Options);

    utils::Conv2dOptions dwiseConv5Options;
    dwiseConv5Options.groups = 192;
    dwiseConv5Options.strides = {2, 2};
    const ml::Operand bottleneck5 =
        BuildLinearBottleneck(builder, bottleneck4, {226, 163, 229}, 6, &dwiseConv5Options, false);

    utils::Conv2dOptions dwiseConv6Options;
    dwiseConv6Options.groups = 384;
    const ml::Operand bottleneck6 =
        BuildLinearBottleneck(builder, bottleneck5, {104, 254, 143}, 7, &dwiseConv6Options);

    utils::Conv2dOptions dwiseConv7Options;
    dwiseConv7Options.groups = 384;
    const ml::Operand bottleneck7 =
        BuildLinearBottleneck(builder, bottleneck6, {25, 142, 202}, 8, &dwiseConv7Options);

    utils::Conv2dOptions dwiseConv8Options = dwiseConv7Options;
    const ml::Operand bottleneck8 =
        BuildLinearBottleneck(builder, bottleneck7, {225, 129, 98}, 9, &dwiseConv8Options);

    utils::Conv2dOptions dwiseConv9Options = dwiseConv7Options;
    const ml::Operand bottleneck9 =
        BuildLinearBottleneck(builder, bottleneck8, {169, 2, 246}, 10, &dwiseConv9Options, false);

    utils::Conv2dOptions dwiseConv10Options;
    dwiseConv10Options.groups = 576;
    const ml::Operand bottleneck10 =
        BuildLinearBottleneck(builder, bottleneck9, {162, 87, 106}, 11, &dwiseConv10Options);

    utils::Conv2dOptions dwiseConv11Options = dwiseConv10Options;
    const ml::Operand bottleneck11 =
        BuildLinearBottleneck(builder, bottleneck10, {52, 22, 40}, 12, &dwiseConv11Options);

    utils::Conv2dOptions dwiseConv12Options;
    dwiseConv12Options.groups = 576;
    dwiseConv12Options.strides = {2, 2};
    const ml::Operand bottleneck12 = BuildLinearBottleneck(builder, bottleneck11, {114, 65, 242},
                                                           13, &dwiseConv12Options, false);

    utils::Conv2dOptions dwiseConv13Options;
    dwiseConv13Options.groups = 960;
    const ml::Operand bottleneck13 =
        BuildLinearBottleneck(builder, bottleneck12, {203, 250, 92}, 14, &dwiseConv13Options);

    utils::Conv2dOptions dwiseConv14Options = dwiseConv13Options;
    const ml::Operand bottleneck14 =
        BuildLinearBottleneck(builder, bottleneck13, {133, 130, 258}, 15, &dwiseConv14Options);

    utils::Conv2dOptions dwiseConv15Options = dwiseConv13Options;
    const ml::Operand bottleneck15 = BuildLinearBottleneck(builder, bottleneck14, {60, 248, 100},
                                                           16, &dwiseConv15Options, false);

    utils::Conv2dOptions conv3Options;
    conv3Options.autoPad = ml::AutoPad::SameUpper;
    conv3Options.inputLayout = ml::InputOperandLayout::Nhwc;
    conv3Options.filterLayout = ml::FilterOperandLayout::Ohwi;
    const ml::Operand conv3 =
        BuildConv(builder, bottleneck15, 71, true, &conv3Options, "Conv_1_Conv2D");

    utils::Pool2dOptions poolOptions;
    poolOptions.windowDimensions = {7, 7};
    poolOptions.layout = ml::InputOperandLayout::Nhwc;
    const ml::Operand averagePool2d = builder.AveragePool2d(conv3, poolOptions.AsPtr());

    utils::Conv2dOptions conv4Options = conv3Options;
    const ml::Operand conv4 =
        BuildConv(builder, averagePool2d, 222, false, &conv3Options, "Logits_Conv2d_1c_1x1_Conv2D");

    const std::vector<int32_t> newShape = {1, -1};
    const ml::Operand reshape = builder.Reshape(conv4, newShape.data(), newShape.size());
    const ml::Operand output = softmax ? builder.Softmax(reshape) : reshape;
    mGraph = utils::AwaitBuild(builder, {{"output", output}});
    if (!mGraph) {
        dawn::ErrorLog() << "Failed to create graph.";
        return false;
    }
    mConstants.clear();

    return true;
}

bool MobileNetV2::LoadBatchNormNCHW(const std::string& weightsPath, bool softmax) {
    mDataPath = weightsPath;
    const std::vector<int32_t> padding = {1, 1, 1, 1};
    const std::vector<int32_t> strides = {2, 2};
    const ml::GraphBuilder builder = ml::CreateGraphBuilder(mContext);
    const ml::Operand input = utils::BuildInput(builder, "input", {1, 3, 224, 224});
    utils::Conv2dOptions conv0Options;
    conv0Options.padding = padding;
    conv0Options.strides = strides;
    const ml::Operand batchNorm0 = BuildConvBatchNorm(builder, input, 0, &conv0Options);
    utils::Conv2dOptions fire0Options;
    fire0Options.padding = padding;
    fire0Options.groups = 32;
    const ml::Operand fire0 =
        BuildBatchNormFire(builder, builder.Relu(batchNorm0), 0, &fire0Options);
    utils::Conv2dOptions fire1Options;
    fire1Options = conv0Options;
    fire1Options.groups = 96;
    const ml::Operand fire1 = BuildBatchNormFire(builder, fire0, 1, &fire1Options);
    utils::Conv2dOptions fire2Options;
    fire2Options.padding = padding;
    fire2Options.groups = 144;
    const ml::Operand fire2 = BuildBatchNormFire(builder, fire1, 2, &fire2Options);
    const ml::Operand add0 = builder.Add(fire1, fire2);
    utils::Conv2dOptions fire3Options = conv0Options;
    fire3Options.groups = 144;
    const ml::Operand fire3 = BuildBatchNormFire(builder, add0, 3, &fire3Options);
    utils::Conv2dOptions fire4Options;
    fire4Options.padding = padding;
    fire4Options.groups = 192;
    const ml::Operand fire4 = BuildBatchNormFire(builder, fire3, 4, &fire4Options);
    const ml::Operand add1 = builder.Add(fire3, fire4);
    utils::Conv2dOptions fire5Options = fire4Options;
    const ml::Operand fire5 = BuildBatchNormFire(builder, add1, 5, &fire5Options);
    const ml::Operand add2 = builder.Add(add1, fire5);
    utils::Conv2dOptions fire6Options = fire4Options;
    const ml::Operand fire6 = BuildBatchNormFire(builder, add2, 6, &fire6Options);
    utils::Conv2dOptions fire7Options;
    fire7Options.padding = padding;
    fire7Options.groups = 384;
    const ml::Operand fire7 = BuildBatchNormFire(builder, fire6, 7, &fire7Options);
    const ml::Operand add3 = builder.Add(fire6, fire7);
    utils::Conv2dOptions fire8Options = fire7Options;
    const ml::Operand fire8 = BuildBatchNormFire(builder, add3, 8, &fire8Options);
    const ml::Operand add4 = builder.Add(add3, fire8);
    utils::Conv2dOptions fire9Options = fire7Options;
    const ml::Operand fire9 = BuildBatchNormFire(builder, add4, 9, &fire9Options);
    const ml::Operand add5 = builder.Add(add4, fire9);
    utils::Conv2dOptions fire10Options = conv0Options;
    fire10Options.groups = 384;
    const ml::Operand fire10 = BuildBatchNormFire(builder, add5, 10, &fire10Options);
    utils::Conv2dOptions fire11Options;
    fire11Options.padding = padding;
    fire11Options.groups = 576;
    const ml::Operand fire11 = BuildBatchNormFire(builder, fire10, 11, &fire11Options);
    const ml::Operand add6 = builder.Add(fire10, fire11);
    utils::Conv2dOptions fire12Options = fire11Options;
    const ml::Operand fire12 = BuildBatchNormFire(builder, add6, 12, &fire12Options);
    const ml::Operand add7 = builder.Add(add6, fire12);
    utils::Conv2dOptions fire13Options = conv0Options;
    fire13Options.groups = 576;
    const ml::Operand fire13 = BuildBatchNormFire(builder, add7, 13, &fire13Options);
    utils::Conv2dOptions fire14Options;
    fire14Options.padding = padding;
    fire14Options.groups = 960;
    const ml::Operand fire14 = BuildBatchNormFire(builder, fire13, 14, &fire14Options);
    const ml::Operand add8 = builder.Add(fire13, fire14);
    utils::Conv2dOptions fire15Options = fire14Options;
    const ml::Operand fire15 = BuildBatchNormFire(builder, add8, 15, &fire15Options);
    const ml::Operand add9 = builder.Add(add8, fire15);
    utils::Conv2dOptions fire16Options = fire14Options;
    const ml::Operand fire16 = BuildBatchNormFire(builder, add9, 16, &fire16Options);
    const ml::Operand batchNorm1 = BuildConvBatchNorm(builder, fire16, 1);
    const ml::Operand pool0 = builder.AveragePool2d(builder.Relu(batchNorm1));
    const ml::Operand convWeights1 =
        BuildConstantFromNpy(builder, mDataPath + "mobilenetv20_output_pred_weight.npy");
    const ml::Operand conv1 = builder.Conv2d(pool0, convWeights1);
    const std::vector<int32_t> newShape = {1, -1};
    const ml::Operand reshape0 = builder.Reshape(conv1, newShape.data(), newShape.size());
    const ml::Operand output = softmax ? builder.Softmax(reshape0) : reshape0;
    mGraph = utils::AwaitBuild(builder, {{"output", output}});
    if (!mGraph) {
        dawn::ErrorLog() << "Failed to create graph.";
        return false;
    }
    mConstants.clear();

    return true;
}

ml::Result MobileNetV2::Compute(const void* inputData, size_t inputLength) {
    if (!mGraph) {
        dawn::ErrorLog() << "Compilation is not ready.";
        return ml::Result();
    }
    mResults = utils::AwaitCompute(mGraph, {{"input", {inputData, inputLength}}});
    if (mResults.GetHandle() == nullptr) {
        return ml::Result();
    }
    return mResults.Get("output");
}
