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

#ifndef NODE_UTILS_H_
#define NODE_UTILS_H_

#define NAPI_EXPERIMENTAL
#include <napi.h>
#include <node.h>
#include <cmath>
#include <map>
#include <unordered_map>

#include "Operand.h"
#include "Operator.h"

#define WEBNN_NODE_THROW(message) \
    Napi::Error::New(info.Env(), message).ThrowAsJavaScriptException();

#define WEBNN_NODE_THROW_AND_RETURN(message) \
    WEBNN_NODE_THROW(message);               \
    return info.Env().Undefined();

#define WEBNN_NODE_ASSERT(condition, message) \
    if (!(condition)) {                       \
        WEBNN_NODE_THROW_AND_RETURN(message); \
    }

#define WEBNN_NODE_ASSERT_AND_RETURN(condition, message) \
    if (!(condition)) {                                  \
        WEBNN_NODE_THROW(message);                       \
        return;                                          \
    }

namespace node {

    const int kMaxInt = 0x7FFFFFFF;
    const int kMinInt = -kMaxInt - 1;
    const int kMaxInt8 = (1 << 7) - 1;
    const int kMinInt8 = -(1 << 7);
    const int kMaxUInt8 = (1 << 8) - 1;
    const uint32_t kMaxUInt32 = 0xFFFFFFFFu;

    template <class T>
    bool GetMappedValue(const std::unordered_map<std::string, T> map,
                        const std::string name,
                        T& value) {
        if (map.find(name) == map.end()) {
            return false;
        }
        value = map.at(name);
        return true;
    }

    inline bool GetOperandType(const Napi::Value& jsValue, wnn::OperandType& value) {
        const std::unordered_map<std::string, wnn::OperandType> operandTypeMap = {
            {"float32", wnn::OperandType::Float32}, {"float16", wnn::OperandType::Float16},
            {"int32", wnn::OperandType::Int32},     {"uint32", wnn::OperandType::Uint32},
            {"int8", wnn::OperandType::Int8},       {"uint8", wnn::OperandType::Uint8},
        };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(operandTypeMap, jsValue.As<Napi::String>().Utf8Value(), value);
    }

    inline bool GetInputOperandLayout(const Napi::Value& jsValue, wnn::InputOperandLayout& value) {
        const std::unordered_map<std::string, wnn::InputOperandLayout> inputOperandLayoutMap = {
            {"nchw", wnn::InputOperandLayout::Nchw},
            {"nhwc", wnn::InputOperandLayout::Nhwc},
        };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(inputOperandLayoutMap, jsValue.As<Napi::String>().Utf8Value(), value);
    };

    inline bool GetConv2dFilterOperandLayout(const Napi::Value& jsValue,
                                             wnn::Conv2dFilterOperandLayout& value) {
        const std::unordered_map<std::string, wnn::Conv2dFilterOperandLayout>
            filterOperandLayoutMap = {
                {"oihw", wnn::Conv2dFilterOperandLayout::Oihw},
                {"hwio", wnn::Conv2dFilterOperandLayout::Hwio},
                {"ohwi", wnn::Conv2dFilterOperandLayout::Ohwi},
                {"ihwo", wnn::Conv2dFilterOperandLayout::Ihwo},
            };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(filterOperandLayoutMap, jsValue.As<Napi::String>().Utf8Value(),
                              value);
    };

    inline bool GetConvTranspose2dFilterOperandLayout(
        const Napi::Value& jsValue,
        wnn::ConvTranspose2dFilterOperandLayout& value) {
        const std::unordered_map<std::string, wnn::ConvTranspose2dFilterOperandLayout>
            filterOperandLayoutMap = {
                {"iohw", wnn::ConvTranspose2dFilterOperandLayout::Iohw},
                {"hwoi", wnn::ConvTranspose2dFilterOperandLayout::Hwoi},
                {"ohwi", wnn::ConvTranspose2dFilterOperandLayout::Ohwi},
            };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(filterOperandLayoutMap, jsValue.As<Napi::String>().Utf8Value(),
                              value);
    };

    inline bool GetAutopad(const Napi::Value& jsValue, wnn::AutoPad& value) {
        const std::unordered_map<std::string, wnn::AutoPad> AutoPadMap = {
            {"explicit", wnn::AutoPad::Explicit},
            {"same-upper", wnn::AutoPad::SameUpper},
            {"same-lower", wnn::AutoPad::SameLower},
        };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(AutoPadMap, jsValue.As<Napi::String>().Utf8Value(), value);
    };

    inline bool GetPaddingMode(const Napi::Value& jsValue, wnn::PaddingMode& value) {
        const std::unordered_map<std::string, wnn::PaddingMode> paddingModeMap = {
            {"constant", wnn::PaddingMode::Constant},
            {"edge", wnn::PaddingMode::Edge},
            {"reflection", wnn::PaddingMode::Reflection},
            {"symmetric", wnn::PaddingMode::Symmetric},
        };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(paddingModeMap, jsValue.As<Napi::String>().Utf8Value(), value);
    };

    inline bool GetInterpolationMode(const Napi::Value& jsValue, wnn::InterpolationMode& value) {
        const std::unordered_map<std::string, wnn::InterpolationMode> interpolationModeMap = {
            {"nearest-neighbor", wnn::InterpolationMode::NearestNeighbor},
            {"linear", wnn::InterpolationMode::Linear},
        };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(interpolationModeMap, jsValue.As<Napi::String>().Utf8Value(), value);
    };

    inline bool GetRecurrentNetworkWeightLayout(const Napi::Value& jsValue,
                                                wnn::RecurrentNetworkWeightLayout& value) {
        const std::unordered_map<std::string, wnn::RecurrentNetworkWeightLayout>
            recurrentNetworkWeightLayoutMap = {
                {"zrn", wnn::RecurrentNetworkWeightLayout::Zrn},
                {"rzn", wnn::RecurrentNetworkWeightLayout::Rzn},
            };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(recurrentNetworkWeightLayoutMap,
                              jsValue.As<Napi::String>().Utf8Value(), value);
    };

    inline bool GetRecurrentNetworkDirection(const Napi::Value& jsValue,
                                             wnn::RecurrentNetworkDirection& value) {
        const std::unordered_map<std::string, wnn::RecurrentNetworkDirection>
            recurrentNetworkDirectionMap = {
                {"forward", wnn::RecurrentNetworkDirection::Forward},
                {"backward", wnn::RecurrentNetworkDirection::Backward},
                {"both", wnn::RecurrentNetworkDirection::Both},
            };
        if (!jsValue.IsString()) {
            return false;
        }
        return GetMappedValue(recurrentNetworkDirectionMap, jsValue.As<Napi::String>().Utf8Value(),
                              value);
    };

    inline bool GetValue(const Napi::Value& jsValue, int32_t& value) {
        if (!jsValue.IsNumber()) {
            return false;
        }

        // Here is a workaround to check int32 following
        // https://github.com/nodejs/node-addon-api/issues/57.
        double doubleValue = jsValue.As<Napi::Number>().DoubleValue();
        int32_t intValue = jsValue.As<Napi::Number>().Int32Value();
        if (doubleValue < kMinInt || doubleValue > kMaxInt ||
            std::fabs(doubleValue - intValue) > 1e-6) {
            // It's not integer type.
            return false;
        }
        value = jsValue.As<Napi::Number>().Int32Value();
        return true;
    }

    inline bool GetValue(const Napi::Value& jsValue, uint32_t& value) {
        if (!jsValue.IsNumber()) {
            return false;
        }

        // Here is a algorithm to check uint32 refering to the Chronmium
        double doubleValue = jsValue.As<Napi::Number>().DoubleValue();
        uint32_t uIntValue = jsValue.As<Napi::Number>().Uint32Value();
        if (doubleValue < 0 || doubleValue > kMaxUInt32 ||
            std::fabs(doubleValue - uIntValue) > 1e-6) {
            return false;
        }
        value = jsValue.As<Napi::Number>().Uint32Value();
        return true;
    }

    inline bool GetValue(const Napi::Value& jsValue, int8_t& value) {
        if (!jsValue.IsNumber()) {
            return false;
        }

        double doubleValue = jsValue.As<Napi::Number>().DoubleValue();
        uint32_t int8Value = jsValue.As<Napi::Number>().Int32Value();
        if (doubleValue < kMinInt8 || doubleValue > kMaxInt8 ||
            std::fabs(doubleValue - int8Value) > 1e-6) {
            return false;
        }
        value = static_cast<int8_t>(jsValue.As<Napi::Number>().Int32Value());
        return true;
    }

    inline bool GetValue(const Napi::Value& jsValue, uint8_t& value) {
        if (!jsValue.IsNumber()) {
            return false;
        }

        double doubleValue = jsValue.As<Napi::Number>().DoubleValue();
        uint32_t uIntValue = jsValue.As<Napi::Number>().Int32Value();
        if (doubleValue < 0 || doubleValue > kMaxUInt8 ||
            std::fabs(doubleValue - uIntValue) > 1e-6) {
            return false;
        }
        value = static_cast<uint8_t>(jsValue.As<Napi::Number>().Uint32Value());
        return true;
    }

    inline bool GetValue(const Napi::Value& jsValue, float& value) {
        if (!jsValue.IsNumber()) {
            return false;
        }
        value = jsValue.As<Napi::Number>().FloatValue();
        return true;
    }

    inline bool GetValue(const Napi::Value& jsValue, bool& value) {
        if (!jsValue.IsBoolean()) {
            return false;
        }
        value = jsValue.As<Napi::Boolean>().Value();
        return true;
    }

    inline bool GetValue(const Napi::Value& jsValue, std::string& value) {
        if (!jsValue.IsString()) {
            return false;
        }
        value = jsValue.As<Napi::String>().Utf8Value();
        return true;
    }

    template <typename T>
    inline bool GetArray(const Napi::Value& jsValue,
                         std::vector<T>& array,
                         const size_t size = std::numeric_limits<size_t>::max()) {
        if (!jsValue.IsArray()) {
            return false;
        }
        Napi::Array jsArray = jsValue.As<Napi::Array>();
        if (size != std::numeric_limits<size_t>::max() && size != jsArray.Length()) {
            return false;
        }
        for (uint32_t i = 0; i < jsArray.Length(); ++i) {
            Napi::Value jsItem = static_cast<Napi::Value>(jsArray[i]);
            T value;
            if (!GetValue(jsItem, value)) {
                return false;
            }
            array.push_back(value);
        }
        return true;
    }

    inline uint32_t SizeOfShape(const std::vector<int32_t>& dimensions) {
        uint32_t size = 1;
        for (auto dim : dimensions) {
            size *= dim;
        }
        return size;
    }

    inline bool GetOperand(const Napi::Value& jsValue,
                           wnn::Operand& operand,
                           std::vector<napi_value>& args) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsObject = jsValue.As<Napi::Object>();
        if (!jsObject.InstanceOf(Operand::constructor.Value())) {
            return false;
        }
        operand = Napi::ObjectWrap<Operand>::Unwrap(jsObject)->GetImpl();
        args.push_back(jsObject.As<Napi::Value>());
        return true;
    }

    inline bool GetOperator(const Napi::Value& jsValue,
                            wnn::FusionOperator& mlOperator,
                            std::vector<napi_value>& args) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsObject = jsValue.As<Napi::Object>();
        if (!jsObject.InstanceOf(Operator::constructor.Value())) {
            return false;
        }
        mlOperator = Napi::ObjectWrap<Operator>::Unwrap(jsObject)->GetImpl();
        args.push_back(jsObject.As<Napi::Value>());
        return true;
    }

    inline bool GetOperandArray(const Napi::Value& jsValue,
                                std::vector<wnn::Operand>& array,
                                std::vector<napi_value>& args) {
        if (!jsValue.IsArray()) {
            return false;
        }
        Napi::Array jsArray = jsValue.As<Napi::Array>();
        for (size_t j = 0; j < jsArray.Length(); j++) {
            if (!jsArray.Get(j).IsObject()) {
                return false;
            }
            Napi::Object object = jsArray.Get(j).As<Napi::Object>();
            if (!object.InstanceOf(Operand::constructor.Value())) {
                return false;
            }
            Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
            array.push_back(operand->GetImpl());
            args.push_back(object.As<Napi::Value>());
        }
        return true;
    }

    inline bool GetOperatorArray(const Napi::Value& jsValue,
                                 wnn::OperatorArray& operatorArray,
                                 std::vector<napi_value>& args) {
        if (!jsValue.IsArray()) {
            return false;
        }
        Napi::Array jsArray = jsValue.As<Napi::Array>();
        operatorArray = wnn::CreateOperatorArray();
        for (size_t i = 0; i < jsArray.Length(); i++) {
            if (!jsArray.Get(i).IsObject()) {
                return false;
            } else {
                wnn::FusionOperator mlOperator;
                if (GetOperator(jsArray.Get(i), mlOperator, args)) {
                    operatorArray.Set(mlOperator);
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    struct OperandDescriptor {
      public:
        wnn::OperandType type;
        std::vector<int32_t> dimensions;

        const wnn::OperandDescriptor* AsPtr() {
            if (!dimensions.empty()) {
                mDesc.dimensions = dimensions.data();
                mDesc.dimensionsCount = dimensions.size();
            }
            mDesc.type = type;
            return &mDesc;
        }

      private:
        wnn::OperandDescriptor mDesc;
    };

    inline bool GetOperandDescriptor(const Napi::Value& jsValue, OperandDescriptor& desc) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsDesc = jsValue.As<Napi::Object>();
        if (!jsDesc.Has("type")) {
            return false;
        }
        if (!GetOperandType(jsDesc.Get("type"), desc.type)) {
            return false;
        }
        if (jsDesc.Has("dimensions")) {
            if (!GetArray(jsDesc.Get("dimensions"), desc.dimensions)) {
                return false;
            }
        }
        return true;
    }

    inline bool GetArrayBufferView(const Napi::Value& jsValue,
                                   wnn::ArrayBufferView& arrayBufferView) {
        if (!jsValue.IsTypedArray()) {
            return false;
        }
        Napi::TypedArray jsTypedArray = jsValue.As<Napi::TypedArray>();
        // FIXME: Invalid argument error when passing SharedArrayBuffer
        // see https://github.com/webmachinelearning/webnn-native/issues/106
        // The fix depends N-API to support accessing SharedArrayBuffer,
        // see https://github.com/nodejs/node/issues/23276
        arrayBufferView.buffer =
            reinterpret_cast<void*>(reinterpret_cast<int8_t*>(jsTypedArray.ArrayBuffer().Data()));
        arrayBufferView.byteLength = jsTypedArray.ByteLength();
        arrayBufferView.byteOffset = jsTypedArray.ByteOffset();
        return true;
    }

    inline bool GetArrayBufferView(const Napi::Value& jsValue,
                                   const wnn::OperandType type,
                                   const std::vector<int32_t>& dimensions,
                                   wnn::ArrayBufferView& arrayBufferView) {
        const std::unordered_map<wnn::OperandType, napi_typedarray_type> arrayTypeMap = {
            {wnn::OperandType::Float32, napi_float32_array},
            {wnn::OperandType::Float16, napi_uint16_array},
            {wnn::OperandType::Int32, napi_int32_array},
            {wnn::OperandType::Uint32, napi_uint32_array},
            {wnn::OperandType::Int8, napi_int8_array},
            {wnn::OperandType::Uint8, napi_uint8_array},
        };
        if (!jsValue.IsTypedArray()) {
            return false;
        }

        Napi::TypedArray jsTypedArray = jsValue.As<Napi::TypedArray>();
        if (arrayTypeMap.find(type) == arrayTypeMap.end()) {
            return false;
        }
        if (arrayTypeMap.at(type) != jsTypedArray.TypedArrayType()) {
            return false;
        }

        if (!GetArrayBufferView(jsTypedArray, arrayBufferView)) {
            return false;
        }

        size_t expectedSize;
        switch (type) {
            case wnn::OperandType::Float32:
                expectedSize = sizeof(float);
                break;
            case wnn::OperandType::Float16:
                expectedSize = sizeof(uint16_t);
                break;
            case wnn::OperandType::Int32:
                expectedSize = sizeof(int32_t);
                break;
            case wnn::OperandType::Uint32:
                expectedSize = sizeof(uint32_t);
                break;
            case wnn::OperandType::Int8:
                expectedSize = sizeof(int8_t);
                break;
            case wnn::OperandType::Uint8:
                expectedSize = sizeof(uint8_t);
                break;
            default:
                return false;
        }
        expectedSize *= SizeOfShape(dimensions);
        if (expectedSize != arrayBufferView.byteLength) {
            return false;
        }
        return true;
    }

    inline bool GetNamedOperands(const Napi::Value& jsValue,
                                 wnn::NamedOperands& namedOperands,
                                 std::vector<std::string>& names) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsOutputs = jsValue.As<Napi::Object>();
        namedOperands = wnn::CreateNamedOperands();
        Napi::Array outputNames = jsOutputs.GetPropertyNames();
        if (outputNames.Length() == 0) {
            return false;
        }
        for (size_t j = 0; j < outputNames.Length(); ++j) {
            std::string name = outputNames.Get(j).As<Napi::String>().Utf8Value();
            Napi::Object output = jsOutputs.Get(name).As<Napi::Object>();
            if (!output.InstanceOf(Operand::constructor.Value())) {
                return false;
            }
            wnn::Operand operand = Napi::ObjectWrap<Operand>::Unwrap(output)->GetImpl();
            namedOperands.Set(name.data(), operand);
            names.push_back(name);
        }
        return true;
    }

    inline bool HasOptionMember(const Napi::Object& jsOptions, const std::string& name) {
        return jsOptions.Has(name) && !jsOptions.Get(name).IsUndefined();
    }

    struct Input {
      public:
        wnn::ArrayBufferView bufferView;
        std::vector<int32_t> dimensions;

        const wnn::Input* AsPtr() {
            mInput.resource.arrayBufferView = bufferView;
            mInput.resource.gpuBufferView = {};
            if (!dimensions.empty()) {
                mInput.dimensions = dimensions.data();
                mInput.dimensionsCount = dimensions.size();
            }
            return &mInput;
        }

      private:
        wnn::Input mInput;
    };

    inline bool GetNamedInputs(const Napi::Value& jsValue,
                               std::map<std::string, Input>& namedInputs) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsNamedInputs = jsValue.As<Napi::Object>();
        Napi::Array names = jsNamedInputs.GetPropertyNames();
        if (names.Length() == 0) {
            return false;
        }
        // typedef (MLBufferView or WebGLTexture or GPUTexture) MLResource;
        // dictionary MLInput {
        //   required MLResource resource;
        //   required sequence<long> dimensions;
        // };
        // typedef record<DOMString, (MLResource or MLInput)> MLNamedInputs;
        for (size_t i = 0; i < names.Length(); ++i) {
            Input input = {};
            std::string name = names.Get(i).As<Napi::String>().Utf8Value();
            // FIXME: validate the type of typed array.
            Napi::TypedArray jsTypedArray;
            if (jsNamedInputs.Get(name).IsTypedArray()) {
                jsTypedArray = jsNamedInputs.Get(name).As<Napi::TypedArray>();
            } else {
                Napi::Object jsInput = jsNamedInputs.Get(name).As<Napi::Object>();
                if (!jsInput.Has("resource") || !jsInput.Has("dimensions")) {
                    // Input resource and dimensions are required.
                    return false;
                }
                if (!jsInput.Get("resource").IsTypedArray()) {
                    return false;
                }
                jsTypedArray = jsInput.Get("resource").As<Napi::TypedArray>();

                if (!GetArray(jsInput.Get("dimensions"), input.dimensions)) {
                    return false;
                }
                if (SizeOfShape(input.dimensions) != jsTypedArray.ElementSize()) {
                    return false;
                }
            }
            if (!GetArrayBufferView(jsTypedArray, input.bufferView)) {
                return false;
            }
            namedInputs[name] = input;
        }
        return true;
    }

    inline bool GetNamedOutputs(const Napi::Value& jsValue,
                                std::map<std::string, wnn::Resource>& namedOutputs) {
        if (!jsValue.IsObject()) {
            return false;
        }
        Napi::Object jsNamedOutputs = jsValue.As<Napi::Object>();
        Napi::Array names = jsNamedOutputs.GetPropertyNames();
        if (names.Length() == 0) {
            return false;
        }
        // typedef (MLBufferView or WebGLTexture or GPUTexture) MLResource;
        // typedef record<DOMString, MLResource> MLNamedOutputs;
        for (size_t i = 0; i < names.Length(); ++i) {
            wnn::ArrayBufferView arrayBuffer = {};
            std::string name = names.Get(i).As<Napi::String>().Utf8Value();
            if (!GetArrayBufferView(jsNamedOutputs.Get(name), arrayBuffer)) {
                return false;
            }
            namedOutputs[name] = {arrayBuffer, {}};
        }
        return true;
    }

}  // namespace node

#endif  // NODE_UTILS_H_
