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

#include "webnn_native/onednn/GraphDNNL.h"

#include <numeric>

#include "common/Assert.h"
#include "common/Log.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"

#define FAILED(status) (((dnnl_status_t)(status)) != dnnl_success)

const char* dnnl_status2str(dnnl_status_t v) {
    if (v == dnnl_success)
        return "success";
    if (v == dnnl_out_of_memory)
        return "out_of_memory";
    if (v == dnnl_invalid_arguments)
        return "invalid_arguments";
    if (v == dnnl_unimplemented)
        return "unimplemented";
    if (v == dnnl_iterator_ends)
        return "iterator_ends";
    if (v == dnnl_runtime_error)
        return "runtime_error";
    if (v == dnnl_not_required)
        return "not_required";
    return "unknown status";
}

#define COMPLAIN_DNNL_ERROR_AND_RETURN_DNNL_ERROR(what, status)                           \
    do {                                                                                  \
        dawn::ErrorLog() << what << " returns oneDNN error: " << dnnl_status2str(status); \
        return status;                                                                    \
    } while (0)

#define DNNL_TRY(f)                                            \
    do {                                                       \
        dnnl_status_t s_ = f;                                  \
        if (s_ != dnnl_success)                                \
            COMPLAIN_DNNL_ERROR_AND_RETURN_DNNL_ERROR(#f, s_); \
    } while (0)

#define COMPLAIN_DNNL_ERROR_AND_RETURN_DAWN_ERROR(what, status)                            \
    do {                                                                                   \
        std::string message = std::string(what) + std::string(" returns oneDNN error: ") + \
                              std::string(dnnl_status2str(s_));                            \
        return DAWN_INTERNAL_ERROR(message.c_str());                                       \
    } while (0)

#if defined(DAWN_TRY)
#    undef DAWN_TRY
#endif

#define DAWN_TRY(f)                                            \
    do {                                                       \
        dnnl_status_t s_ = f;                                  \
        if (s_ != dnnl_success)                                \
            COMPLAIN_DNNL_ERROR_AND_RETURN_DAWN_ERROR(#f, s_); \
    } while (0)

#define COMPLAIN_DNNL_ERROR_AND_CALLBACK(what, status)                                     \
    do {                                                                                   \
        std::string message = std::string(what) + std::string(" returns oneDNN error: ") + \
                              std::string(dnnl_status2str(s_));                            \
        if (callback) {                                                                    \
            callback(MLComputeGraphStatus_Error, nullptr, message.c_str(), userdata);      \
            return MLComputeGraphStatus_Error;                                             \
        } else {                                                                           \
            dawn::ErrorLog() << message;                                                   \
            return MLComputeGraphStatus_Error;                                             \
        }                                                                                  \
    } while (0)

#define CALLBACK_TRY(f)                               \
    do {                                              \
        dnnl_status_t s_ = f;                         \
        if (s_ != dnnl_success)                       \
            COMPLAIN_DNNL_ERROR_AND_CALLBACK(#f, s_); \
    } while (0)

namespace webnn_native { namespace onednn {

    class Result : public ResultBase {
      public:
        explicit Result(void* buffer, uint32_t buffer_size, std::vector<int32_t>& dimensions)
            : ResultBase(buffer, buffer_size, dimensions) {
        }
        ~Result() {
            free(mBuffer);
        }
    };

    namespace {
        dnnl_status_t GetDnnlDataType(ml::OperandType operandType, dnnl_data_type_t& dnnlDataType) {
            if (operandType == ml::OperandType::Float32) {
                dnnlDataType = dnnl_f32;
            } else if (operandType == ml::OperandType::Float16) {
                dnnlDataType = dnnl_f16;
            } else if (operandType == ml::OperandType::Int32) {
                dnnlDataType = dnnl_s32;
            } else {
                return dnnl_invalid_arguments;
            }
            return dnnl_success;
        }

        dnnl_status_t GetDnnlDimsAndFormartTag(int32_t const* dimensions,
                                               uint32_t dimensionsCount,
                                               std::vector<dnnl_dim_t>& dnnlDims,
                                               dnnl_format_tag_t& tag) {
            if (dimensionsCount > DNNL_MAX_NDIMS) {
                return dnnl_invalid_arguments;
            } else {
                if (dimensionsCount > 0) {
                    dnnlDims.resize(dimensionsCount);
                    for (uint32_t i = 0; i < dimensionsCount; ++i) {
                        int32_t d = dimensions[i];
                        if (d < 0) {
                            dawn::ErrorLog()
                                << "oneDNN doesn't support the negative dimension value";
                            return dnnl_invalid_arguments;
                        }
                        dnnlDims[i] = d;
                    }
                } else {
                    // for scalar constant
                    dimensionsCount = 1;
                    dnnlDims.resize(dimensionsCount);
                    dnnlDims[0] = 1;
                }
                const dnnl_format_tag_t tags[12] = {
                    dnnl_a,            ///< plain 1D tensor
                    dnnl_ab,           ///< plain 2D tensor
                    dnnl_abc,          ///< plain 3D tensor
                    dnnl_abcd,         ///< plain 4D tensor
                    dnnl_abcde,        ///< plain 5D tensor
                    dnnl_abcdef,       ///< plain 6D tensor
                    dnnl_abcdefg,      ///< plain 7D tensor
                    dnnl_abcdefgh,     ///< plain 8D tensor
                    dnnl_abcdefghi,    ///< plain 9D tensor
                    dnnl_abcdefghij,   ///< plain 10D tensor
                    dnnl_abcdefghijk,  ///< plain 11D tensor
                    dnnl_abcdefghijkl  ///< plain 12D tensor
                };
                tag = tags[dimensionsCount - 1];
                return dnnl_success;
            }
        }

        enum AccessMode { READ, WRITE };

        dnnl_status_t AccessMemory(void* buffer,
                                   size_t size,
                                   dnnl_memory_t mem,
                                   const AccessMode mode) {
            DAWN_ASSERT(buffer != nullptr);
            dnnl_engine_t engine;
            DNNL_TRY(dnnl_memory_get_engine(mem, &engine));
            const dnnl_memory_desc_t* md;
            DNNL_TRY(dnnl_memory_get_memory_desc(mem, &md));
            size_t bytes = dnnl_memory_desc_get_size(md);
            if (bytes != size) {
                dawn::ErrorLog() << "The size is incorrect.";
                return dnnl_invalid_arguments;
            }
            dnnl_engine_kind_t engineKind;
            DNNL_TRY(dnnl_engine_get_kind(engine, &engineKind));
            if (engineKind == dnnl_cpu) {
                void* ptr = nullptr;
                DNNL_TRY(dnnl_memory_get_data_handle(mem, &ptr));
                if (ptr) {
                    if (mode == WRITE) {
                        memcpy(ptr, buffer, bytes);
                    } else {
                        memcpy(buffer, ptr, bytes);
                    }
                } else {
                    dawn::ErrorLog() << "Failed to get memory data handle.";
                    return dnnl_runtime_error;
                }
            } else {
                dawn::ErrorLog() << "Only cpu engine is supported.";
                return dnnl_invalid_arguments;
            }
            return dnnl_success;
        }
        dnnl_status_t WriteToMemory(const void* value, size_t size, dnnl_memory_t mem) {
            return AccessMemory(const_cast<void*>(value), size, mem, WRITE);
        }

        dnnl_status_t ReadFromMemory(void* buffer, size_t size, dnnl_memory_t mem) {
            return AccessMemory(buffer, size, mem, READ);
        }

        dnnl_status_t CreateDnnlMemory(dnnl_engine_t engine,
                                       const OperandDescriptor* desc,
                                       dnnl_memory_t* memory,
                                       const void* value = nullptr,
                                       size_t size = 0) {
            dnnl_data_type_t dataType;
            DNNL_TRY(GetDnnlDataType(desc->type, dataType));
            std::vector<dnnl_dim_t> dims;
            dnnl_format_tag_t tag;
            DNNL_TRY(GetDnnlDimsAndFormartTag(desc->dimensions, desc->dimensionsCount, dims, tag));
            dnnl_memory_desc_t md;
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&md, dims.size(), dims.data(), dataType, tag));
            void* flag;
            if (value != nullptr) {
                flag = DNNL_MEMORY_ALLOCATE;
            } else {
                flag = DNNL_MEMORY_NONE;
            }
            DNNL_TRY(dnnl_memory_create(memory, &md, engine, flag));
            if (value != nullptr) {
                DNNL_TRY(WriteToMemory(value, size, *memory));
            }
            return dnnl_success;
        }

        std::vector<dnnl_dim_t> ShrinkDimensions(const std::vector<dnnl_dim_t>& dims, size_t rank) {
            DAWN_ASSERT(rank <= dims.size());
            std::vector<dnnl_dim_t> newDims(rank);
            for (size_t i = 0; i < rank; ++i) {
                newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
            }
            return newDims;
        }

        std::vector<dnnl_dim_t> ExpandDimensions(const std::vector<dnnl_dim_t>& dims, size_t rank) {
            DAWN_ASSERT(rank >= dims.size());
            std::vector<dnnl_dim_t> newDims(rank, 1);
            for (size_t i = 0; i < dims.size(); ++i) {
                newDims[newDims.size() - i - 1] = dims[dims.size() - i - 1];
            }
            return newDims;
        }

        dnnl_status_t BroadcastDimensions(std::vector<dnnl_dim_t>& aDims,
                                          std::vector<dnnl_dim_t>& bDims,
                                          std::vector<dnnl_dim_t>& cDims,
                                          bool& aBroadcasted,
                                          bool& bBroadcasted,
                                          size_t skipAxis = 0) {
            auto aRank = aDims.size();
            auto bRank = bDims.size();
            auto cRank = cDims.size();
            auto newRank = std::max(aRank, bRank);
            std::vector<dnnl_dim_t> aNewDims;
            std::vector<dnnl_dim_t> bNewDims;
            std::vector<dnnl_dim_t> cNewDims;
            if (newRank > aRank) {
                aNewDims = ExpandDimensions(aDims, newRank);
                aBroadcasted = true;
            } else {
                aNewDims = aDims;
            }
            if (newRank > bRank) {
                bNewDims = ExpandDimensions(bDims, newRank);
                bBroadcasted = true;
            } else {
                bNewDims = bDims;
            }
            if (newRank > cRank) {
                cNewDims = ExpandDimensions(cDims, newRank);
            } else {
                cNewDims = cDims;
            }
            for (size_t i = 0; i < newRank - skipAxis; i++) {
                if (aNewDims[i] == 1 && bNewDims[i] != 1) {
                    cNewDims[i] = bNewDims[i];
                } else if (bNewDims[i] == 1 && aNewDims[i] != 1) {
                    cNewDims[i] = aNewDims[i];
                } else if (aNewDims[i] != bNewDims[i]) {
                    return dnnl_invalid_arguments;
                } else {
                    cNewDims[i] = aNewDims[i];
                }
            }
            aDims = aNewDims;
            bDims = bNewDims;
            cDims = cNewDims;
            return dnnl_success;
        }

        dnnl_status_t ComputeImplicitPaddingForAutoPad(ml::AutoPad autoPad,
                                                       uint32_t& paddingBegin,
                                                       uint32_t& paddingEnd,
                                                       uint32_t dilation,
                                                       uint32_t inputSize,
                                                       uint32_t filterSize,
                                                       uint32_t stride) {
            uint32_t outSize = (inputSize + stride - 1) / stride;
            uint32_t effectiveFilter = (filterSize - 1) * dilation + 1;
            uint32_t neededInput = (outSize - 1) * stride + effectiveFilter;
            uint32_t totalPadding = neededInput - inputSize > 0 ? neededInput - inputSize : 0;
            switch (autoPad) {
                case ml::AutoPad::SameUpper:
                    paddingBegin = totalPadding / 2;
                    paddingEnd = (totalPadding + 1) / 2;
                    return dnnl_success;
                case ml::AutoPad::SameLower:
                    paddingBegin = (totalPadding + 1) / 2;
                    paddingEnd = totalPadding / 2;
                    return dnnl_success;
                default:
                    return dnnl_invalid_arguments;
            }
        }
    }  // anonymous namespace

    Graph::Graph(Context* context) : GraphBase(context) {
    }

    Graph::~Graph() {
        for (auto memory : mMemories) {
            dnnl_memory_destroy(memory);
        }
        for (auto op : mOperations) {
            dnnl_primitive_destroy(op.primitive);
        }
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        const OperandDescriptor* desc = constant->GetOperandDescriptor();
        dnnl_memory_t memory;
        DAWN_TRY(CreateDnnlMemory(GetEngine(), desc, &memory, constant->GetValue(),
                                  constant->GetSize()));
        mMemories.push_back(memory);
        mConstantMemories.insert(memory);
        mOperandMemoryMap.insert(std::make_pair(constant, memory));
        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        const OperandDescriptor* desc = input->GetOperandDescriptor();
        dnnl_memory_t memory;
        DAWN_TRY(CreateDnnlMemory(GetEngine(), desc, &memory));
        mMemories.push_back(memory);
        mOperandMemoryMap.insert(std::make_pair(input, memory));
        mInputMemoryMap.insert(std::make_pair(input->GetName(), memory));
        return {};
    }

    dnnl_status_t Graph::BuildPrimitives() {
        if (mOperandsToBuild.empty()) {
            dawn::ErrorLog() << "No operators to build.";
            return dnnl_invalid_arguments;
        }
        auto& info = mOperandsToBuild[0];
        if (mOperandsToBuild.size() == 1) {
            if (info.opType == OperandType::UNARY) {
                DNNL_TRY(AddUnaryImpl(reinterpret_cast<const op::Unary*>(info.op)));
            } else if (info.opType == OperandType::CLAMP) {
                DNNL_TRY(AddClampImpl(reinterpret_cast<const op::Clamp*>(info.op)));
            } else if (info.opType == OperandType::BINARY) {
                DNNL_TRY(AddBinaryImpl(reinterpret_cast<const op::Binary*>(info.op)));
            } else if (info.opType == OperandType::CONV2D) {
                DNNL_TRY(AddConv2dImpl(reinterpret_cast<const op::Conv2d*>(info.op)));
            } else if (info.opType == OperandType::POOL2D) {
                DNNL_TRY(AddPool2dImpl(reinterpret_cast<const op::Pool2d*>(info.op)));
            } else {
                return dnnl_unimplemented;
            }
        } else if (info.opType == OperandType::CONV2D) {
            // Try to fuse add and clamp into conv2d
            const op::Conv2d* conv2d = reinterpret_cast<const op::Conv2d*>(info.op);
            if (mOperandsToBuild.size() > 3) {
                dawn::ErrorLog() << "Cannot fuse conv2d subgraph with more than 3 ops.";
                return dnnl_invalid_arguments;
            }
            const op::Binary* add = nullptr;
            const op::Clamp* clamp = nullptr;
            for (size_t i = 1; i < mOperandsToBuild.size(); ++i) {
                auto& postOp = mOperandsToBuild[i];
                if (postOp.opType == OperandType::BINARY &&
                    reinterpret_cast<const op::Binary*>(postOp.op)->GetType() ==
                        op::BinaryOpType::kAdd) {
                    add = reinterpret_cast<const op::Binary*>(postOp.op);
                } else if (postOp.opType == OperandType::CLAMP) {
                    clamp = reinterpret_cast<const op::Clamp*>(postOp.op);
                }
            }
            if ((mOperandsToBuild.size() == 2 && !add && !clamp) ||
                (mOperandsToBuild.size() == 3 && (!add || !clamp))) {
                dawn::ErrorLog() << "Failed to fuse conv2d subgraph.";
                return dnnl_invalid_arguments;
            }
            DNNL_TRY(AddConv2dImpl(conv2d, add, clamp));
        } else {
            return dnnl_unimplemented;
        }
        return dnnl_success;
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        DAWN_TRY(BuildPrimitives());
        DAWN_ASSERT(mOperandMemoryMap.find(output) != mOperandMemoryMap.end());
        dnnl_memory_t plainOutputMemory;
        DAWN_TRY(ReorderToPlainFormat(mOperandMemoryMap.at(output), &plainOutputMemory));
        mOutputMemoryMap.insert(std::make_pair(name, plainOutputMemory));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        mOperandsToBuild.push_back({OperandType::BINARY, binary});
        return {};
    }

    dnnl_status_t Graph::AddBinaryImpl(const op::Binary* binary) {
        DAWN_ASSERT(binary->Inputs().size() == 2);
        DAWN_ASSERT(mOperandMemoryMap.find(binary->Inputs()[0].Get()) != mOperandMemoryMap.end());
        dnnl_memory_t aMemory = mOperandMemoryMap.at(binary->Inputs()[0].Get());
        const dnnl_memory_desc_t* aMemoryDesc;
        DNNL_TRY(GetMemoryDesc(aMemory, &aMemoryDesc));
        DAWN_ASSERT(mOperandMemoryMap.find(binary->Inputs()[1].Get()) != mOperandMemoryMap.end());
        dnnl_memory_t bMemory = mOperandMemoryMap.at(binary->Inputs()[1].Get());
        const dnnl_memory_desc_t* bMemoryDesc;
        DNNL_TRY(GetMemoryDesc(bMemory, &bMemoryDesc));
        std::vector<dnnl_dim_t> aDims(aMemoryDesc->dims, aMemoryDesc->dims + aMemoryDesc->ndims);
        std::vector<dnnl_dim_t> bDims(bMemoryDesc->dims, bMemoryDesc->dims + bMemoryDesc->ndims);
        std::vector<dnnl_dim_t> cDims;
        bool aBroadcasted = false;
        bool bBroadcasted = false;
        const int aRank = aDims.size();
        const int bRank = bDims.size();
        int cRank = 0;
        bool needBroadcast = true;
        size_t broadcastSkipAxis = 0;
        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            if (aRank == 1 && bRank == 1) {
                // If both a and b are 1-D, the operation is a vector dot-product,
                // which produces a scalar output.
                cRank = 1;
            } else {
                // The output is a N-D tensor whose rank is the maximum rank of the
                // input tensors.
                cRank = std::max(aRank, bRank);
            }
            if (aRank == 1) {
                // If a is 1-D, it is converted to a 2-D tensor by prepending a 1 to its dimensions
                dnnl_dim_t dim = aDims[0];
                aDims.resize(2);
                aDims[0] = 1;
                aDims[1] = dim;
                aBroadcasted = true;
            }
            if (bRank == 1) {
                // If b is 1-D, it is converted to a 2-D tensor by by appending a 1 to its
                // dimensions.
                dnnl_dim_t dim = bDims[0];
                bDims.resize(2);
                bDims[0] = dim;
                bDims[1] = 1;
                bBroadcasted = true;
            }
            if (aDims.size() > 2 || bDims.size() > 2) {
                // If either a or b is N-D, N > 2, it is treated as a stack of matrices
                // with dimensions corresponding to the last two indices. The matrix
                // multiplication will be broadcasted accordingly by following
                // [numpy-broadcasting-rule].
                needBroadcast = true;
                broadcastSkipAxis = 2;
            } else {
                needBroadcast = false;
            }
            // Set output dims.
            cDims.resize(2);
            cDims[0] = aDims[aDims.size() - 2];
            cDims[1] = bDims[bDims.size() - 1];
        } else {
            // The element-wise binary operation will be broadcasted according to
            // [numpy-broadcasting-rule].
            needBroadcast = true;
            broadcastSkipAxis = 0;
        }

        if (needBroadcast) {
            DNNL_TRY(BroadcastDimensions(aDims, bDims, cDims, aBroadcasted, bBroadcasted,
                                         broadcastSkipAxis));
        }
        dnnl_memory_desc_t aBroadcastedMemoryDesc;
        if (aBroadcasted) {
            DNNL_TRY(dnnl_memory_desc_reshape(&aBroadcastedMemoryDesc, aMemoryDesc, aDims.size(),
                                              aDims.data()));
            aMemoryDesc = &aBroadcastedMemoryDesc;
        }
        dnnl_memory_desc_t bBroadcastedMemoryDesc;
        if (bBroadcasted) {
            DNNL_TRY(dnnl_memory_desc_reshape(&bBroadcastedMemoryDesc, bMemoryDesc, bDims.size(),
                                              bDims.data()));
            bMemoryDesc = &bBroadcastedMemoryDesc;
        }
        dnnl_memory_desc_t cInitDesc;
        DNNL_TRY(dnnl_memory_desc_init_by_tag(&cInitDesc, cDims.size(), cDims.data(),
                                              aMemoryDesc->data_type, dnnl_format_tag_any));
        dnnl_primitive_desc_t primitiveDesc;
        dnnl_data_type_t dataType = aMemoryDesc->data_type;
        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            dnnl_memory_desc_t aInitDesc;
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&aInitDesc, aDims.size(), aDims.data(), dataType,
                                                  dnnl_format_tag_any));
            dnnl_memory_desc_t bInitDesc;
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&bInitDesc, bDims.size(), bDims.data(), dataType,
                                                  dnnl_format_tag_any));
            dnnl_matmul_desc_t matmulDesc;
            DNNL_TRY(dnnl_matmul_desc_init(&matmulDesc, &aInitDesc, &bInitDesc, NULL, &cInitDesc));
            DNNL_TRY(
                dnnl_primitive_desc_create(&primitiveDesc, &matmulDesc, NULL, GetEngine(), NULL));
            const dnnl_memory_desc_t* input0InternalMemoryDesc =
                dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_src_md, 0);
            DNNL_TRY(ReorderIfNeeded(aMemoryDesc, aMemory, input0InternalMemoryDesc, &aMemory));
            const dnnl_memory_desc_t* input1InternalMemoryDesc =
                dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_weights_md, 0);
            DNNL_TRY(ReorderIfNeeded(bMemoryDesc, bMemory, input1InternalMemoryDesc, &bMemory));
        } else {
            dnnl_alg_kind_t algKind;
            if (binary->GetType() == op::BinaryOpType::kAdd) {
                algKind = dnnl_binary_add;
            } else if (binary->GetType() == op::BinaryOpType::kMul) {
                algKind = dnnl_binary_mul;
            } else {
                return dnnl_unimplemented;
            }
            dnnl_binary_desc_t binaryDesc;
            DNNL_TRY(
                dnnl_binary_desc_init(&binaryDesc, algKind, aMemoryDesc, bMemoryDesc, &cInitDesc));
            DNNL_TRY(
                dnnl_primitive_desc_create(&primitiveDesc, &binaryDesc, NULL, GetEngine(), NULL));
        }
        dnnl_memory_t cMemory;
        dnnl_primitive_t primitive;
        const dnnl_memory_desc_t* cMemoryDesc =
            dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_dst_md, 0);
        DNNL_TRY(dnnl_memory_create(&cMemory, cMemoryDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));
        DNNL_TRY(dnnl_primitive_create(&primitive, primitiveDesc));
        DNNL_TRY(dnnl_primitive_desc_destroy(primitiveDesc));
        std::vector<dnnl_exec_arg_t> args;
        if (binary->GetType() == op::BinaryOpType::kMatMul) {
            args = {{DNNL_ARG_SRC, aMemory}, {DNNL_ARG_WEIGHTS, bMemory}, {DNNL_ARG_DST, cMemory}};
        } else {
            args = {{DNNL_ARG_SRC_0, aMemory}, {DNNL_ARG_SRC_1, bMemory}, {DNNL_ARG_DST, cMemory}};
        }
        mOperations.push_back({primitive, args});
        mMemories.push_back(cMemory);
        mOperandMemoryMap.insert(std::make_pair(binary, cMemory));
        if (cRank != 0 && cRank < cMemoryDesc->ndims) {
            std::vector<dnnl_dim_t> cDims(cMemoryDesc->dims,
                                          cMemoryDesc->dims + cMemoryDesc->ndims);
            std::vector<dnnl_dim_t> cNewDims = ShrinkDimensions(cDims, cRank);
            dnnl_memory_desc_t cNewMemoryDesc;
            DNNL_TRY(dnnl_memory_desc_reshape(&cNewMemoryDesc, cMemoryDesc, cNewDims.size(),
                                              cNewDims.data()));
            mMemoryReinterprets.insert(std::make_pair(cMemory, cNewMemoryDesc));
        }
        return dnnl_success;
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        mOperandsToBuild.push_back({OperandType::CONV2D, conv2d});
        return {};
    }

    dnnl_status_t Graph::AddConv2dImpl(const op::Conv2d* conv2d,
                                       const op::Binary* add,
                                       const op::Clamp* clamp) {
        DAWN_ASSERT(conv2d->Inputs().size() == 2);
        const OperandBase* inputOperand = conv2d->Inputs()[0].Get();
        DAWN_ASSERT(mOperandMemoryMap.find(inputOperand) != mOperandMemoryMap.end());
        dnnl_memory_t inputMemory = mOperandMemoryMap.at(inputOperand);
        const dnnl_memory_desc_t* inputMemoryDesc;
        DNNL_TRY(GetMemoryDesc(inputMemory, &inputMemoryDesc));
        std::vector<dnnl_dim_t> inputDims;
        const Conv2dOptions* options = conv2d->GetOptions();
        const dnnl_memory_desc_t* actualInputMemoryDesc;
        dnnl_memory_desc_t transposedInputMemoryDesc;
        if (options->inputLayout == ml::InputOperandLayout::Nhwc) {
            const int permute[] = {0, 2, 3, 1};
            DNNL_TRY(dnnl_memory_desc_permute_axes(&transposedInputMemoryDesc, inputMemoryDesc,
                                                   permute));
            inputDims.assign(transposedInputMemoryDesc.dims,
                             transposedInputMemoryDesc.dims + transposedInputMemoryDesc.ndims);
            // logical dimension is always in {NCHW}
            // physical layout is nhwc for input
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&transposedInputMemoryDesc, inputDims.size(),
                                                  inputDims.data(), inputMemoryDesc->data_type,
                                                  dnnl_nhwc));
            actualInputMemoryDesc = &transposedInputMemoryDesc;
        } else {
            inputDims.assign(inputMemoryDesc->dims, inputMemoryDesc->dims + inputMemoryDesc->ndims);
            actualInputMemoryDesc = inputMemoryDesc;
        }

        const OperandBase* filterOperand = conv2d->Inputs()[1].Get();
        DAWN_ASSERT(mOperandMemoryMap.find(filterOperand) != mOperandMemoryMap.end());
        dnnl_memory_t filterMemory = mOperandMemoryMap.at(filterOperand);
        const dnnl_memory_desc_t* filterMemoryDesc;
        DNNL_TRY(GetMemoryDesc(filterMemory, &filterMemoryDesc));
        std::vector<dnnl_dim_t> filterDims;
        std::vector<dnnl_dim_t> groupFilterDims;
        const dnnl_memory_desc_t* actualFilterMemoryDesc;
        dnnl_memory_desc_t transposedFilterMemoryDesc;
        if (options->filterLayout == ml::FilterOperandLayout::Hwio) {
            const int permute[] = {2, 3, 1, 0};
            DNNL_TRY(dnnl_memory_desc_permute_axes(&transposedFilterMemoryDesc, filterMemoryDesc,
                                                   permute));
            // logical dimension is always in {OIHW}
            // physical layout is hwio for filter
            filterDims.assign(transposedFilterMemoryDesc.dims,
                              transposedFilterMemoryDesc.dims + transposedFilterMemoryDesc.ndims);
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&transposedFilterMemoryDesc, filterDims.size(),
                                                  filterDims.data(), filterMemoryDesc->data_type,
                                                  dnnl_hwio));

            actualFilterMemoryDesc = &transposedFilterMemoryDesc;
        } else if (options->filterLayout == ml::FilterOperandLayout::Ohwi) {
            const int permute[] = {0, 2, 3, 1};
            DNNL_TRY(dnnl_memory_desc_permute_axes(&transposedFilterMemoryDesc, filterMemoryDesc,
                                                   permute));
            filterDims.assign(transposedFilterMemoryDesc.dims,
                              transposedFilterMemoryDesc.dims + transposedFilterMemoryDesc.ndims);
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&transposedFilterMemoryDesc, filterDims.size(),
                                                  filterDims.data(), filterMemoryDesc->data_type,
                                                  dnnl_ohwi));
            actualFilterMemoryDesc = &transposedFilterMemoryDesc;
        } else if (options->filterLayout == ml::FilterOperandLayout::Ihwo) {
            const int permute[] = {1, 2, 3, 0};
            DNNL_TRY(dnnl_memory_desc_permute_axes(&transposedFilterMemoryDesc, filterMemoryDesc,
                                                   permute));
            filterDims.assign(transposedFilterMemoryDesc.dims,
                              transposedFilterMemoryDesc.dims + transposedFilterMemoryDesc.ndims);
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&transposedFilterMemoryDesc, filterDims.size(),
                                                  filterDims.data(), filterMemoryDesc->data_type,
                                                  dnnl_ihwo));
            actualFilterMemoryDesc = &transposedFilterMemoryDesc;
        } else {
            filterDims.assign(filterMemoryDesc->dims,
                              filterMemoryDesc->dims + filterMemoryDesc->ndims);
            actualFilterMemoryDesc = filterMemoryDesc;
        }

        dnnl_memory_desc_t newFilterMemoryDesc;
        if (options->groups != 1) {
            groupFilterDims = {options->groups, filterDims[0] / options->groups, filterDims[1],
                               filterDims[2], filterDims[3]};
            switch (options->filterLayout) {
                case ml::FilterOperandLayout::Oihw:
                    DNNL_TRY(dnnl_memory_desc_init_by_tag(
                        &newFilterMemoryDesc, groupFilterDims.size(), groupFilterDims.data(),
                        filterMemoryDesc->data_type, dnnl_goihw));
                    break;
                case ml::FilterOperandLayout::Hwio:
                    DNNL_TRY(dnnl_memory_desc_init_by_tag(
                        &newFilterMemoryDesc, groupFilterDims.size(), groupFilterDims.data(),
                        filterMemoryDesc->data_type, dnnl_hwigo));
                    break;
                case ml::FilterOperandLayout::Ohwi:
                    DNNL_TRY(dnnl_memory_desc_init_by_tag(
                        &newFilterMemoryDesc, groupFilterDims.size(), groupFilterDims.data(),
                        filterMemoryDesc->data_type, dnnl_gohwi));
                    break;
                case ml::FilterOperandLayout::Ihwo:
                    DNNL_TRY(dnnl_memory_desc_init_by_tag(
                        &newFilterMemoryDesc, groupFilterDims.size(), groupFilterDims.data(),
                        filterMemoryDesc->data_type, dnnl_idhwo));
                    break;
                default:
                    break;
            }
            actualFilterMemoryDesc = &newFilterMemoryDesc;
        }

        dnnl_data_type_t dataType = actualInputMemoryDesc->data_type;
        dnnl_memory_desc_t inputInitDesc;
        DNNL_TRY(dnnl_memory_desc_init_by_tag(&inputInitDesc, inputDims.size(), inputDims.data(),
                                              dataType, dnnl_format_tag_any));

        dnnl_memory_desc_t filterInitDesc;
        if (options->groups == 1) {
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&filterInitDesc, filterDims.size(),
                                                  filterDims.data(), dataType,
                                                  dnnl_format_tag_any));
        } else {
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&filterInitDesc, groupFilterDims.size(),
                                                  groupFilterDims.data(), dataType,
                                                  dnnl_format_tag_any));
        }
        std::vector<dnnl_dim_t> strides = {options->strides[0], options->strides[1]};
        // Non-dilated convolution is defined by setting the dilation parameters to 0
        std::vector<dnnl_dim_t> dilates = {options->dilations[0] == 1 ? 0 : options->dilations[0],
                                           options->dilations[1] == 1 ? 0 : options->dilations[1]};

        uint32_t paddingTop = static_cast<uint32_t>(options->padding[0]);
        uint32_t paddingBottom = static_cast<uint32_t>(options->padding[1]);
        uint32_t paddingLeft = static_cast<uint32_t>(options->padding[2]);
        uint32_t paddingRight = static_cast<uint32_t>(options->padding[3]);

        if (options->autoPad != ml::AutoPad::Explicit) {
            DNNL_TRY(ComputeImplicitPaddingForAutoPad(options->autoPad, paddingTop, paddingBottom,
                                                      options->dilations[0], inputDims[2],
                                                      filterDims[2], strides[0]));
            DNNL_TRY(ComputeImplicitPaddingForAutoPad(options->autoPad, paddingLeft, paddingRight,
                                                      options->dilations[1], inputDims[3],
                                                      filterDims[3], strides[1]));
        }

        std::vector<dnnl_dim_t> padding_l = {paddingTop, paddingLeft};
        std::vector<dnnl_dim_t> padding_r = {paddingBottom, paddingRight};
        std::vector<dnnl_dim_t> outputDims(4);
        outputDims[0] = inputDims[0];
        outputDims[1] = filterDims[0];
        for (int i = 2; i < 4; ++i) {
            int src = inputDims[i];
            int ker = filterDims[i];
            int dil = dilates[i - 2];
            int pad_l = padding_l[i - 2];
            int pad_r = padding_r[i - 2];
            int str = strides[i - 2];
            int ker_range = 1 + (ker - 1) * (dil + 1);
            outputDims[i] = (src - ker_range + pad_l + pad_r) / str + 1;
        }
        dnnl_memory_desc_t outputInitDesc;
        DNNL_TRY(dnnl_memory_desc_init_by_tag(&outputInitDesc, outputDims.size(), outputDims.data(),
                                              dataType, dnnl_format_tag_any));

        // dnnl_memory_desc_t biasInitDesc;
        dnnl_memory_t biasMemory = nullptr;
        const dnnl_memory_desc_t* biasMemoryDesc;
        if (add) {
            DAWN_ASSERT(add->Inputs().size() == 2);
            OperandBase* biasOperand = nullptr;
            if (conv2d == add->Inputs()[0].Get()) {
                biasOperand = add->Inputs()[1].Get();
            } else if (conv2d == add->Inputs()[1].Get()) {
                biasOperand = add->Inputs()[0].Get();
            } else {
                dawn::ErrorLog() << "The add is not fusable.";
                return dnnl_invalid_arguments;
            }

            DAWN_ASSERT(mOperandMemoryMap.find(biasOperand) != mOperandMemoryMap.end());
            biasMemory = mOperandMemoryMap.at(biasOperand);
            DNNL_TRY(GetMemoryDesc(biasMemory, &biasMemoryDesc));
        }

        dnnl_primitive_attr_t attr = nullptr;
        dnnl_post_ops_t postops = nullptr;
        if (clamp) {
            float outputMin = -std::numeric_limits<float>::infinity();
            float outputMax = +std::numeric_limits<float>::infinity();
            if (add) {
                if (add != clamp->Inputs()[0].Get()) {
                    dawn::ErrorLog() << "The clamp is not fusable.";
                    return dnnl_invalid_arguments;
                }
            } else {
                if (conv2d != clamp->Inputs()[0].Get()) {
                    dawn::ErrorLog() << "The clamp is not fusable.";
                    return dnnl_invalid_arguments;
                }
            }
            const ClampOptions* options = clamp->GetOptions();
            if (options->minValue != nullptr) {
                DAWN_ASSERT(mOperandMemoryMap.find(options->minValue) != mOperandMemoryMap.end());
                dnnl_memory_t minMemory = mOperandMemoryMap.at(options->minValue);
                DNNL_TRY(ReadFromMemory(&outputMin, sizeof(outputMin), minMemory));
            }
            if (options->maxValue != nullptr) {
                DAWN_ASSERT(mOperandMemoryMap.find(options->maxValue) != mOperandMemoryMap.end());
                dnnl_memory_t maxMemory = mOperandMemoryMap.at(options->maxValue);
                DNNL_TRY(ReadFromMemory(&outputMax, sizeof(outputMax), maxMemory));
            }
            DNNL_TRY(dnnl_post_ops_create(&postops));
            DNNL_TRY(dnnl_post_ops_append_eltwise(postops, 1.0, dnnl_eltwise_clip, outputMin,
                                                  outputMax));
            DNNL_TRY(dnnl_primitive_attr_create(&attr));
            DNNL_TRY(dnnl_primitive_attr_set_post_ops(attr, postops));
        }

        dnnl_convolution_desc_t convDesc;
        DNNL_TRY(dnnl_dilated_convolution_forward_desc_init(
            &convDesc, dnnl_forward, dnnl_convolution_direct, &inputInitDesc, &filterInitDesc,
            add ? biasMemoryDesc : NULL, &outputInitDesc, strides.data(), dilates.data(),
            padding_l.data(), padding_r.data()));
        dnnl_primitive_desc_t primitiveDesc;
        DNNL_TRY(dnnl_primitive_desc_create(&primitiveDesc, &convDesc, clamp ? attr : NULL,
                                            GetEngine(), NULL));

        if (attr) {
            DNNL_TRY(dnnl_primitive_attr_destroy(attr));
        }
        if (postops) {
            DNNL_TRY(dnnl_post_ops_destroy(postops));
        }

        const dnnl_memory_desc_t* inputInternalMemoryDesc =
            dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_src_md, 0);

        dnnl_memory_t inputInternalMemory;
        DNNL_TRY(ReorderIfNeeded(actualInputMemoryDesc, inputMemory, inputInternalMemoryDesc,
                                 &inputInternalMemory));
        const dnnl_memory_desc_t* filterInternalMemoryDesc =
            dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_weights_md, 0);
        dnnl_memory_t filterInternalMemory;
        DNNL_TRY(ReorderIfNeeded(actualFilterMemoryDesc, filterMemory, filterInternalMemoryDesc,
                                 &filterInternalMemory));
        const dnnl_memory_desc_t* outputMemoryDesc =
            dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_dst_md, 0);
        dnnl_memory_t outputMemory;
        DNNL_TRY(
            dnnl_memory_create(&outputMemory, outputMemoryDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));

        dnnl_primitive_t primitive;
        DNNL_TRY(dnnl_primitive_create(&primitive, primitiveDesc));
        DNNL_TRY(dnnl_primitive_desc_destroy(primitiveDesc));
        std::vector<dnnl_exec_arg_t> args = {{DNNL_ARG_SRC, inputInternalMemory},
                                             {DNNL_ARG_WEIGHTS, filterInternalMemory},
                                             {DNNL_ARG_DST, outputMemory}};
        if (add) {
            args.push_back({DNNL_ARG_BIAS, biasMemory});
        }
        mOperations.push_back({primitive, args});
        mMemories.push_back(outputMemory);

        const OperandBase* output = clamp ? reinterpret_cast<const OperandBase*>(clamp)
                                          : (add ? reinterpret_cast<const OperandBase*>(add)
                                                 : reinterpret_cast<const OperandBase*>(conv2d));

        if (options->inputLayout == ml::InputOperandLayout::Nhwc) {
            // reorder the output from primitive query layout to nhwc
            dnnl_memory_desc_t finalOutputMemoryDesc;
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&finalOutputMemoryDesc, outputDims.size(),
                                                  outputDims.data(), dataType, dnnl_nhwc));
            dnnl_memory_t finalOutputMemory;
            DNNL_TRY(ReorderIfNeeded(outputMemoryDesc, outputMemory, &finalOutputMemoryDesc,
                                     &finalOutputMemory));
            mOperandMemoryMap.insert(std::make_pair(output, finalOutputMemory));

            // transpose the output logical dims to nhwc
            dnnl_memory_desc_t transposeOutputMemoryDesc;
            std::vector<dnnl_dim_t> finalOutputDims(4);
            finalOutputDims[0] = outputDims[0];
            finalOutputDims[1] = outputDims[2];
            finalOutputDims[2] = outputDims[3];
            finalOutputDims[3] = outputDims[1];
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&transposeOutputMemoryDesc,
                                                  finalOutputDims.size(), finalOutputDims.data(),
                                                  dataType, dnnl_nchw));
            mMemoryReinterprets.insert(
                std::make_pair(finalOutputMemory, transposeOutputMemoryDesc));
        } else {
            mOperandMemoryMap.insert(std::make_pair(output, outputMemory));
        }

        return dnnl_success;
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        mOperandsToBuild.push_back({POOL2D, pool2d});
        return {};
    }

    dnnl_status_t Graph::AddPool2dImpl(const op::Pool2d* pool2d) {
        DAWN_ASSERT(pool2d->Inputs().size() == 1);
        const OperandBase* inputOperand = pool2d->Inputs()[0].Get();
        DAWN_ASSERT(mOperandMemoryMap.find(inputOperand) != mOperandMemoryMap.end());
        dnnl_memory_t inputMemory = mOperandMemoryMap.at(inputOperand);
        const dnnl_memory_desc_t* inputMemoryDesc;
        DNNL_TRY(GetMemoryDesc(inputMemory, &inputMemoryDesc));
        std::vector<dnnl_dim_t> inputDims(inputMemoryDesc->dims,
                                          inputMemoryDesc->dims + inputMemoryDesc->ndims);
        dnnl_data_type_t dataType = inputMemoryDesc->data_type;
        const Pool2dOptions* options = pool2d->GetOptions();
        if (options->layout != ml::InputOperandLayout::Nchw) {
            // FIXME(nhu): implement the nhwc layout.
            return dnnl_unimplemented;
        }
        std::vector<dnnl_dim_t> kernel;
        if (options->windowDimensions != nullptr) {
            kernel = {options->windowDimensions[0], options->windowDimensions[1]};
        } else {
            kernel = {inputDims[2], inputDims[3]};
        }
        std::vector<dnnl_dim_t> strides = {options->strides[0], options->strides[1]};
        // Non-dilated convolution is defined by setting the dilation parameters to 0
        std::vector<dnnl_dim_t> dilates = {options->dilations[0] == 1 ? 0 : options->dilations[0],
                                           options->dilations[1] == 1 ? 0 : options->dilations[1]};

        std::vector<dnnl_dim_t> padding_l = {options->padding[0], options->padding[2]};
        std::vector<dnnl_dim_t> padding_r = {options->padding[1], options->padding[3]};
        std::vector<dnnl_dim_t> outputDims(4);
        outputDims[0] = inputDims[0];
        // Assume input layout is oihw
        outputDims[1] = inputDims[1];
        for (int i = 2; i < 4; ++i) {
            int src = inputDims[i];
            int ker = kernel[i - 2];
            int dil = dilates[i - 2];
            int pad_l = padding_l[i - 2];
            int pad_r = padding_r[i - 2];
            int str = strides[i - 2];
            int ker_range = 1 + (ker - 1) * (dil + 1);
            outputDims[i] = (src - ker_range + pad_l + pad_r) / str + 1;
        }
        dnnl_memory_desc_t outputInitDesc;
        DNNL_TRY(dnnl_memory_desc_init_by_tag(&outputInitDesc, outputDims.size(), outputDims.data(),
                                              dataType, dnnl_format_tag_any));
        dnnl_alg_kind_t poolType;
        if (pool2d->GetType() == op::Pool2dType::kAveragePool2d) {
            poolType = dnnl_pooling_avg;
        } else if (pool2d->GetType() == op::Pool2dType::kMaxPool2d) {
            poolType = dnnl_pooling_max;
        } else {
            return dnnl_invalid_arguments;
        }
        dnnl_pooling_v2_desc_t poolDesc;
        DNNL_TRY(dnnl_pooling_v2_forward_desc_init(
            &poolDesc, dnnl_forward, poolType, inputMemoryDesc, &outputInitDesc, strides.data(),
            kernel.data(), dilates.data(), padding_l.data(), padding_r.data()));
        dnnl_primitive_desc_t primitiveDesc;
        DNNL_TRY(dnnl_primitive_desc_create(&primitiveDesc, &poolDesc, NULL, GetEngine(), NULL));
        const dnnl_memory_desc_t* outputMemoryDesc =
            dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_dst_md, 0);
        dnnl_memory_t outputMemory;
        DNNL_TRY(
            dnnl_memory_create(&outputMemory, outputMemoryDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));
        dnnl_primitive_t primitive;
        DNNL_TRY(dnnl_primitive_create(&primitive, primitiveDesc));
        std::vector<dnnl_exec_arg_t> args = {{DNNL_ARG_SRC, inputMemory},
                                             {DNNL_ARG_DST, outputMemory}};
        if (poolType == dnnl_pooling_max) {
            const dnnl_memory_desc_t* workspaceMemoryDesc =
                dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_workspace_md, 0);
            dnnl_memory_t workspaceMemory;
            DNNL_TRY(dnnl_memory_create(&workspaceMemory, workspaceMemoryDesc, GetEngine(),
                                        DNNL_MEMORY_ALLOCATE));
            args.push_back({DNNL_ARG_WORKSPACE, workspaceMemory});
            mMemories.push_back(workspaceMemory);
        }
        DNNL_TRY(dnnl_primitive_desc_destroy(primitiveDesc));
        mOperations.push_back({primitive, args});
        mMemories.push_back(outputMemory);
        mOperandMemoryMap.insert(std::make_pair(pool2d, outputMemory));
        return dnnl_success;
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        mOperandsToBuild.push_back({OperandType::UNARY, unary});
        return {};
    }

    dnnl_status_t Graph::AddUnaryImpl(const op::Unary* unary) {
        DAWN_ASSERT(unary->Inputs().size() == 1);
        const OperandBase* inputOperand = unary->Inputs()[0].Get();
        DAWN_ASSERT(mOperandMemoryMap.find(inputOperand) != mOperandMemoryMap.end());
        dnnl_memory_t inputMemory = mOperandMemoryMap.at(inputOperand);
        const dnnl_memory_desc_t* inputMemoryDesc;
        DNNL_TRY(GetMemoryDesc(inputMemory, &inputMemoryDesc));
        dnnl_primitive_desc_t primitiveDesc;
        dnnl_primitive_t primitive;
        dnnl_memory_t outputMemory;
        if (unary->GetType() == op::UnaryOpType::kRelu) {
            dnnl_eltwise_desc_t eltWiseDesc;
            DNNL_TRY(dnnl_eltwise_forward_desc_init(&eltWiseDesc, dnnl_forward, dnnl_eltwise_relu,
                                                    inputMemoryDesc, 0, 0));
            DNNL_TRY(dnnl_primitive_desc_create(&primitiveDesc, &eltWiseDesc, nullptr, GetEngine(),
                                                nullptr));
        } else if (unary->GetType() == op::UnaryOpType::kSoftmax) {
            dnnl_softmax_desc_t softmaxDesc;
            DNNL_TRY(
                dnnl_softmax_forward_desc_init(&softmaxDesc, dnnl_forward, inputMemoryDesc, 1));
            DNNL_TRY(dnnl_primitive_desc_create(&primitiveDesc, &softmaxDesc, nullptr, GetEngine(),
                                                nullptr));
        } else {
            return dnnl_unimplemented;
        }
        const dnnl_memory_desc_t* outputMemoryDesc =
            dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_dst_md, 0);
        DNNL_TRY(
            dnnl_memory_create(&outputMemory, outputMemoryDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));
        DNNL_TRY(dnnl_primitive_create(&primitive, primitiveDesc));
        DNNL_TRY(dnnl_primitive_desc_destroy(primitiveDesc));
        mOperations.push_back(
            {primitive, {{DNNL_ARG_SRC, inputMemory}, {DNNL_ARG_DST, outputMemory}}});
        mMemories.push_back(outputMemory);
        mOperandMemoryMap.insert(std::make_pair(unary, outputMemory));
        return dnnl_success;
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        mOperandsToBuild.push_back({OperandType::CLAMP, clamp});
        return {};
    }

    dnnl_status_t Graph::AddClampImpl(const op::Clamp* clamp) {
        auto inputsOperand = clamp->Inputs();
        DAWN_ASSERT(inputsOperand.size() == 1 || inputsOperand.size() == 2 ||
                    inputsOperand.size() == 3);
        const OperandBase* inputOperand = inputsOperand[0].Get();
        DAWN_ASSERT(mOperandMemoryMap.find(inputOperand) != mOperandMemoryMap.end());
        dnnl_memory_t inputMemory = mOperandMemoryMap.at(inputOperand);
        const dnnl_memory_desc_t* inputMemoryDesc;
        DNNL_TRY(GetMemoryDesc(inputMemory, &inputMemoryDesc));
        std::vector<dnnl_dim_t> inputDims(inputMemoryDesc->dims,
                                          inputMemoryDesc->dims + inputMemoryDesc->ndims);

        const ClampOptions* options = clamp->GetOptions();
        dnnl_memory_t tempMemory;
        std::vector<dnnl_dim_t> tempDims;
        const dnnl_memory_desc_t* tempMemoryDesc;
        // compare input with minValue
        if (options->minValue != nullptr) {
            const OperandBase* minOperand = inputsOperand[1].Get();
            DAWN_ASSERT(mOperandMemoryMap.find(inputsOperand[1].Get()) != mOperandMemoryMap.end());
            dnnl_memory_t minMemory = mOperandMemoryMap.at(minOperand);
            const dnnl_memory_desc_t* minMemoryDesc;
            DNNL_TRY(GetMemoryDesc(minMemory, &minMemoryDesc));
            std::vector<dnnl_dim_t> minDims(minMemoryDesc->dims,
                                            minMemoryDesc->dims + minMemoryDesc->ndims);

            bool inputBroadcasted = false;
            bool minBroadcasted = false;
            DNNL_TRY(BroadcastDimensions(inputDims, minDims, tempDims, inputBroadcasted,
                                         minBroadcasted, 0));
            dnnl_memory_desc_t minBroadcastedMemoryDesc;
            if (minBroadcasted) {
                DNNL_TRY(dnnl_memory_desc_reshape(&minBroadcastedMemoryDesc, minMemoryDesc,
                                                  minDims.size(), minDims.data()));
                minMemoryDesc = &minBroadcastedMemoryDesc;
            }

            dnnl_memory_desc_t tempInitDesc;
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&tempInitDesc, tempDims.size(), tempDims.data(),
                                                  inputMemoryDesc->data_type, dnnl_format_tag_any));

            dnnl_primitive_desc_t primitiveDesc;
            dnnl_alg_kind_t algKind = dnnl_binary_max;
            dnnl_binary_desc_t binaryDesc;
            DNNL_TRY(dnnl_binary_desc_init(&binaryDesc, algKind, inputMemoryDesc, minMemoryDesc,
                                           &tempInitDesc));
            DNNL_TRY(
                dnnl_primitive_desc_create(&primitiveDesc, &binaryDesc, NULL, GetEngine(), NULL));

            dnnl_primitive_t primitive;
            tempMemoryDesc = dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_dst_md, 0);
            DNNL_TRY(
                dnnl_memory_create(&tempMemory, tempMemoryDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));
            DNNL_TRY(dnnl_primitive_create(&primitive, primitiveDesc));
            DNNL_TRY(dnnl_primitive_desc_destroy(primitiveDesc));
            std::vector<dnnl_exec_arg_t> args;
            args = {{DNNL_ARG_SRC_0, inputMemory},
                    {DNNL_ARG_SRC_1, minMemory},
                    {DNNL_ARG_DST, tempMemory}};
            mOperations.push_back({primitive, args});
            mMemories.push_back(tempMemory);
        } else {
            tempMemory = inputMemory;
            tempDims = inputDims;
            tempMemoryDesc = inputMemoryDesc;
        }

        dnnl_memory_t outMemory;
        std::vector<dnnl_dim_t> outDims;
        const dnnl_memory_desc_t* outMemoryDesc;
        // compare temp with maxValue
        if (options->maxValue != nullptr) {
            auto index = options->minValue == nullptr ? 1 : 2;
            const OperandBase* maxOperand = inputsOperand[index].Get();
            DAWN_ASSERT(mOperandMemoryMap.find(inputsOperand[index].Get()) !=
                        mOperandMemoryMap.end());
            dnnl_memory_t maxMemory = mOperandMemoryMap.at(maxOperand);
            const dnnl_memory_desc_t* maxMemoryDesc;
            DNNL_TRY(GetMemoryDesc(maxMemory, &maxMemoryDesc));
            std::vector<dnnl_dim_t> maxDims(maxMemoryDesc->dims,
                                            maxMemoryDesc->dims + maxMemoryDesc->ndims);

            std::vector<dnnl_dim_t> outDims;
            bool tempBroadcasted = false;
            bool maxBroadcasted = false;
            DNNL_TRY(BroadcastDimensions(tempDims, maxDims, outDims, tempBroadcasted,
                                         maxBroadcasted, 0));
            dnnl_memory_desc_t maxBroadcastedMemoryDesc;
            if (maxBroadcasted) {
                DNNL_TRY(dnnl_memory_desc_reshape(&maxBroadcastedMemoryDesc, maxMemoryDesc,
                                                  maxDims.size(), maxDims.data()));
                maxMemoryDesc = &maxBroadcastedMemoryDesc;
            }

            dnnl_memory_desc_t outInitDesc;
            DNNL_TRY(dnnl_memory_desc_init_by_tag(&outInitDesc, outDims.size(), outDims.data(),
                                                  inputMemoryDesc->data_type, dnnl_format_tag_any));

            dnnl_primitive_desc_t primitiveDesc;
            dnnl_alg_kind_t algKind = dnnl_binary_min;
            dnnl_binary_desc_t binaryDesc;
            DNNL_TRY(dnnl_binary_desc_init(&binaryDesc, algKind, tempMemoryDesc, maxMemoryDesc,
                                           &outInitDesc));
            DNNL_TRY(
                dnnl_primitive_desc_create(&primitiveDesc, &binaryDesc, NULL, GetEngine(), NULL));

            dnnl_primitive_t primitive;
            outMemoryDesc = dnnl_primitive_desc_query_md(primitiveDesc, dnnl_query_dst_md, 0);
            DNNL_TRY(
                dnnl_memory_create(&outMemory, outMemoryDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));
            DNNL_TRY(dnnl_primitive_create(&primitive, primitiveDesc));
            DNNL_TRY(dnnl_primitive_desc_destroy(primitiveDesc));
            std::vector<dnnl_exec_arg_t> args;
            args = {{DNNL_ARG_SRC_0, tempMemory},
                    {DNNL_ARG_SRC_1, maxMemory},
                    {DNNL_ARG_DST, outMemory}};
            mOperations.push_back({primitive, args});
            mMemories.push_back(outMemory);
        } else {
            outMemory = tempMemory;
            outDims = tempDims;
            outMemoryDesc = tempMemoryDesc;
        }
        mOperandMemoryMap.insert(std::make_pair(clamp, outMemory));

        return dnnl_success;
    }

    MaybeError Graph::Finish() {
        return {};
    }

    void Graph::CompileImpl(BuildGraphCallbackDelegate delegate) {
        MLBuildGraphStatus status =
            FAILED(dnnl_stream_create(&mStream, GetEngine(), dnnl_stream_default_flags))
                ? MLBuildGraphStatus_Error
                : MLBuildGraphStatus_Success;
        delegate(status, this);
    }

    MLBuildGraphStatus Graph::CompileSyncImpl() {
        MLBuildGraphStatus status =
            FAILED(dnnl_stream_create(&mStream, GetEngine(), dnnl_stream_default_flags))
                ? MLBuildGraphStatus_Error
                : MLBuildGraphStatus_Success;
        return status;
    }

    MLComputeGraphStatus Graph::ComputeSyncImpl(NamedInputsBase* inputs,
                                                NamedOutputsBase* outputs) {
        return this->GenericComputeImpl(inputs, outputs);
    }

    void Graph::ComputeImpl(NamedInputsBase* inputs,
                            MLComputeGraphCallback callback,
                            void* userdata,
                            NamedOutputsBase* outputs) {
        this->GenericComputeImpl(inputs, outputs, callback, userdata);
    }

    MLComputeGraphStatus Graph::GenericComputeImpl(NamedInputsBase* inputs,
                                                   NamedOutputsBase* outputs,
                                                   MLComputeGraphCallback callback,
                                                   void* userdata) {
        for (auto& input : inputs->GetRecords()) {
            dnnl_memory_t inputMemory = mInputMemoryMap.at(input.first);
            CALLBACK_TRY(dnnl_memory_set_data_handle_v2(
                inputMemory, const_cast<void*>(input.second->buffer), mStream));
        }

        for (auto op : mOperations) {
            CALLBACK_TRY(
                dnnl_primitive_execute(op.primitive, mStream, op.args.size(), op.args.data()));
        }

        CALLBACK_TRY(dnnl_stream_wait(mStream));

        std::vector<std::string> outputNames;
        if (outputs != nullptr) {
            for (auto& output : outputs->GetRecords()) {
                outputNames.push_back(output.first);
            }
        } else {
            for (auto& output : mOutputMemoryMap) {
                outputNames.push_back(output.first);
            }
        }

        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            dnnl_memory_t outputMemory = mOutputMemoryMap.at(outputName);
            const dnnl_memory_desc_t* outputMemoryDesc;
            CALLBACK_TRY(GetMemoryDesc(outputMemory, &outputMemoryDesc));
            std::vector<int32_t> dimensions;
            for (int i = 0; i < outputMemoryDesc->ndims; ++i) {
                // FIXME(nhu): convert from int64_t to int32_t.
                dimensions.push_back(outputMemoryDesc->dims[i]);
            }
            size_t bufferLength = dnnl_memory_desc_get_size(outputMemoryDesc);
            void* outputBuffer = malloc(bufferLength);
            CALLBACK_TRY(ReadFromMemory(outputBuffer, bufferLength, outputMemory));
            Ref<ResultBase> result = AcquireRef(new Result(outputBuffer, bufferLength, dimensions));
            results->Set(outputName.c_str(), result.Detach());
            if (outputs != nullptr) {
                const Output* output = outputs->GetRecords().at(outputName);
                if (output->size >= bufferLength) {
                    memcpy(output->buffer, outputBuffer, bufferLength);
                }
            }
        }
        if (callback) {
            callback(MLComputeGraphStatus_Success,
                     reinterpret_cast<MLNamedResults>(results.Detach()), nullptr, userdata);
        }
        return MLComputeGraphStatus_Success;
    }

    dnnl_engine_t Graph::GetEngine() {
        return reinterpret_cast<Context*>(GetContext())->GetEngine();
    }

    dnnl_status_t Graph::GetMemoryDesc(dnnl_memory_t memory, const dnnl_memory_desc_t** desc) {
        if (mMemoryReinterprets.find(memory) != mMemoryReinterprets.end()) {
            *desc = &mMemoryReinterprets.at(memory);
        } else {
            DNNL_TRY(dnnl_memory_get_memory_desc(memory, desc));
        }
        return dnnl_success;
    }

    dnnl_status_t Graph::ReorderIfNeeded(const dnnl_memory_desc_t* srcDesc,
                                         dnnl_memory_t srcMem,
                                         const dnnl_memory_desc_t* dstDesc,
                                         dnnl_memory_t* userDstMem) {
        if (!dnnl_memory_desc_equal(srcDesc, dstDesc)) {
            dnnl_memory_t dstMem;
            DNNL_TRY(dnnl_memory_create(&dstMem, dstDesc, GetEngine(), DNNL_MEMORY_ALLOCATE));
            dnnl_primitive_desc_t reorderDesc;
            DNNL_TRY(dnnl_reorder_primitive_desc_create(&reorderDesc, srcDesc, GetEngine(), dstDesc,
                                                        GetEngine(), NULL));
            dnnl_primitive_t reorder;
            DNNL_TRY(dnnl_primitive_create(&reorder, reorderDesc));
            DNNL_TRY(dnnl_primitive_desc_destroy(reorderDesc));
            std::vector<dnnl_exec_arg_t> args = {{DNNL_ARG_SRC, srcMem}, {DNNL_ARG_DST, dstMem}};
            if (mConstantMemories.find(srcMem) != mConstantMemories.end()) {
                dnnl_stream_t stream;
                DNNL_TRY(dnnl_stream_create(&stream, GetEngine(), dnnl_stream_default_flags));

                DNNL_TRY(dnnl_primitive_execute(reorder, stream, args.size(), args.data()));
                DNNL_TRY(dnnl_primitive_destroy(reorder));
            } else {
                mOperations.push_back({reorder, args});
            }
            mMemories.push_back(dstMem);
            if (userDstMem != nullptr) {
                *userDstMem = dstMem;
            }
        } else {
            if (userDstMem != nullptr) {
                *userDstMem = srcMem;
            }
        }
        return dnnl_success;
    }

    dnnl_status_t Graph::ReorderToPlainFormat(dnnl_memory_t srcMem, dnnl_memory_t* dstMem) {
        const dnnl_memory_desc_t* srcDesc;
        DNNL_TRY(GetMemoryDesc(srcMem, &srcDesc));
        std::vector<int32_t> dimensions(srcDesc->dims, srcDesc->dims + srcDesc->ndims);
        std::vector<dnnl_dim_t> dims;
        dnnl_format_tag_t tag;
        DNNL_TRY(GetDnnlDimsAndFormartTag(dimensions.data(), dimensions.size(), dims, tag));
        dnnl_memory_desc_t plainDesc;
        DNNL_TRY(dnnl_memory_desc_init_by_tag(&plainDesc, dims.size(), dims.data(),
                                              srcDesc->data_type, tag));
        DNNL_TRY(ReorderIfNeeded(srcDesc, srcMem, &plainDesc, dstMem));
        return dnnl_success;
    }

}}  // namespace webnn_native::onednn
