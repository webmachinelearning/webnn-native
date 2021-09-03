#ifndef NODE_OPS_REDUCE_H_
#define NODE_OPS_REDUCE_H_

#include <napi.h>
#include <webnn/webnn_cpp.h>

namespace node { namespace op {

    enum ReduceType {
        kReduceL1 = 0,
        kReduceL2,
        kReduceMax,
        kReduceMean,
        kReduceMin,
        kReduceProduct,
        kReduceSum,
    };

    struct Reduce {
        static Napi::Value Build(ReduceType opType,
                                 const Napi::CallbackInfo& info,
                                 ml::GraphBuilder builder);
    };

}}  // namespace node::op

#endif  // NODE_OPS_REDUCE_H_