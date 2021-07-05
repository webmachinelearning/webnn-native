#ifndef NODE_OPS_ReduceMean_H_
#define NODE_OPS_ReduceMean_H_

#include <napi.h>
#include <webnn/webnn_cpp.h>

namespace node { namespace op {

    struct ReduceMean {
        static Napi::Value Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder);
    };

}}  // namespace node::op

#endif  // NODE_OPS_TRANSPOSE_H_