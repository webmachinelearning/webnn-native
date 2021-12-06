#ifndef NODE_OPS_RESAMPLE2D_H_
#define NODE_OPS_RESAMPLE2D_H_

#include <napi.h>
#include <webnn/webnn_cpp.h>

namespace node { namespace op {

    struct Resample2d {
        static Napi::Value Build(const Napi::CallbackInfo& info, ml::GraphBuilder builder);
    };

}}  // namespace node::op

#endif  // NODE_OPS_RESAMPLE2D_H_