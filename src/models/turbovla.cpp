// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "arch.h"
#include "model.h"
#include "vision_common.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef VLA_TURBOVLA_CUDNN
#include "kernels/turbovla/turbovla_patch_cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include "models/gguf_reader.h"
#include "models/turbovla_trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {

thread_local bool turbovla_cuda_fused_linear = false;
constexpr int32_t turbovla_fused_linear_magic = 0x54424c41;
thread_local bool turbovla_cuda_pytorch_layer_norm = false;
constexpr int32_t turbovla_layer_norm_magic = 0x544c4e31;
constexpr int32_t turbovla_softmax_magic = 0x54534d31;
constexpr int32_t turbovla_bf16_round_magic = 0x54425231;
constexpr int32_t turbovla_rope_magic = 0x54525031;
constexpr int32_t turbovla_rope_linear_magic = 0x54525032;
constexpr int32_t turbovla_scaled_residual_magic = 0x54525331;

struct DinoLayerW {
    ggml_tensor * n1w, * n1b, * n2w, * n2b;
    ggml_tensor * wq, * bq, * wk, * wv, * bv, * wo, * bo;
    ggml_tensor * ls1, * ls2, * upw, * upb, * downw, * downb;
};

struct BertLayerW {
    ggml_tensor * wq, * bq, * wk, * bk, * wv, * bv, * wo, * bo;
    ggml_tensor * n1w, * n1b, * upw, * upb, * downw, * downb, * n2w, * n2b;
};

struct FusionLayerW {
    ggml_tensor * gamma_v, * gamma_l, * nvw, * nvb, * nlw, * nlb;
    ggml_tensor * qvw, * qvb, * ktw, * ktb, * vvw, * vvb, * vtw, * vtb;
    ggml_tensor * ovw, * ovb, * otw, * otb;
    ggml_tensor * inw, * inb, * iow, * iob, * f1w, * f1b, * f2w, * f2b;
    ggml_tensor * t1w, * t1b, * t2w, * t2b;
};

struct DecoderLayerW {
    ggml_tensor * siw, * sib, * sow, * sob;
    ggml_tensor * ciw, * cib, * cow, * cob;
    ggml_tensor * f1w, * f1b, * f2w, * f2b;
    ggml_tensor * n1w, * n1b, * n2w, * n2b, * n3w, * n3b;
};

static ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w,
                        ggml_tensor * b, float eps, bool bf16 = false) {
    if (turbovla_cuda_pytorch_layer_norm && bf16) {
        // PyTorch 2.3.1 dispatches BF16 LayerNorm to a vectorized Welford
        // kernel that fuses the affine epilogue and rounds the result to
        // BF16.  Encode that operation in a GGML_OP_NORM node; the CUDA
        // backend recognizes the marker while other backends retain the
        // portable decomposition below.
        ggml_tensor * y = ggml_norm(c, ggml_cast(c, x, GGML_TYPE_BF16), eps);
        y->src[1] = ggml_cast(c, w, GGML_TYPE_BF16);
        y->src[2] = ggml_cast(c, b, GGML_TYPE_BF16);
        y->op_params[1] = turbovla_layer_norm_magic;
        return y;
    }
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), w), b);
}

static ggml_tensor * linear(ggml_context * c, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x) {
    // The marked CUDA GEMM accepts BF16 activations directly.  LayerNorm's
    // PyTorch-compatible kernel already produces BF16, so bypass its
    // otherwise redundant BF16->F32 cast and avoid converting it straight
    // back to BF16 inside the GEMM dispatcher.
    if (turbovla_cuda_fused_linear && w->type == GGML_TYPE_BF16 &&
        x->type == GGML_TYPE_F32 && x->op == GGML_OP_CPY && x->src[0] &&
        x->src[0]->type == GGML_TYPE_BF16) {
        x = x->src[0];
    }
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    if (turbovla_cuda_fused_linear && w->type == GGML_TYPE_BF16) {
        // Mark biased and bias-free PyTorch linear ops alike so CUDA does not
        // dispatch either form to ggml's MMV/MMF kernels.  src[2] is optional;
        // when present cuBLASLt owns the BF16 bias epilogue, otherwise the
        // regular BF16 cuBLAS GEMM path is used.
        if (b) y->src[2] = b;
        y->op_params[1] = turbovla_fused_linear_magic;
        return y;
    }
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (!b) return y;
    ggml_tensor * bias = b->type == GGML_TYPE_F32
        ? b : ggml_cast(c, b, GGML_TYPE_F32);
    return ggml_add(c, y, bias);
}

static ggml_tensor * diagnostic_matmul(ggml_context * c, ggml_tensor * w,
                                       ggml_tensor * x, bool enabled) {
    if (!enabled) return nullptr;
    ggml_tensor * result = ggml_mul_mat(c, w, x);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

static ggml_tensor * bf16_round(ggml_context * c, ggml_tensor * x, bool enabled) {
    if (!enabled) return x;
    // A BF16 input is already rounded; only expose its values as F32 for the
    // rest of this graph.  GGML_OP_SCALE is F32-only in the CUDA backend.
    if (x->type == GGML_TYPE_BF16)
        return ggml_cast(c, x, GGML_TYPE_F32);
    // TurboVLA's marked cuBLASLt linear epilogue and persistent softmax
    // kernel already round to BF16 before storing the exact rounded value in
    // an F32 ggml tensor. A second F32->BF16->F32 pair is numerically an
    // identity and adds two CUDA kernels after every marked operation.
    int32_t marker = 0;
    if (x && (x->op == GGML_OP_MUL_MAT || x->op == GGML_OP_SOFT_MAX)) {
        if (x->op == GGML_OP_MUL_MAT) {
            marker = x->op_params[1];
        } else {
            std::memcpy(&marker,(const char *)x->op_params+2*sizeof(float),sizeof(marker));
        }
        if (marker == turbovla_fused_linear_magic || marker == turbovla_softmax_magic)
            return x;
    }
    // Fuse the BF16 output epilogue into CUDA elementwise producers.  Their
    // mathematical result is still stored as F32, but it is rounded before
    // the store exactly as the former trailing scale kernel did.
    if (x->op == GGML_OP_ADD || x->op == GGML_OP_MUL) {
        std::memcpy(x->op_params,&turbovla_bf16_round_magic,
                    sizeof(turbovla_bf16_round_magic));
        return x;
    }
    if (x->op == GGML_OP_UNARY &&
        ggml_get_unary_op(x) == GGML_UNARY_OP_GELU_ERF) {
        ggml_tensor * rounded=ggml_new_tensor(
            c,GGML_TYPE_BF16,GGML_MAX_DIMS,x->ne);
        rounded->op=GGML_OP_UNARY;
        rounded->src[0]=x->src[0];
        std::memcpy(rounded->op_params,x->op_params,GGML_MAX_OP_PARAMS);
        std::memcpy((char *)rounded->op_params+sizeof(int32_t),
                    &turbovla_bf16_round_magic,
                    sizeof(turbovla_bf16_round_magic));
        return rounded;
    }
    ggml_tensor * rounded=ggml_scale(c,x,1.0f);
    std::memcpy((char *)rounded->op_params+2*sizeof(float),
                &turbovla_bf16_round_magic,sizeof(turbovla_bf16_round_magic));
    return rounded;
}

static ggml_tensor * pytorch_softmax(ggml_context * c, ggml_tensor * x,
                                     ggml_tensor * mask, bool bf16,
                                     int32_t stable_mode = 0) {
    ggml_tensor * y = ggml_soft_max_ext(c, x, mask, 1.0f, 0.0f);
    if (turbovla_cuda_pytorch_layer_norm && bf16) {
        // The CUDA backend dispatches this marker to PyTorch 2.3.1's
        // persistent-warp reduction order and BF16 output epilogue.
        std::memcpy((char *) y->op_params + 2 * sizeof(float),
                    &turbovla_softmax_magic, sizeof(turbovla_softmax_magic));
        std::memcpy((char *) y->op_params + 2 * sizeof(float) + sizeof(int32_t),
                    &stable_mode, sizeof(stable_mode));
    }
    return bf16_round(c, y, bf16);
}

// PyTorch autocast executes BF16 bmm with BF16 operands and a BF16 result.
// Keeping BF16-valued data in F32 ggml tensors selects a different cuBLAS
// path, which is observably different around cancellation points.
static ggml_tensor * autocast_matmul(ggml_context * c, ggml_tensor * a,
                                     ggml_tensor * b, bool bf16) {
    if (bf16) {
        a = ggml_cast(c, a, GGML_TYPE_BF16);
        b = ggml_cast(c, b, GGML_TYPE_BF16);
    }
    ggml_tensor * result = ggml_mul_mat(c, a, b);
    // Leave BF16 batched matmul at GGML_PREC_DEFAULT so cuBLAS selects a
    // BF16 destination, matching torch.bmm under autocast.  Forcing an F32
    // destination changes the cuBLAS algorithm as well as the epilogue.
    if (!bf16) {
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    return result;
}

static std::string two_digit(size_t value) {
    return (value < 10 ? "0" : "") + std::to_string(value);
}

struct TraceTensorSpec {
    std::string name;
    ggml_tensor * tensor;
    std::vector<int64_t> shape;
    const char * layout;
    const char * operation;
    const char * required_level;
    enum class Transform { None, SubtractGlobalMax, SubtractRowMax } transform;
    ggml_tensor * transform_source;
};

struct TraceLayerNormSpec {
    std::string prefix;
    ggml_tensor * tensor;
    std::vector<int64_t> shape;
    const char * layout;
    const char * stat_layout;
    float eps;
    bool bf16;
};

struct TraceSoftmaxSpec {
    std::string prefix;
    ggml_tensor * tensor;
    std::vector<int64_t> shape;
    const char * layout;
    bool bf16;
};

struct TraceFusionSpec {
    std::string prefix;
    ggml_tensor * logits;
    ggml_tensor * text_mask;
    int64_t heads;
    int64_t visual_tokens;
    int64_t text_tokens;
    bool bf16;
};

struct TraceHostSpec {
    std::string name;
    std::vector<float> data;
    std::vector<int64_t> shape;
    const char * layout;
    const char * operation;
};

struct TraceNumericSpec {
    std::string name;
    ggml_tensor * tensor;
    std::vector<int64_t> shape;
    const char * layout;
    const char * operation;
    bool round_to_bf16;
};

// ggml's graph allocator is allowed to reuse intermediate buffers. Marking an
// operation as an output keeps its complete value alive until it is copied to
// the semantic trace after graph execution.
class TraceTensorRegistry {
public:
    explicit TraceTensorRegistry(TurboVlaTrace * writer) : writer_(writer) {}

    bool includes(const char * required_level = "op") const {
        return writer_ && writer_->includes(required_level);
    }

    void expand_graph(ggml_cgraph * graph) const {
        if (!graph) return;
        for (const TraceTensorSpec & spec : tensors_)
            ggml_build_forward_expand(graph, spec.tensor);
        for (const TraceLayerNormSpec & spec : layer_norms_)
            ggml_build_forward_expand(graph, spec.tensor);
        for (const TraceSoftmaxSpec & spec : softmaxes_)
            ggml_build_forward_expand(graph, spec.tensor);
        for (const TraceFusionSpec & spec : fusions_) {
            ggml_build_forward_expand(graph, spec.logits);
            if (spec.text_mask) ggml_build_forward_expand(graph, spec.text_mask);
        }
    }

    void add(const std::string & name, ggml_tensor * tensor,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation, const char * required_level = "op") {
        if (!writer_ || !writer_->includes(required_level) || !tensor) return;
        ggml_set_output(tensor);
        tensors_.push_back({name, tensor, std::vector<int64_t>(shape), layout,
                            operation, required_level, TraceTensorSpec::Transform::None,
                            nullptr});
    }

    void add_subtract_global_max(
             const std::string & name, ggml_tensor * tensor,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation, const char * required_level = "op",
             ggml_tensor * max_source = nullptr) {
        if (!writer_ || !writer_->includes(required_level) || !tensor) return;
        ggml_set_output(tensor);
        if (max_source) ggml_set_output(max_source);
        tensors_.push_back({name, tensor, std::vector<int64_t>(shape), layout,
                            operation, required_level,
                            TraceTensorSpec::Transform::SubtractGlobalMax,
                            max_source ? max_source : tensor});
    }

    void add_subtract_row_max(
             const std::string & name, ggml_tensor * tensor,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation, const char * required_level = "op") {
        if (!writer_ || !writer_->includes(required_level) || !tensor) return;
        ggml_set_output(tensor);
        tensors_.push_back({name, tensor, std::vector<int64_t>(shape), layout,
                            operation, required_level,
                            TraceTensorSpec::Transform::SubtractRowMax,
                            nullptr});
    }

    void add_layer_norm_diagnostics(
             const std::string & prefix, ggml_tensor * tensor,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * stat_layout, float eps, bool bf16) {
        if (!writer_ || !writer_->includes("exhaustive") || !tensor) return;
        add(prefix + ".input", tensor, shape, layout, "identity", "exhaustive");
        ggml_set_output(tensor);
        layer_norms_.push_back({prefix, tensor, std::vector<int64_t>(shape),
                                layout, stat_layout, eps, bf16});
    }

    void add_softmax_diagnostics(
             const std::string & prefix, ggml_tensor * tensor,
             std::initializer_list<int64_t> shape, const char * layout,
             bool bf16) {
        if (!writer_ || !writer_->includes("exhaustive") || !tensor) return;
        ggml_set_output(tensor);
        softmaxes_.push_back({prefix, tensor, std::vector<int64_t>(shape),
                              layout, bf16});
    }

    void add_fusion_diagnostics(
             const std::string & prefix, ggml_tensor * logits,
             ggml_tensor * text_mask, int64_t heads, int64_t visual_tokens,
             int64_t text_tokens, bool bf16) {
        if (!writer_ || !writer_->includes("exhaustive") || !logits) return;
        ggml_set_output(logits);
        if (text_mask) ggml_set_output(text_mask);
        fusions_.push_back({prefix, logits, text_mask, heads, visual_tokens,
                            text_tokens, bf16});
    }

    void add_host_f32(const std::string & name,
             std::initializer_list<float> data,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation) {
        if (!writer_ || !writer_->includes("exhaustive")) return;
        hosts_.push_back({name,std::vector<float>(data),std::vector<int64_t>(shape),
                          layout,operation});
    }

    void add_numeric(const std::string & name, ggml_tensor * tensor,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation, bool round_to_bf16=true) {
        if (!writer_ || !writer_->includes("exhaustive") || !tensor) return;
        numerics_.push_back({name,tensor,std::vector<int64_t>(shape),layout,
                             operation,round_to_bf16});
    }

    void dump(ggml_backend_t backend) const {
        if (!writer_) return;
        for (const TraceTensorSpec & spec : tensors_) {
            if (spec.transform == TraceTensorSpec::Transform::SubtractGlobalMax) {
                writer_->tensor_f32_subtract_global_max(
                    spec.name, backend, spec.tensor, spec.transform_source,
                    spec.shape, spec.layout, spec.operation,
                    spec.required_level);
            } else if (spec.transform == TraceTensorSpec::Transform::SubtractRowMax) {
                writer_->tensor_f32_subtract_row_max(
                    spec.name, backend, spec.tensor, spec.shape, spec.layout,
                    spec.operation, spec.required_level);
            } else {
                writer_->tensor_f32(spec.name, backend, spec.tensor, spec.shape,
                                    spec.layout, spec.operation,
                                    spec.required_level);
            }
        }
        for (const TraceLayerNormSpec & spec : layer_norms_) {
            writer_->tensor_f32_layer_norm_diagnostics(
                spec.prefix, backend, spec.tensor, spec.shape, spec.layout,
                spec.stat_layout, spec.eps, spec.bf16, "exhaustive");
        }
        for (const TraceSoftmaxSpec & spec : softmaxes_) {
            writer_->tensor_f32_softmax_diagnostics(
                spec.prefix, backend, spec.tensor, spec.shape, spec.layout,
                spec.bf16, "exhaustive");
        }
        for (const TraceFusionSpec & spec : fusions_) {
            writer_->tensor_f32_fusion_diagnostics(
                spec.prefix, backend, spec.logits, spec.text_mask,
                spec.heads, spec.visual_tokens, spec.text_tokens,
                spec.bf16, "exhaustive");
        }
        for (const TraceHostSpec & spec : hosts_) {
            writer_->f32(spec.name,spec.data.data(),spec.data.size(),spec.shape,
                         spec.layout,spec.operation,"exhaustive");
        }
        for (const TraceNumericSpec & spec : numerics_) {
            writer_->tensor_numeric_f32(
                spec.name,backend,spec.tensor,spec.shape,spec.layout,
                spec.operation,spec.round_to_bf16,"exhaustive");
        }
    }

private:
    TurboVlaTrace * writer_;
    std::vector<TraceTensorSpec> tensors_;
    std::vector<TraceLayerNormSpec> layer_norms_;
    std::vector<TraceSoftmaxSpec> softmaxes_;
    std::vector<TraceFusionSpec> fusions_;
    std::vector<TraceHostSpec> hosts_;
    std::vector<TraceNumericSpec> numerics_;
};

static float round_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

static ggml_tensor * heads(ggml_context * c, ggml_tensor * x, int64_t hd,
                           int64_t nh) {
    return ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, x, hd, nh, x->ne[1]),
                                     0, 2, 1, 3));
}

static ggml_tensor * values(ggml_context * c, ggml_tensor * x, int64_t hd,
                            int64_t nh) {
    return ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, x, hd, nh, x->ne[1]),
                                     1, 2, 0, 3));
}

static ggml_tensor * merge_heads(ggml_context * c, ggml_tensor * x, int64_t dim,
                                 int64_t length) {
    return ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)),
                           dim, length);
}

static ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k,
                               ggml_tensor * v, int64_t dim, int64_t nh,
                               ggml_tensor * mask = nullptr) {
    const int64_t hd = dim / nh;
    ggml_tensor * qh = heads(c, q, hd, nh);
    ggml_tensor * kh = heads(c, k, hd, nh);
    ggml_tensor * vh = values(c, v, hd, nh);
    ggml_tensor * score = ggml_mul_mat(c, kh, qh);
    ggml_mul_mat_set_prec(score, GGML_PREC_F32);
    ggml_tensor * prob = ggml_soft_max_ext(c, score, mask,
                                           1.0f / std::sqrt((float) hd), 0.0f);
    return merge_heads(c, ggml_mul_mat(c, vh, prob), dim, q->ne[1]);
}

static ggml_tensor * traced_attention(ggml_context * c, ggml_tensor * q,
                                      ggml_tensor * k, ggml_tensor * v,
                                      int64_t dim, int64_t nh,
                                      bool round,
                                      ggml_tensor * mask,
                                      TraceTensorRegistry * trace,
                                      const std::string & prefix) {
    const int64_t hd = dim / nh;
    const int64_t nq = q->ne[1];
    const int64_t nk = k->ne[1];
    const bool encoder_self_attention = prefix.rfind("interaction.", 0) == 0;
    const char * logits_layout = encoder_self_attention ? "B,H,N,N" : "B,H,Q,K";
    const char * context_heads_layout = encoder_self_attention ? "B,H,N,Dh" : "B,H,Q,Dh";
    const char * context_layout = encoder_self_attention ? "B,N,D" : "B,Q,D";
    if (trace) {
        trace->add(prefix + ".q_linear", q, {1,nq,dim}, "B,N,D", "linear_projection");
        trace->add(prefix + ".k_linear", k, {1,nk,dim}, "B,N,D", "linear_projection");
        trace->add(prefix + ".v_linear", v, {1,nk,dim}, "B,N,D", "linear_projection");
    }
    ggml_tensor * qh = heads(c, q, hd, nh);
    ggml_tensor * kh = heads(c, k, hd, nh);
    ggml_tensor * vh = values(c, v, hd, nh);
    // ggml keeps values in the matmul-friendly [Dh,N,H] physical order.  The
    // PyTorch semantic trace is [B,H,N,Dh], so materialize a trace-only
    // canonical branch from the pre-head value projection.
    ggml_tensor * vh_trace = trace && trace->includes()
        ? heads(c, v, hd, nh) : nullptr;
    if (trace) {
        trace->add(prefix + ".q_heads", qh, {1,nh,nq,hd}, "B,H,N,Dh", "reshape_permute", "exhaustive");
        trace->add(prefix + ".k_heads", kh, {1,nh,nk,hd}, "B,H,N,Dh", "reshape_permute", "exhaustive");
        trace->add(prefix + ".v_heads", vh_trace, {1,nh,nk,hd}, "B,H,N,Dh", "reshape_permute", "exhaustive");
    }
    ggml_tensor * score_mat = autocast_matmul(c, kh, qh, round);
    ggml_tensor * score=bf16_round(c,score_mat,round);
    ggml_tensor * logits = bf16_round(c,ggml_scale(c, score, 1.0f / std::sqrt((float) hd)),round);
    if (trace) trace->add(prefix + ".logits", logits, {1,nh,nq,nk},
                          logits_layout, "matmul_scale");
    ggml_tensor * masked_logits = mask ? ggml_add(c, logits, mask) : logits;
    if (trace) trace->add(prefix + ".masked_logits", masked_logits,
                          {1,nh,nq,nk}, logits_layout, "masked_fill");
    ggml_tensor * mask_trace = trace && trace->includes("exhaustive") && !mask
        ? ggml_scale(c, logits, 0.0f) : nullptr;
    if (trace && !mask) trace->add(prefix + ".mask", mask_trace,
        {1,nh,nq,nk}, logits_layout, "mask", "exhaustive");
    ggml_tensor * prob = pytorch_softmax(c, logits, mask, round);
    if (trace) trace->add_softmax_diagnostics(
        prefix + ".softmax", masked_logits, {1,nh,nq,nk}, logits_layout, round);
    if (trace) trace->add(prefix + ".softmax.probs", prob, {1,nh,nq,nk},
                          logits_layout, "scale_mask_softmax");
    ggml_tensor * context_heads = bf16_round(c,autocast_matmul(c, vh, prob, round),round);
    ggml_tensor * merged = merge_heads(c, context_heads, dim, nq);
    // merged is a reshape view over a temporary contiguous tensor.  Pinning
    // the view alone does not keep that backing allocation alive until the
    // post-graph dump, so copy it into a dedicated trace output.
    ggml_tensor * merged_trace = trace && trace->includes()
        ? ggml_dup(c, merged) : nullptr;
    if (trace) {
        trace->add(prefix + ".context_heads", context_heads, {1,nh,nq,hd},
                   context_heads_layout, "attention_value_matmul");
        trace->add(prefix + ".context_merged", merged_trace, {1,nq,dim},
                   context_layout, "merge_heads");
    }
    return merged;
}

static ggml_tensor * rotate_half(ggml_context * c, ggml_tensor * x, int64_t hd) {
    const int64_t n = x->ne[1], h = x->ne[2], half = hd / 2;
    ggml_tensor * xc = ggml_cont(c, x);
    ggml_tensor * a = ggml_cont(c, ggml_view_3d(c, xc, half, n, h,
                                                xc->nb[1], xc->nb[2], 0));
    ggml_tensor * b = ggml_cont(c, ggml_view_3d(c, xc, half, n, h,
                                                xc->nb[1], xc->nb[2],
                                                half * xc->nb[0]));
    return ggml_concat(c, ggml_scale(c, b, -1.0f), a, 0);
}

static ggml_tensor * rope_2d(ggml_context * c, ggml_tensor * x,
                             ggml_tensor * cos_t, ggml_tensor * sin_t, int64_t hd,
                             bool round, bool fused) {
    ggml_tensor * co = ggml_reshape_3d(c, cos_t, hd, x->ne[1], 1);
    ggml_tensor * si = ggml_reshape_3d(c, sin_t, hd, x->ne[1], 1);
    if (fused) {
        ggml_tensor * y=ggml_add(c,x,co);
        y->src[2]=si;
        std::memcpy(y->op_params,&turbovla_rope_magic,sizeof(turbovla_rope_magic));
        const int32_t head_dim=(int32_t)hd;
        std::memcpy((char *)y->op_params+sizeof(int32_t),&head_dim,sizeof(head_dim));
        return y;
    }
    // PyTorch executes each BF16 elementwise operation separately here:
    // q*cos, rotate_half(q)*sin, then their sum.  Rounding only the final
    // expression materially changes the following attention logits.
    ggml_tensor * direct = bf16_round(c, ggml_mul(c, x, co), round);
    ggml_tensor * rotated = bf16_round(c,
        ggml_mul(c, rotate_half(c, x, hd), si), round);
    return bf16_round(c, ggml_add(c, direct, rotated), round);
}

static ggml_tensor * rope_2d_from_linear(ggml_context * c,ggml_tensor * x,
                                         ggml_tensor * cos_t,
                                         ggml_tensor * sin_t,int64_t hd,
                                         int64_t heads_n) {
    ggml_tensor * y=ggml_new_tensor_3d(c,GGML_TYPE_F32,hd,x->ne[1],heads_n);
    y->op=GGML_OP_ADD;
    y->src[0]=x;
    y->src[1]=ggml_reshape_3d(c,cos_t,hd,x->ne[1],1);
    y->src[2]=ggml_reshape_3d(c,sin_t,hd,x->ne[1],1);
    std::memcpy(y->op_params,&turbovla_rope_linear_magic,
                sizeof(turbovla_rope_linear_magic));
    const int32_t head_dim=(int32_t)hd;
    std::memcpy((char *)y->op_params+sizeof(int32_t),&head_dim,sizeof(head_dim));
    return y;
}

static ggml_tensor * weight_slice(ggml_context * c, ggml_tensor * w, int64_t dim,
                                  int part) {
    return ggml_view_2d(c, w, dim, dim, w->nb[1], (size_t) part * dim * w->nb[1]);
}

static ggml_tensor * bias_slice(ggml_context * c, ggml_tensor * b, int64_t dim,
                                int part) {
    return ggml_view_1d(c, b, dim, (size_t) part * dim * b->nb[0]);
}

static ggml_tensor * packed_attention(ggml_context * c, ggml_tensor * query,
                                      ggml_tensor * memory, ggml_tensor * iw,
                                      ggml_tensor * ib, ggml_tensor * ow,
                                      ggml_tensor * ob, int64_t dim, int64_t nh,
                                      bool round,
                                      ggml_tensor * mask = nullptr,
                                      TraceTensorRegistry * trace = nullptr,
                                      const std::string & prefix = {}) {
    ggml_tensor * q = bf16_round(c,linear(c, weight_slice(c, iw, dim, 0), bias_slice(c, ib, dim, 0), query),round);
    ggml_tensor * k = bf16_round(c,linear(c, weight_slice(c, iw, dim, 1), bias_slice(c, ib, dim, 1), memory),round);
    ggml_tensor * v = bf16_round(c,linear(c, weight_slice(c, iw, dim, 2), bias_slice(c, ib, dim, 2), memory),round);
    ggml_tensor * merged = trace
        ? traced_attention(c, q, k, v, dim, nh, round, mask, trace, prefix)
        : attention(c, q, k, v, dim, nh, mask);
    ggml_tensor * output = bf16_round(c,linear(c, ow, ob, merged),round);
    const char * output_layout = prefix.rfind("interaction.", 0) == 0
        ? "B,N,D" : "B,Q,D";
    if (trace) trace->add(prefix + ".output_projection", output,
                          {1,query->ne[1],dim}, output_layout, "linear_projection");
    return output;
}

static void dump_f32(const char * dir, const char * name, const float * data, size_t n) {
    if (!dir || !*dir) {
        return;
    }
    std::filesystem::create_directories(dir);
    std::ofstream f(std::filesystem::path(dir) / name, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(n * sizeof(float)));
}

static std::string gguf_tensor_name(const std::string & original) {
    static const std::pair<const char *, const char *> prefixes[] = {
        {"text_encoder.bert.embeddings.", "t.e."},
        {"text_encoder.bert.encoder.layer.", "t.b."},
        {"text_encoder.bert.pooler.dense.", "t.pool."},
        {"text_encoder.text_projection.", "t.p."},
        {"vision_encoder.backbone.embeddings.", "v.e."},
        {"vision_encoder.backbone.layer.", "v.b."},
        {"vision_encoder.backbone.norm.", "v.n."},
        {"vision_projection.", "vp."},
        {"vision_language_interaction.text_layers.", "i.t."},
        {"vision_language_interaction.fusion_layers.", "i.f."},
        {"action_head.state_projection.", "a.s."},
        {"action_head.decoder.action_queries.", "a.q."},
        {"action_head.decoder.decoder.layers.", "a.d."},
        {"action_head.decoder.action_projection.layers.", "a.p."},
    };
    std::string name=original;
    for(const auto & p:prefixes){const size_t n=std::strlen(p.first);if(name.compare(0,n,p.first)==0){name=p.second+name.substr(n);break;}}
    if(name.size()>=7&&name.compare(name.size()-7,7,".weight")==0)name.replace(name.size()-7,7,".w");
    else if(name.size()>=5&&name.compare(name.size()-5,5,".bias")==0)name.replace(name.size()-5,5,".b");
    return name;
}

} // namespace

struct TurboVlaModelArch final : public ModelArchBase {
    TurboVlaModelArch() : ModelArchBase(Arch::TURBOVLA) {}
    ~TurboVlaModelArch() override {
#ifdef VLA_TURBOVLA_CUDNN
        turbovla_patch_embed_cuda_destroy(patch_cuda);
#endif
        if (vision_alloc) ggml_gallocr_free(vision_alloc);
        if (main_alloc) ggml_gallocr_free(main_alloc);
        if (vision_ctx) ggml_free(vision_ctx);
        if (main_ctx) ggml_free(main_ctx);
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend) ggml_backend_free(backend);
    }

    ggml_backend_t backend = nullptr;
    bool is_cuda = false;
    ggml_context * ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_type mt = GGML_TYPE_BF16;
    int n_threads = default_cpu_threads();
    uint64_t forward_index = 0;
    ggml_context * vision_ctx = nullptr;
    ggml_context * main_ctx = nullptr;
    ggml_gallocr_t vision_alloc = nullptr;
    ggml_gallocr_t main_alloc = nullptr;
#ifdef VLA_TURBOVLA_CUDNN
    TurboVlaPatchCudaContext * patch_cuda = nullptr;
#endif

    static constexpr int64_t image_size = 256, patch = 16, npatch = 256;
    static constexpr int64_t vd = 768, vh = 12, vhd = 64, vi = 3072, vlayers = 12;
    static constexpr int64_t bd = 768, bh = 12, bhd = 64, bi = 3072, blayers = 12;
    static constexpr int64_t d = 256, fusion_layers_n = 6, fusion_inner = 1024;
    static constexpr int64_t views_n = 2, visual_n = views_n * npatch;
    static constexpr int64_t action_n = 12, action_d = 7;
    int64_t state_d = 8;
    int64_t text_length = 21;      // 0 = dynamic (runtime token count)
    int64_t text_max_length = 256;

    ggml_tensor * view_emb;
    ggml_tensor * word_emb, * pos_emb, * type_emb, * bert_ew, * bert_eb;
    ggml_tensor * text_pw, * text_pb;
    std::vector<BertLayerW> bert;

    ggml_tensor * cls, * regs, * patch_w, * patch_b;
    std::vector<DinoLayerW> vision;
    ggml_tensor * vpnw, * vpnb, * vp1w, * vp1b, * vp2w, * vp2b, * vpsw;
    ggml_tensor * vponw, * vponb;

    std::vector<FusionLayerW> fusion;
    ggml_tensor * state_pos, * state_nw, * state_nb, * state_w1, * state_b1;
    ggml_tensor * state_w2, * state_b2, * state_ow, * state_ob;
    ggml_tensor * action_queries;
    std::vector<DecoderLayerW> decoder;
    ggml_tensor * action_w1, * action_b1, * action_w2, * action_b2, * action_w3, * action_b3;

    std::vector<float> predict(const Inputs & in) override;
};

std::unique_ptr<ModelArchBase> turbovla_create(const std::string & mmproj_path,
                                               const std::string & ckpt_path,
                                               const std::string &) {
    if (!mmproj_path.empty())
        std::printf("vla(turbovla): mmproj '%s' ignored; GGUF already contains BERT and DINOv3\n",
                    mmproj_path.c_str());
    auto m = std::make_unique<TurboVlaModelArch>();
    m->mt = std::getenv("VLA_TURBOVLA_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;
    const bool f32_matmul_weights = m->mt == GGML_TYPE_F32 ||
        std::getenv("VLA_TURBOVLA_F32_MATMUL") != nullptr;

    gguf_reader g("turbovla");
    if (!g.open(ckpt_path) || !g.has("turbovla.architecture")) {
        std::fprintf(stderr, "vla(turbovla): expected a TurboVLA combined GGUF\n");
        return nullptr;
    }

    if (g.has("turbovla.state_dim"))
        m->state_d = (int64_t)g.u32("turbovla.state_dim");
    if (g.has("turbovla.text_length"))
        m->text_length = (int64_t)g.u32("turbovla.text_length");
    if (g.has("turbovla.text_max_length"))
        m->text_max_length = (int64_t)g.u32("turbovla.text_max_length");
    if (m->state_d < 1 || m->text_length < 0 || m->text_max_length < 1 ||
        (m->text_length > 0 && m->text_length > m->text_max_length)) {
        std::fprintf(stderr,
                     "vla(turbovla): invalid state/text dimensions in GGUF metadata\n");
        return nullptr;
    }

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) {
        m->is_cuda = true;
        std::printf("vla(turbovla): backend = CUDA (device 0)\n");
    }
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) std::printf("vla(turbovla): backend = Metal\n");
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) return nullptr;
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(turbovla): backend = CPU (%d threads)\n", m->n_threads);
    }

    m->ctx_weights = ggml_init({64u * 1024u * 1024u, nullptr, true});
    ggml_context * wctx = m->ctx_weights;
    bool ok = true;
    auto mk = [&](const std::string & name, ggml_type prefer) -> ggml_tensor * {
        const std::string stored=gguf_tensor_name(name);
        const ggml_tensor * src = g.meta(stored.c_str());
        if (!src) {
            std::fprintf(stderr, "vla(turbovla): missing tensor %s\n", name.c_str());
            ok = false;
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(wctx, g.resident_type(src, prefer),
                                          ggml_n_dims(src), src->ne);
        ggml_set_name(t, stored.c_str());
        return t;
    };
    auto mm = [&](const std::string & n) {
        return mk(n, f32_matmul_weights ? GGML_TYPE_F32 : m->mt);
    };
    auto f32 = [&](const std::string & n) { return mk(n, GGML_TYPE_F32); };
    auto lb = [&](const std::string & n) {
        return mk(n, m->is_cuda && m->mt == GGML_TYPE_BF16 && !f32_matmul_weights
                         ? GGML_TYPE_BF16 : GGML_TYPE_F32);
    };

    m->view_emb = f32("view_embedding");
    const std::string be = "text_encoder.bert.embeddings.";
    m->word_emb = mm(be + "word_embeddings.weight");
    m->pos_emb = mm(be + "position_embeddings.weight");
    m->type_emb = mm(be + "token_type_embeddings.weight");
    m->bert_ew = f32(be + "LayerNorm.weight"); m->bert_eb = f32(be + "LayerNorm.bias");
    m->bert.resize(m->blayers);
    for (int i = 0; i < m->blayers; ++i) {
        auto & x = m->bert[i];
        const std::string p = "text_encoder.bert.encoder.layer." + std::to_string(i) + ".";
        x.wq=mm(p+"attention.self.query.weight"); x.bq=lb(p+"attention.self.query.bias");
        x.wk=mm(p+"attention.self.key.weight");   x.bk=lb(p+"attention.self.key.bias");
        x.wv=mm(p+"attention.self.value.weight"); x.bv=lb(p+"attention.self.value.bias");
        x.wo=mm(p+"attention.output.dense.weight"); x.bo=lb(p+"attention.output.dense.bias");
        x.n1w=f32(p+"attention.output.LayerNorm.weight"); x.n1b=f32(p+"attention.output.LayerNorm.bias");
        x.upw=mm(p+"intermediate.dense.weight"); x.upb=lb(p+"intermediate.dense.bias");
        x.downw=mm(p+"output.dense.weight"); x.downb=lb(p+"output.dense.bias");
        x.n2w=f32(p+"output.LayerNorm.weight"); x.n2b=f32(p+"output.LayerNorm.bias");
    }
    m->text_pw=mm("text_encoder.text_projection.weight");
    m->text_pb=lb("text_encoder.text_projection.bias");

    const std::string ve = "vision_encoder.backbone.";
    m->cls=f32(ve+"embeddings.cls_token"); m->regs=f32(ve+"embeddings.register_tokens");
    // PyTorch autocast supplies BF16 weights to cuDNN for this convolution.
    // The portable ggml fallback retains its more accurate F32 weight path.
#ifdef VLA_TURBOVLA_CUDNN
    m->patch_w=(m->is_cuda && m->mt==GGML_TYPE_BF16)
        ? mk(ve+"embeddings.patch_embeddings.weight",GGML_TYPE_BF16)
        : f32(ve+"embeddings.patch_embeddings.weight");
#else
    m->patch_w=f32(ve+"embeddings.patch_embeddings.weight");
#endif
    m->patch_b=f32(ve+"embeddings.patch_embeddings.bias");
    m->vision.resize(m->vlayers);
    for (int i = 0; i < m->vlayers; ++i) {
        auto & x=m->vision[i]; const std::string p=ve+"layer."+std::to_string(i)+".";
        x.n1w=f32(p+"norm1.weight"); x.n1b=f32(p+"norm1.bias");
        x.n2w=f32(p+"norm2.weight"); x.n2b=f32(p+"norm2.bias");
        x.wq=mm(p+"attention.q_proj.weight"); x.bq=lb(p+"attention.q_proj.bias");
        x.wk=mm(p+"attention.k_proj.weight");
        x.wv=mm(p+"attention.v_proj.weight"); x.bv=lb(p+"attention.v_proj.bias");
        x.wo=mm(p+"attention.o_proj.weight"); x.bo=lb(p+"attention.o_proj.bias");
        x.ls1=f32(p+"layer_scale1.lambda1"); x.ls2=f32(p+"layer_scale2.lambda1");
        x.upw=mm(p+"mlp.up_proj.weight"); x.upb=lb(p+"mlp.up_proj.bias");
        x.downw=mm(p+"mlp.down_proj.weight"); x.downb=lb(p+"mlp.down_proj.bias");
    }
    m->vpnw=f32("vision_projection.input_norm.weight"); m->vpnb=f32("vision_projection.input_norm.bias");
    m->vp1w=mm("vision_projection.mlp.0.weight"); m->vp1b=lb("vision_projection.mlp.0.bias");
    m->vp2w=mm("vision_projection.mlp.3.weight"); m->vp2b=lb("vision_projection.mlp.3.bias");
    m->vpsw=mm("vision_projection.skip.weight");
    m->vponw=f32("vision_projection.output_norm.weight"); m->vponb=f32("vision_projection.output_norm.bias");

    m->fusion.resize(m->fusion_layers_n);
    for (int i=0;i<m->fusion_layers_n;++i) {
        auto & x=m->fusion[i];
        const std::string fp="vision_language_interaction.fusion_layers."+std::to_string(i)+".";
        const std::string tp="vision_language_interaction.text_layers."+std::to_string(i)+".";
        x.gamma_v=f32(fp+"gamma_v"); x.gamma_l=f32(fp+"gamma_l");
        x.nvw=f32(fp+"layer_norm_v.weight"); x.nvb=f32(fp+"layer_norm_v.bias");
        x.nlw=f32(fp+"layer_norm_l.weight"); x.nlb=f32(fp+"layer_norm_l.bias");
        x.qvw=mm(fp+"attn.v_proj.weight"); x.qvb=lb(fp+"attn.v_proj.bias");
        x.ktw=mm(fp+"attn.l_proj.weight"); x.ktb=lb(fp+"attn.l_proj.bias");
        x.vvw=mm(fp+"attn.values_v_proj.weight"); x.vvb=lb(fp+"attn.values_v_proj.bias");
        x.vtw=mm(fp+"attn.values_l_proj.weight"); x.vtb=lb(fp+"attn.values_l_proj.bias");
        x.ovw=mm(fp+"attn.out_v_proj.weight"); x.ovb=lb(fp+"attn.out_v_proj.bias");
        x.otw=mm(fp+"attn.out_l_proj.weight"); x.otb=lb(fp+"attn.out_l_proj.bias");
        x.inw=mm(tp+"self_attn.in_proj_weight"); x.inb=lb(tp+"self_attn.in_proj_bias");
        x.iow=mm(tp+"self_attn.out_proj.weight"); x.iob=lb(tp+"self_attn.out_proj.bias");
        x.f1w=mm(tp+"linear1.weight"); x.f1b=lb(tp+"linear1.bias");
        x.f2w=mm(tp+"linear2.weight"); x.f2b=lb(tp+"linear2.bias");
        x.t1w=f32(tp+"norm1.weight"); x.t1b=f32(tp+"norm1.bias");
        x.t2w=f32(tp+"norm2.weight"); x.t2b=f32(tp+"norm2.bias");
    }

    const std::string sp="action_head.state_projection.";
    m->state_pos=f32(sp+"position"); m->state_nw=f32(sp+"net.0.weight"); m->state_nb=f32(sp+"net.0.bias");
    m->state_w1=mm(sp+"net.1.weight"); m->state_b1=lb(sp+"net.1.bias");
    m->state_w2=mm(sp+"net.4.weight"); m->state_b2=lb(sp+"net.4.bias");
    m->state_ow=f32(sp+"output_norm.weight"); m->state_ob=f32(sp+"output_norm.bias");
    m->action_queries=f32("action_head.decoder.action_queries.weight");
    m->decoder.resize(3);
    for (int i=0;i<3;++i) {
        auto & x=m->decoder[i];
        const std::string p="action_head.decoder.decoder.layers."+std::to_string(i)+".";
        x.siw=mm(p+"self_attn.in_proj_weight"); x.sib=lb(p+"self_attn.in_proj_bias");
        x.sow=mm(p+"self_attn.out_proj.weight"); x.sob=lb(p+"self_attn.out_proj.bias");
        x.ciw=mm(p+"multihead_attn.in_proj_weight"); x.cib=lb(p+"multihead_attn.in_proj_bias");
        x.cow=mm(p+"multihead_attn.out_proj.weight"); x.cob=lb(p+"multihead_attn.out_proj.bias");
        x.f1w=mm(p+"linear1.weight"); x.f1b=lb(p+"linear1.bias");
        x.f2w=mm(p+"linear2.weight"); x.f2b=lb(p+"linear2.bias");
        x.n1w=f32(p+"norm1.weight"); x.n1b=f32(p+"norm1.bias");
        x.n2w=f32(p+"norm2.weight"); x.n2b=f32(p+"norm2.bias");
        x.n3w=f32(p+"norm3.weight"); x.n3b=f32(p+"norm3.bias");
    }
    const std::string ap="action_head.decoder.action_projection.layers.";
    m->action_w1=mm(ap+"0.weight"); m->action_b1=lb(ap+"0.bias");
    m->action_w2=mm(ap+"1.weight"); m->action_b2=lb(ap+"1.bias");
    m->action_w3=mm(ap+"2.weight"); m->action_b3=lb(ap+"2.bias");
    if (!ok) return nullptr;

    m->weight_buf=ggml_backend_alloc_ctx_tensors(wctx,m->backend);
    if (!m->weight_buf) { std::fprintf(stderr,"vla(turbovla): weight allocation failed\n"); return nullptr; }
    for (ggml_tensor * t=ggml_get_first_tensor(wctx); t; t=ggml_get_next_tensor(wctx,t)) {
        auto bytes=g.read_convert(ggml_get_name(t),t->type);
        if (bytes.size()!=ggml_nbytes(t)) {
            std::fprintf(stderr,"vla(turbovla): failed loading %s\n",ggml_get_name(t)); return nullptr;
        }
        ggml_backend_tensor_set(t,bytes.data(),0,bytes.size());
    }
#ifdef VLA_TURBOVLA_CUDNN
    if (m->is_cuda && m->mt == GGML_TYPE_BF16 && m->patch_w->type == GGML_TYPE_BF16) {
        m->patch_cuda=turbovla_patch_embed_cuda_create((const float *)m->patch_b->data);
        if (!m->patch_cuda) {
            std::fprintf(stderr,"vla(turbovla): failed creating persistent cuDNN patch context\n");
            return nullptr;
        }
    }
#endif
    m->cfg.n_img=m->visual_n;
    m->cfg.n_lang=m->text_length > 0 ? m->text_length : m->text_max_length;
    m->cfg.n_state=2;
    m->cfg.n_suffix=m->action_n; m->cfg.num_steps=m->action_n;
    m->cfg.max_state_dim=m->state_d; m->cfg.real_state_dim=m->state_d;
    m->cfg.max_action_dim=m->action_d; m->cfg.real_action_dim=m->action_d;
    m->cfg.hidden=m->d;
    std::printf("vla(turbovla): weights resident %.2f GiB (%s%s)\n",
                ggml_backend_buffer_get_size(m->weight_buf)/(1024.0*1024.0*1024.0),
                m->mt==GGML_TYPE_F32?"F32":"BF16",
                f32_matmul_weights && m->mt==GGML_TYPE_BF16?", F32 matmul":"");
    return m;
}

namespace {

static ggml_tensor * dino_block(ggml_context * c, const DinoLayerW & w,
                                ggml_tensor * x, ggml_tensor * co, ggml_tensor * si,
                                bool round, TraceTensorRegistry * trace,
                                const std::string & prefix) {
    constexpr int64_t d=768, n=261, h=12, hd=64;
    if(trace)trace->add(prefix+".input",x,{1,n,d},"B,N,D","layer_input","layer");
    if(trace)trace->add_layer_norm_diagnostics(
        prefix+".norm_1",x,{1,n,d},"B,N,D","B,N,1",1e-5f,round);
    ggml_tensor * z=bf16_round(c,ln(c,x,w.n1w,w.n1b,1e-5f,round),round);
    if(trace)trace->add(prefix+".norm_1.output",z,{1,n,d},"B,N,D","layer_norm","exhaustive");
    const bool exhaustive_trace=trace&&trace->includes("exhaustive");
    const bool fused_attention=round && turbovla_cuda_fused_linear &&
        !trace->includes() &&
        std::getenv("VLA_TURBOVLA_DISABLE_FLASH_ATTN") == nullptr;
    ggml_tensor * q_mat=diagnostic_matmul(c,w.wq,z,exhaustive_trace);
    ggml_tensor * k_mat=diagnostic_matmul(c,w.wk,z,exhaustive_trace);
    ggml_tensor * v_mat=diagnostic_matmul(c,w.wv,z,exhaustive_trace);
    ggml_tensor * q=bf16_round(c,linear(c,w.wq,w.bq,z),round);
    ggml_tensor * k=bf16_round(c,linear(c,w.wk,nullptr,z),round);
    ggml_tensor * v=bf16_round(c,linear(c,w.wv,w.bv,z),round);
    if(trace){
        ggml_tensor * qkv_mat=exhaustive_trace
            ? ggml_concat(c,ggml_concat(c,q_mat,k_mat,0),v_mat,0):nullptr;
        ggml_tensor * qkv_bias_add=exhaustive_trace
            ? ggml_concat(c,ggml_concat(c,q,k,0),v,0):nullptr;
        trace->add(prefix+".attn.qkv_linear.matmul",qkv_mat,{1,n,3*d},"B,N,3D","matmul","exhaustive");
        trace->add(prefix+".attn.qkv_linear.bias_add",qkv_bias_add,{1,n,3*d},"B,N,3D","add","exhaustive");
        trace->add(prefix+".attn.q",q,{1,n,d},"B,N,D","linear_projection");
        trace->add(prefix+".attn.k",k,{1,n,d},"B,N,D","linear_projection");
        trace->add(prefix+".attn.v",v,{1,n,d},"B,N,D","linear_projection");
    }
    ggml_tensor * q_heads=fused_attention?nullptr:heads(c,q,hd,h);
    ggml_tensor * k_heads=fused_attention?nullptr:heads(c,k,hd,h);
    ggml_tensor * qh=fused_attention
        ? rope_2d_from_linear(c,q,co,si,hd,h)
        : rope_2d(c,q_heads,co,si,hd,round,false);
    ggml_tensor * kh=fused_attention
        ? rope_2d_from_linear(c,k,co,si,hd,h)
        : rope_2d(c,k_heads,co,si,hd,round,false);
    ggml_tensor * vh=!fused_attention?values(c,v,hd,h):nullptr;
    ggml_tensor * vh_trace=trace&&trace->includes()?heads(c,v,hd,h):nullptr;
    if(trace){
        trace->add(prefix+".attn.q_heads",qh,{1,h,n,hd},"B,H,N,Dh","reshape_rope");
        trace->add(prefix+".attn.k_heads",kh,{1,h,n,hd},"B,H,N,Dh","reshape_rope");
        trace->add(prefix+".attn.v_heads",vh_trace,{1,h,n,hd},"B,H,N,Dh","reshape_permute");
    }
    // The production PyTorch backbone dispatches this whole subgraph through
    // scaled_dot_product_attention.  Use ggml's fused CUDA attention when no
    // semantic trace is requested; exhaustive tracing retains the explicit
    // logits/softmax branch because those are user-visible comparison points.
    ggml_tensor * raw_score_mat=nullptr;
    ggml_tensor * raw_score=nullptr;
    ggml_tensor * score=nullptr;
    ggml_tensor * prob=nullptr;
    ggml_tensor * ctx=nullptr;
    ggml_tensor * merged=nullptr;
    if(fused_attention){
        ggml_tensor * v_fa=ggml_permute(
            c,ggml_reshape_3d(c,v,hd,h,n),0,2,1,3);
        ggml_tensor * fa=ggml_flash_attn_ext(
            c,qh,kh,v_fa,nullptr,1.0f/std::sqrt((float)hd),0.0f,0.0f);
        ggml_flash_attn_ext_set_prec(fa,GGML_PREC_F32);
        // Keep the fused attention output in BF16 so the following output
        // projection consumes it directly instead of round-to-F32 and then
        // converting the same values back to BF16 in cuBLAS.
        ctx=ggml_cast(c,fa,GGML_TYPE_BF16);
        merged=ggml_reshape_2d(c,ctx,d,n);
    }
    if(!fused_attention||trace->includes()){
        raw_score_mat=autocast_matmul(c,kh,qh,round);
        raw_score=bf16_round(c,raw_score_mat,round);
        score=bf16_round(c,ggml_scale(c,raw_score,1.0f/std::sqrt((float)hd)),round);
        prob=pytorch_softmax(c,score,nullptr,round);
    }
    if(trace&&trace->includes()){
        trace->add_host_f32(prefix+".attn.scale",{1.0f/std::sqrt((float)hd)},
                            {},"","constant");
        trace->add(prefix+".attn.logits",raw_score,{1,h,n,n},"B,H,N,N","matmul");
        trace->add(prefix+".attn.scaled_logits",score,{1,h,n,n},"B,H,N,N","scale");
        trace->add(prefix+".attn.softmax.probs",prob,{1,h,n,n},"B,H,N,N","softmax");
        trace->add_softmax_diagnostics(
            prefix+".attn.softmax",score,{1,h,n,n},"B,H,N,N",round);
    }
    if(!fused_attention){
        ctx=bf16_round(c,autocast_matmul(c,vh,prob,round),round);
        merged=merge_heads(c,ctx,d,n);
    }
    ggml_tensor * merged_trace=trace&&trace->includes()?ggml_dup(c,merged):nullptr;
    ggml_tensor * a=bf16_round(c,linear(c,w.wo,w.bo,merged),round);
    if(trace){
        trace->add(prefix+".attn.context_heads",ctx,{1,h,n,hd},"B,H,N,Dh","attention_value_matmul");
        trace->add(prefix+".attn.context_merged",merged_trace,{1,n,d},"B,N,D","merge_heads");
        trace->add(prefix+".attn.output_projection",a,{1,n,d},"B,N,D","linear_projection");
    }
    ggml_tensor * residual_1_left=x;
    ggml_tensor * scaled1=nullptr;
    if(fused_attention){
        x=ggml_add(c,x,a);
        x->src[2]=w.ls1;
        std::memcpy(x->op_params,&turbovla_scaled_residual_magic,
                    sizeof(turbovla_scaled_residual_magic));
        if(trace->includes())
            scaled1=bf16_round(c,ggml_mul(c,a,w.ls1),round);
    }else{
        scaled1=bf16_round(c,ggml_mul(c,a,w.ls1),round);
        x=bf16_round(c,ggml_add(c,x,scaled1),round);
    }
    if(trace){
        trace->add(prefix+".layer_scale_1.input",a,{1,n,d},"B,N,D","identity","exhaustive");
        trace->add(prefix+".layer_scale_1.output",scaled1,{1,n,d},"B,N,D","multiply");
        trace->add(prefix+".residual_1.left",residual_1_left,{1,n,d},"B,N,D","identity");
        trace->add(prefix+".residual_1.right",scaled1,{1,n,d},"B,N,D","identity");
        trace->add(prefix+".residual_1.sum",x,{1,n,d},"B,N,D","residual_add");
    }
    if(trace)trace->add_layer_norm_diagnostics(
        prefix+".norm_2",x,{1,n,d},"B,N,D","B,N,1",1e-5f,round);
    z=bf16_round(c,ln(c,x,w.n2w,w.n2b,1e-5f,round),round);
    if(trace)trace->add(prefix+".norm_2.output",z,{1,n,d},"B,N,D","layer_norm","exhaustive");
    ggml_tensor * up_input=z;
    ggml_tensor * up_mat=diagnostic_matmul(c,w.upw,up_input,exhaustive_trace);
    z=bf16_round(c,linear(c,w.upw,w.upb,up_input),round);
    if(trace)trace->add(prefix+".mlp.linear_1.matmul",up_mat,{1,n,3072},"B,N,D","matmul","exhaustive");
    if(trace)trace->add(prefix+".mlp.linear_1.bias_add",z,{1,n,3072},"B,N,D","linear_projection","exhaustive");
    if(trace)trace->add(prefix+".mlp.activation.input",z,{1,n,3072},"B,N,F","activation","exhaustive");
    z=bf16_round(c,ggml_gelu_erf(c,z),round);
    if(trace)trace->add(prefix+".mlp.activation.output",z,{1,n,3072},"B,N,F","gelu");
    ggml_tensor * down_input=z;
    ggml_tensor * down_mat=diagnostic_matmul(c,w.downw,down_input,exhaustive_trace);
    z=bf16_round(c,linear(c,w.downw,w.downb,down_input),round);
    if(trace)trace->add(prefix+".mlp.linear_2.matmul",down_mat,{1,n,d},"B,N,D","matmul","exhaustive");
    if(trace)trace->add(prefix+".mlp.linear_2.bias_add",z,{1,n,d},"B,N,D","linear_projection","exhaustive");
    if(trace)trace->add(prefix+".layer_scale_2.input",z,{1,n,d},"B,N,D","identity","exhaustive");
    ggml_tensor * scaled2=nullptr;
    if(!fused_attention||trace->includes())
        scaled2=bf16_round(c,ggml_mul(c,z,w.ls2),round);
    if(trace)trace->add(prefix+".layer_scale_2.output",scaled2,{1,n,d},"B,N,D","multiply");
    ggml_tensor * residual_2_left=x;
    ggml_tensor * output=nullptr;
    if(fused_attention){
        output=ggml_add(c,x,z);
        output->src[2]=w.ls2;
        std::memcpy(output->op_params,&turbovla_scaled_residual_magic,
                    sizeof(turbovla_scaled_residual_magic));
    }else{
        output=bf16_round(c,ggml_add(c,x,scaled2),round);
    }
    if(trace){
        trace->add(prefix+".residual_2.left",residual_2_left,{1,n,d},"B,N,D","identity");
        trace->add(prefix+".residual_2.right",scaled2,{1,n,d},"B,N,D","identity");
        trace->add(prefix+".residual_2.sum",output,{1,n,d},"B,N,D","residual_add");
    }
    return output;
}

static void normalize_image(const ImageView & image, std::vector<float> & out) {
    constexpr int64_t s=256;
    constexpr int64_t pixels=s*s;
    static const float mean[3]={0.485f,0.456f,0.406f};
    static const float stdv[3]={0.229f,0.224f,0.225f};
    out.resize(3*pixels);
    if(image.format==PixelFormat::U8){
        const auto * src=(const uint8_t *)image.data;
        const float scale[3]={1.0f/(255.0f*stdv[0]),1.0f/(255.0f*stdv[1]),
                              1.0f/(255.0f*stdv[2])};
        const float bias[3]={-mean[0]/stdv[0],-mean[1]/stdv[1],-mean[2]/stdv[2]};
        for(int64_t i=0;i<pixels;++i){
            out[i]=src[3*i]*scale[0]+bias[0];
            out[pixels+i]=src[3*i+1]*scale[1]+bias[1];
            out[2*pixels+i]=src[3*i+2]*scale[2]+bias[2];
        }
    }else{
        const auto * src=(const float *)image.data;
        const float inv[3]={1.0f/stdv[0],1.0f/stdv[1],1.0f/stdv[2]};
        for(int64_t i=0;i<pixels;++i){
            out[i]=(src[3*i]-mean[0])*inv[0];
            out[pixels+i]=(src[3*i+1]-mean[1])*inv[1];
            out[2*pixels+i]=(src[3*i+2]-mean[2])*inv[2];
        }
    }
}

} // namespace

std::vector<float> TurboVlaModelArch::predict(const Inputs & in) {
    using clock=std::chrono::steady_clock; const auto start=clock::now(); stats=Stats{};
    const int64_t max_input_text = text_length > 0 ? text_length : text_max_length;
    if (!in.images || in.n_images!=views_n || !in.lang_tokens || in.n_lang<1 ||
        in.n_lang>max_input_text || !in.state) {
        std::fprintf(stderr,
                     "vla(turbovla): need exactly 2 images, 1..%lld BERT tokens "
                     "and a %lld-D normalized state\n",
                     (long long)max_input_text, (long long)state_d);
        return {};
    }
    for (int i=0;i<views_n;++i) if (!view_is_side(in.images[i].data,in.images[i].w,in.images[i].h,image_size)) {
        std::fprintf(stderr,"vla(turbovla): image %d must be 256x256\n",i); return {};
    }
    for (int i=0;i<in.n_lang;++i) if (in.lang_tokens[i]<0 || in.lang_tokens[i]>=30522) {
        std::fprintf(stderr,"vla(turbovla): token %d out of BERT vocabulary\n",in.lang_tokens[i]); return {};
    }
    const int64_t BL=in.n_lang;
    // Mark BF16 linears for the CUDA F32-accumulator + fused BF16 bias/output
    // epilogue.  This also forces thin matrices away from ggml's MMV kernels,
    // whose dispatch cannot consume the attached bias.
    turbovla_cuda_fused_linear = is_cuda && mt == GGML_TYPE_BF16 &&
        bert[0].wq->type == GGML_TYPE_BF16 &&
        std::getenv("VLA_TURBOVLA_DISABLE_FUSED_LINEAR") == nullptr;
    turbovla_cuda_pytorch_layer_norm = is_cuda && mt == GGML_TYPE_BF16;
    // LIBERO checkpoints use a fixed padded interaction layout. Checkpoints
    // trained with tokenizer padding="longest" store text_length=0, for which
    // batch-size-one inference naturally uses the runtime token count.
    const int64_t L=text_length > 0 ? text_length : BL;
    const uint64_t this_forward=forward_index++;
    const char * trace_root=std::getenv("VLA_TURBOVLA_TRACE_ROOT");
    uint64_t trace_max=1;
    if(const char * value=std::getenv("VLA_TURBOVLA_TRACE_MAX_FORWARDS"))
        trace_max=std::strtoull(value,nullptr,10);
    TurboVlaTrace trace((trace_max==0||this_forward<trace_max)?trace_root:nullptr,this_forward);
    ggml_tensor * visual_device_output=nullptr;
    std::vector<float> visual_host;
    if(trace.active()||std::getenv("VLA_TURBOVLA_DUMP_DIR"))
        visual_host.resize((size_t)d*visual_n);
    {
        const auto tv=clock::now();
        std::vector<float> im;
        std::vector<float> pixel_values_host((size_t)views_n*3*image_size*image_size);
        for(int view=0;view<views_n;++view){
            normalize_image(in.images[view],im);
            std::copy(im.begin(),im.end(),pixel_values_host.begin()+(size_t)view*im.size());
        }
        const auto tv_normalized=clock::now();
        bool use_cudnn_patch=false;
#ifdef VLA_TURBOVLA_CUDNN
        use_cudnn_patch=patch_cuda!=nullptr;
#endif
        if(!vision_ctx) vision_ctx=ggml_init({64u*1024u*1024u,nullptr,true});
        else ggml_reset(vision_ctx);
        ggml_context * c=vision_ctx;
        TraceTensorRegistry vision_ops(&trace);
        ggml_tensor * co=ggml_new_tensor_2d(c,GGML_TYPE_F32,vhd,npatch+5); ggml_set_input(co);
        ggml_tensor * si=ggml_new_tensor_2d(c,GGML_TYPE_F32,vhd,npatch+5); ggml_set_input(si);
        std::vector<ggml_tensor*> pixels(views_n),patch_inputs(views_n,nullptr),pixel_values_trace(views_n),projected(views_n);
        std::vector<ggml_tensor*> projection_skips(views_n),projection_activations(views_n);
        std::vector<ggml_tensor*> projection_residuals(views_n),projection_positioned(views_n);
        std::vector<ggml_tensor*> final_tokens_trace(views_n),patch_tokens_trace(views_n);
        std::vector<std::vector<ggml_tensor*>> vision_layers_trace(views_n);
        ggml_tensor * co_patches=ggml_view_2d(c,co,vhd,npatch,co->nb[1],5*co->nb[1]);
        ggml_tensor * si_patches=ggml_view_2d(c,si,vhd,npatch,si->nb[1],5*si->nb[1]);
        ggml_tensor * rope_position=ggml_concat(c,co_patches,si_patches,2);
        for (int view=0;view<views_n;++view) {
            pixels[view]=ggml_new_tensor_3d(c,GGML_TYPE_F32,image_size,image_size,3); ggml_set_input(pixels[view]);
            ggml_tensor * pixel_values=bf16_round(c,pixels[view],mt==GGML_TYPE_BF16);
            pixel_values_trace[view]=pixel_values;
            ggml_tensor * pt=nullptr;
            if(use_cudnn_patch){
                patch_inputs[view]=ggml_new_tensor_2d(c,GGML_TYPE_F32,vd,npatch);
                ggml_set_input(patch_inputs[view]);
                pt=patch_inputs[view];
            }else{
                ggml_tensor * conv=bf16_round(c,ggml_conv_2d(c,patch_w,pixel_values,patch,patch,0,0,1,1),mt==GGML_TYPE_BF16);
                pt=ggml_cont(c,ggml_transpose(c,ggml_reshape_2d(c,conv,npatch,vd)));
                pt=bf16_round(c,ggml_add(c,pt,patch_b),mt==GGML_TYPE_BF16);
            }
            ggml_tensor * conv_trace=ggml_cont(c,ggml_transpose(c,pt));
            ggml_tensor * prefix=ggml_concat(c,ggml_reshape_2d(c,cls,vd,1),ggml_reshape_2d(c,regs,vd,4),1);
            ggml_tensor * x=ggml_concat(c,prefix,pt,1);
            const std::string view_prefix="vision.view_"+std::to_string(view);
            vision_ops.add(view_prefix+".patch_embed.input",pixel_values,{1,3,image_size,image_size},"B,C,H,W","input");
            vision_ops.add_numeric(view_prefix+".patch_embed.conv.weight",patch_w,
                                   {vd,3,patch,patch},"OUT,IN,KH,KW","parameter");
            vision_ops.add_numeric(view_prefix+".patch_embed.conv.bias",patch_b,
                                   {vd},"OUT","parameter");
            vision_ops.add(view_prefix+".patch_embed.conv.output",conv_trace,{1,vd,16,16},"B,D,PH,PW","conv2d");
            vision_ops.add(view_prefix+".patch_embed.flatten",conv_trace,{1,vd,npatch},"B,D,N","flatten","exhaustive");
            vision_ops.add(view_prefix+".patch_embed.transpose",pt,{1,npatch,vd},"B,N,D","transpose");
            vision_ops.add(view_prefix+".cls_token",cls,{1,1,vd},"B,1,D","parameter_expand");
            vision_ops.add(view_prefix+".register_tokens",regs,{1,4,vd},"B,R,D","parameter_expand");
            vision_ops.add(view_prefix+".position_embedding",rope_position,{2,npatch,vhd},"2,N,Dh","rope");
            vision_ops.add(view_prefix+".tokens_before_position",x,{1,npatch+5,vd},"B,P+N,D","concat","exhaustive");
            vision_ops.add(view_prefix+".tokens_after_position",x,{1,npatch+5,vd},"B,P+N,D","rope_deferred","exhaustive");
            vision_ops.add(view_prefix+".tokens_with_prefix",x,{1,npatch+5,vd},"B,P+N,D","concat","layer");
            for (size_t layer_index=0;layer_index<vision.size();++layer_index) {
                x=dino_block(c,vision[layer_index],x,co,si,mt==GGML_TYPE_BF16,
                             &vision_ops,view_prefix+".block_"+two_digit(layer_index));
                if(trace.includes_layer())vision_layers_trace[view].push_back(x);
            }
            // TurboVLA intentionally consumes outputs.hidden_states[-1], which
            // is the last block output before DINOv3's final backbone norm.
            final_tokens_trace[view]=x;
            ggml_tensor * final_prefix_trace=ggml_cont(c,ggml_view_2d(c,x,vd,5,x->nb[1],0));
            vision_ops.add(view_prefix+".prefix_tokens",final_prefix_trace,{1,5,vd},"B,P,D","slice");
            x=ggml_cont(c,ggml_view_2d(c,x,vd,npatch,x->nb[1],5*x->nb[1]));
            patch_tokens_trace[view]=x;
        }
        // PyTorch applies the projector to [B,V,N,D] in one operation, which
        // flattens to a single n=V*N cuBLAS GEMM.  Projecting each view with
        // separate n=N calls selects a different cuBLAS schedule and changes
        // a handful of BF16 rounding decisions.
        ggml_tensor * projection_input=ggml_concat(c,patch_tokens_trace[0],patch_tokens_trace[1],1);
        const bool projection_round=mt==GGML_TYPE_BF16;
        const bool projection_exhaustive=trace.includes_exhaustive();
        ggml_tensor * projection_skip_mat=diagnostic_matmul(
            c,vpsw,projection_input,projection_exhaustive);
        ggml_tensor * projection_skip=bf16_round(c,linear(c,vpsw,nullptr,projection_input),projection_round);
        vision_ops.add_layer_norm_diagnostics(
            "vision_projection.input_norm",projection_input,
            {1,views_n,npatch,vd},"B,V,N,Din","B,V,N,1in",1e-5f,projection_round);
        ggml_tensor * projection_input_norm=bf16_round(
            c,ln(c,projection_input,vpnw,vpnb,1e-5f,projection_round),projection_round);
        ggml_tensor * projection_mlp_1_mat=diagnostic_matmul(
            c,vp1w,projection_input_norm,projection_exhaustive);
        ggml_tensor * projection_mlp=bf16_round(c,linear(c,vp1w,vp1b,projection_input_norm),projection_round);
        ggml_tensor * projection_linear_1=projection_mlp;
        projection_mlp=bf16_round(c,ggml_gelu_erf(c,projection_mlp),projection_round);
        ggml_tensor * projection_activation=projection_mlp;
        ggml_tensor * projection_mlp_2_mat=diagnostic_matmul(
            c,vp2w,projection_mlp,projection_exhaustive);
        projection_mlp=bf16_round(c,linear(c,vp2w,vp2b,projection_mlp),projection_round);
        ggml_tensor * projection_linear_2=projection_mlp;
        ggml_tensor * projection_residual=bf16_round(c,ggml_add(c,projection_skip,projection_mlp),projection_round);
        vision_ops.add_layer_norm_diagnostics(
            "vision_projection.output_norm",projection_residual,
            {1,views_n,npatch,d},"B,V,N,D","B,V,N,1",1e-5f,projection_round);
        ggml_tensor * projection_norm=bf16_round(c,ln(c,projection_residual,vponw,vponb,1e-5f),projection_round);
        for (int view=0;view<views_n;++view) {
            ggml_tensor * p=ggml_cont(c,ggml_view_2d(c,projection_norm,d,npatch,
                                                     projection_norm->nb[1],
                                                     view*npatch*projection_norm->nb[1]));
            ggml_tensor * ve=ggml_cont(c,ggml_view_2d(c,view_emb,d,1,view_emb->nb[1],view*view_emb->nb[1]));
            projected[view]=bf16_round(c,ggml_add(c,p,ve),mt==GGML_TYPE_BF16);
            projection_positioned[view]=projected[view];
        }
        ggml_tensor * projection_position=ggml_concat(c,projection_positioned[0],projection_positioned[1],1);
        vision_ops.add("vision_projection.skip.matmul",projection_skip_mat,{1,views_n,npatch,d},"B,V,N,D","matmul","exhaustive");
        vision_ops.add("vision_projection.skip.output",projection_skip,{1,views_n,npatch,d},"B,V,N,D","linear");
        vision_ops.add("vision_projection.input_norm.output",projection_input_norm,
                       {1,views_n,npatch,vd},"B,V,N,Din","layer_norm","exhaustive");
        vision_ops.add("vision_projection.mlp.linear_1.bias_add",projection_linear_1,
                       {1,views_n,npatch,fusion_inner},"B,V,N,F","linear","exhaustive");
        vision_ops.add("vision_projection.mlp.linear_1.matmul",projection_mlp_1_mat,
                       {1,views_n,npatch,fusion_inner},"B,V,N,F","matmul","exhaustive");
        vision_ops.add("vision_projection.mlp.gelu.input",projection_linear_1,
                       {1,views_n,npatch,fusion_inner},"B,V,N,F","gelu","exhaustive");
        vision_ops.add("vision_projection.mlp.gelu.output",projection_activation,{1,views_n,npatch,fusion_inner},"B,V,N,F","gelu");
        vision_ops.add("vision_projection.mlp.linear_2.matmul",projection_mlp_2_mat,
                       {1,views_n,npatch,d},"B,V,N,D","matmul","exhaustive");
        vision_ops.add("vision_projection.mlp.linear_2.bias_add",projection_linear_2,
                       {1,views_n,npatch,d},"B,V,N,D","add","exhaustive");
        vision_ops.add("vision_projection.residual.left",projection_skip,
                       {1,views_n,npatch,d},"B,V,N,D","identity","exhaustive");
        vision_ops.add("vision_projection.residual.right",projection_linear_2,
                       {1,views_n,npatch,d},"B,V,N,D","identity","exhaustive");
        vision_ops.add("vision_projection.residual.sum",projection_residual,{1,views_n,npatch,d},"B,V,N,D","add");
        vision_ops.add("vision_projection.output_norm.output",projection_norm,
                       {1,views_n,npatch,d},"B,V,N,D","layer_norm","exhaustive");
        vision_ops.add("vision_projection.view_embedding",view_emb,{1,views_n,d},"1,V,D","parameter");
        vision_ops.add("vision_projection.position.before_add",projection_norm,
                       {1,views_n,npatch,d},"B,V,N,D","identity","exhaustive");
        vision_ops.add("vision_projection.position.after_add",projection_position,{1,views_n,npatch,d},"B,V,N,D","position_add");
        vision_ops.add("vision_projection.before_flatten",projection_position,
                       {1,views_n,npatch,d},"B,V,N,D","identity","exhaustive");
        ggml_tensor * out=ggml_concat(c,projected[0],projected[1],1); ggml_set_output(out);
        visual_device_output=out;
        ggml_tensor * patch_stacked_trace=nullptr;
        if(trace.active()){
            patch_stacked_trace=projection_input;
            ggml_set_output(patch_stacked_trace);
            for(int view=0;view<views_n;++view){
                ggml_set_output(pixel_values_trace[view]);
                ggml_set_output(final_tokens_trace[view]);
                ggml_set_output(patch_tokens_trace[view]);
                for(ggml_tensor * tensor:vision_layers_trace[view])ggml_set_output(tensor);
            }
        }
        ggml_cgraph * graph=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(graph,out);
        if(patch_stacked_trace)ggml_build_forward_expand(graph,patch_stacked_trace);
        if(trace.active())for(ggml_tensor * tensor:pixel_values_trace)
            ggml_build_forward_expand(graph,tensor);
        vision_ops.expand_graph(graph);
        if(!vision_alloc)
            vision_alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_t alloc=vision_alloc;
        if (!alloc || !ggml_gallocr_alloc_graph(alloc,graph)) {
            std::fprintf(stderr,"vla(turbovla): vision graph allocation failed\n"); return{};
        }
        const auto tv_graph_ready=clock::now();
        const size_t image_elements=(size_t)3*image_size*image_size;
        const size_t patch_elements=(size_t)npatch*vd;
#ifdef VLA_TURBOVLA_CUDNN
        if(use_cudnn_patch&&!turbovla_patch_embed_cuda_two_views(
               patch_cuda,pixel_values_host.data(),patch_w->data,
               (float *)patch_inputs[0]->data,(float *)patch_inputs[1]->data)){
            std::fprintf(stderr,"vla(turbovla): cuDNN patch projection failed\n");
            return{};
        }
#endif
        const auto tv_patch_ready=clock::now();
        for(int view=0;view<views_n;++view){
            // With the cuDNN patch path, the raw-pixel ggml tensor is only a
            // trace source.  It is not part of the production graph and hence
            // has no gallocr buffer when tracing is disabled.
            if(!use_cudnn_patch||trace.active())ggml_backend_tensor_set(pixels[view],
                pixel_values_host.data()+(size_t)view*image_elements,0,
                image_elements*sizeof(float));
            if(use_cudnn_patch&&std::getenv("VLA_TURBOVLA_DUMP_DIR")){
                std::vector<float> patch_host(patch_elements);
                ggml_backend_tensor_get(patch_inputs[view],patch_host.data(),0,
                                        patch_host.size()*sizeof(float));
                dump_f32(std::getenv("VLA_TURBOVLA_DUMP_DIR"),
                         ("vision.patch_embed.view_"+std::to_string(view)+".f32").c_str(),
                         patch_host.data(),patch_host.size());
            }
        }
        std::vector<float> cv((npatch+5)*vhd,1.0f),sv((npatch+5)*vhd,0.0f);
        for(int py=0;py<16;++py)for(int px=0;px<16;++px){int p=py*16+px;float yy=2.0f*((py+0.5f)/16.0f)-1.0f,xx=2.0f*((px+0.5f)/16.0f)-1.0f;
            float base[32];for(int j=0;j<16;++j){float inv=1.0f/std::pow(100.0f,j/16.0f);if(mt==GGML_TYPE_BF16)inv=round_bf16(inv);base[j]=2.0f*(float)M_PI*yy*inv;base[16+j]=2.0f*(float)M_PI*xx*inv;}
            for(int j=0;j<64;++j){float a=base[j%32];float co_value=std::cos(a),si_value=std::sin(a);if(mt==GGML_TYPE_BF16){co_value=round_bf16(co_value);si_value=round_bf16(si_value);}cv[(p+5)*64+j]=co_value;sv[(p+5)*64+j]=si_value;}}
        ggml_backend_tensor_set(co,cv.data(),0,ggml_nbytes(co)); ggml_backend_tensor_set(si,sv.data(),0,ggml_nbytes(si));
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS){std::fprintf(stderr,"vla(turbovla): vision compute failed\n");return{};}
        if(!visual_host.empty())
            ggml_backend_tensor_get(out,visual_host.data(),0,visual_host.size()*sizeof(float));
        const auto tv_compute_ready=clock::now();
        if(trace.active()){
            std::vector<float> pixel_values_model(pixel_values_host.size());
            for(int view=0;view<views_n;++view){
                const size_t offset=(size_t)view*3*image_size*image_size;
                ggml_backend_tensor_get(pixel_values_trace[view],pixel_values_model.data()+offset,0,3*image_size*image_size*sizeof(float));
                std::vector<uint8_t> rotated((size_t)3*image_size*image_size);
                if(in.images[view].format==PixelFormat::U8)
                    std::memcpy(rotated.data(),in.images[view].data,rotated.size());
                else for(size_t i=0;i<rotated.size();++i)
                    rotated[i]=(uint8_t)std::lround(std::clamp(((const float*)in.images[view].data)[i],0.0f,1.0f)*255.0f);
                trace.u8("input.rotated.view_"+std::to_string(view)+"_rgb_u8",rotated.data(),rotated.size(),{image_size,image_size,3},"H,W,C","rotate_180");
            }
            trace.f32("input.pixel_values_f32",pixel_values_host.data(),pixel_values_host.size(),{1,views_n,3,image_size,image_size},"B,V,C,H,W","stack");
            trace.f32("input.pixel_values_model_dtype",pixel_values_model.data(),pixel_values_model.size(),{1,views_n,3,image_size,image_size},"B,V,C,H,W","cast");
            for(int view=0;view<views_n;++view){
                trace.tensor_f32("vision.view_"+std::to_string(view)+".input",backend,pixel_values_trace[view],{1,3,image_size,image_size},"B,C,H,W","input");
                for(size_t layer=0;layer<vision_layers_trace[view].size();++layer)
                    trace.tensor_f32("vision.view_"+std::to_string(view)+".block_"+two_digit(layer)+".output",backend,vision_layers_trace[view][layer],{1,npatch+5,vd},"B,N,D","layer_output","layer");
                trace.tensor_f32("vision.view_"+std::to_string(view)+".tokens_with_prefix_final",backend,final_tokens_trace[view],{1,npatch+5,vd},"B,P+N,D","encoder_output");
                trace.tensor_f32("vision.view_"+std::to_string(view)+".patch_tokens",backend,patch_tokens_trace[view],{1,npatch,vd},"B,N,D","slice");
            }
            trace.tensor_f32("vision.patch_tokens_stacked",backend,patch_stacked_trace,{1,views_n,npatch,vd},"B,V,N,D","stack");
            trace.tensor_f32("vision_projection.input",backend,patch_stacked_trace,{1,views_n,npatch,vd},"B,V,N,Din","input");
            trace.f32("vision_projection.flattened",visual_host.data(),visual_host.size(),{1,visual_n,d},"B,VxN,D","flatten");
            vision_ops.dump(backend);
        }
        dump_f32(std::getenv("VLA_TURBOVLA_DUMP_DIR"),"vision.flattened.f32",visual_host.data(),visual_host.size());
        stats.ms_vision=std::chrono::duration<float,std::milli>(clock::now()-tv).count();
        if(std::getenv("VLA_TURBOVLA_PROFILE"))std::fprintf(stderr,
            "vla(turbovla): vision profile normalize=%.3f graph_build_alloc=%.3f patch=%.3f graph_compute_readback=%.3f cleanup=%.3f ms\n",
            std::chrono::duration<float,std::milli>(tv_normalized-tv).count(),
            std::chrono::duration<float,std::milli>(tv_graph_ready-tv_normalized).count(),
            std::chrono::duration<float,std::milli>(tv_patch_ready-tv_graph_ready).count(),
            std::chrono::duration<float,std::milli>(tv_compute_ready-tv_patch_ready).count(),
            std::chrono::duration<float,std::milli>(clock::now()-tv_compute_ready).count());
    }

    const auto ti=clock::now();
    if(!main_ctx) main_ctx=ggml_init({96u*1024u*1024u,nullptr,true});
    else ggml_reset(main_ctx);
    ggml_context * c=main_ctx;
    TraceTensorRegistry main_ops(&trace);
    ggml_tensor * ids=ggml_new_tensor_1d(c,GGML_TYPE_I32,BL);ggml_set_input(ids);
    ggml_tensor * pids=ggml_new_tensor_1d(c,GGML_TYPE_I32,BL);ggml_set_input(pids);
    ggml_tensor * zero=ggml_new_tensor_1d(c,GGML_TYPE_I32,1);ggml_set_input(zero);
    ggml_tensor * bmask=ggml_new_tensor_2d(c,GGML_TYPE_F32,BL,BL);ggml_set_input(bmask);
    ggml_tensor * cross_mask=ggml_new_tensor_2d(c,GGML_TYPE_F32,L,visual_n);ggml_set_input(cross_mask);
    ggml_tensor * self_mask=ggml_new_tensor_2d(c,GGML_TYPE_F32,L,L);ggml_set_input(self_mask);
    ggml_tensor * word=ggml_get_rows(c,word_emb,ids);
    ggml_tensor * position=ggml_get_rows(c,pos_emb,pids);
    ggml_tensor * token_type_lookup=ggml_get_rows(c,type_emb,zero);
    ggml_tensor * token_type=ggml_repeat(c,token_type_lookup,word);
    const bool model_bf16=mt==GGML_TYPE_BF16;
    // HuggingFace BertEmbeddings adds token type before absolute position.
    // Both adds execute in BF16 under the model dtype, so preserve the order
    // and the intermediate rounding (BF16 addition is not associative).
    ggml_tensor * text=bf16_round(c,ggml_add(c,word,token_type),model_bf16);
    text=bf16_round(c,ggml_add(c,text,position),model_bf16);
    main_ops.add("text.bert.embeddings.word",word,{1,BL,bd},"B,N,D","embedding_lookup");
    main_ops.add("text.bert.embeddings.position",position,{1,BL,bd},"B,N,D","embedding_lookup");
    main_ops.add("text.bert.embeddings.token_type",token_type,{1,BL,bd},"B,N,D","embedding_lookup");
    main_ops.add_numeric("text.bert.embeddings.norm.weight",bert_ew,{bd},"D","parameter");
    main_ops.add_numeric("text.bert.embeddings.norm.bias",bert_eb,{bd},"D","parameter");
    main_ops.add("text.bert.embeddings.sum_before_norm",text,{1,BL,bd},"B,N,D","add","exhaustive");
    main_ops.add_layer_norm_diagnostics(
        "text.bert.embeddings.norm",text,{1,BL,bd},"B,N,D","B,N,1",1e-12f,model_bf16);
    text=bf16_round(c,ln(c,text,bert_ew,bert_eb,1e-12f,model_bf16),model_bf16);
    main_ops.add("text.bert.embeddings.norm.output",text,{1,BL,bd},"B,N,D","layer_norm","exhaustive");
    main_ops.add("text.bert.embeddings.output",text,{1,BL,bd},"B,N,D","dropout","layer");
    std::vector<ggml_tensor*> bert_layers_trace;
    for(size_t layer_index=0;layer_index<bert.size();++layer_index){
        const auto & w=bert[layer_index];
        const std::string prefix="text.bert.layer_"+two_digit(layer_index);
        main_ops.add(prefix+".input",text,{1,BL,bd},"B,N,D","layer_input","layer");
        ggml_tensor * q_mat=ggml_mul_mat(c,w.wq,text),*k_mat=ggml_mul_mat(c,w.wk,text),*v_mat=ggml_mul_mat(c,w.wv,text);
        ggml_mul_mat_set_prec(q_mat,GGML_PREC_F32);ggml_mul_mat_set_prec(k_mat,GGML_PREC_F32);ggml_mul_mat_set_prec(v_mat,GGML_PREC_F32);
        ggml_tensor * q=turbovla_cuda_fused_linear
            ? bf16_round(c,linear(c,w.wq,w.bq,text),model_bf16)
            : bf16_round(c,ggml_add(c,q_mat,w.bq),model_bf16);
        ggml_tensor * k=turbovla_cuda_fused_linear
            ? bf16_round(c,linear(c,w.wk,w.bk,text),model_bf16)
            : bf16_round(c,ggml_add(c,k_mat,w.bk),model_bf16);
        ggml_tensor * v=turbovla_cuda_fused_linear
            ? bf16_round(c,linear(c,w.wv,w.bv,text),model_bf16)
            : bf16_round(c,ggml_add(c,v_mat,w.bv),model_bf16);
        main_ops.add(prefix+".attn.q_linear.matmul",q_mat,{1,BL,bd},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".attn.q_linear.bias_add",q,{1,BL,bd},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".attn.k_linear.matmul",k_mat,{1,BL,bd},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".attn.k_linear.bias_add",k,{1,BL,bd},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".attn.v_linear.matmul",v_mat,{1,BL,bd},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".attn.v_linear.bias_add",v,{1,BL,bd},"B,N,D","add","exhaustive");
        ggml_tensor * qh=heads(c,q,bhd,bh),*kh=heads(c,k,bhd,bh),*vh=values(c,v,bhd,bh);
        ggml_tensor * vh_trace=trace.includes_op()?heads(c,v,bhd,bh):nullptr;
        main_ops.add(prefix+".attn.q.reshape",q,{1,BL,bh,bhd},"B,N,H,Dh","reshape","exhaustive");
        main_ops.add(prefix+".attn.k.reshape",k,{1,BL,bh,bhd},"B,N,H,Dh","reshape","exhaustive");
        main_ops.add(prefix+".attn.v.reshape",v,{1,BL,bh,bhd},"B,N,H,Dh","reshape","exhaustive");
        main_ops.add(prefix+".attn.q.transpose",qh,{1,bh,BL,bhd},"B,H,N,Dh","reshape_transpose","exhaustive");
        main_ops.add(prefix+".attn.k.transpose",kh,{1,bh,BL,bhd},"B,H,N,Dh","reshape_transpose","exhaustive");
        main_ops.add(prefix+".attn.v.transpose",vh_trace,{1,bh,BL,bhd},"B,H,N,Dh","reshape_transpose","exhaustive");
        ggml_tensor * logits_mat=autocast_matmul(c,kh,qh,model_bf16);
        ggml_tensor * logits=bf16_round(c,logits_mat,model_bf16);
        ggml_tensor * scaled_logits=bf16_round(c,ggml_scale(c,logits,1.0f/std::sqrt((float)bhd)),model_bf16);
        ggml_tensor * masked_logits=ggml_add(c,scaled_logits,bmask);
        ggml_tensor * prob=pytorch_softmax(c,scaled_logits,bmask,model_bf16);
        ggml_tensor * k_transposed_trace=trace.includes_exhaustive()
            ? ggml_cont(c,ggml_transpose(c,kh)):nullptr;
        ggml_tensor * context_heads=bf16_round(c,autocast_matmul(c,vh,prob,model_bf16),model_bf16);
        ggml_tensor * context=merge_heads(c,context_heads,bd,BL);
        ggml_tensor * context_trace=trace.includes_op()?ggml_dup(c,context):nullptr;
        main_ops.add(prefix+".attn.qk_matmul",logits,{1,bh,BL,BL},"B,H,N,N","matmul");
        main_ops.add_host_f32(prefix+".attn.scale",{std::sqrt((float)bhd)},
                              {},"","constant");
        main_ops.add(prefix+".attn.k_transposed",k_transposed_trace,{1,bh,bhd,BL},"B,H,Dh,N","transpose","exhaustive");
        main_ops.add(prefix+".attn.scaled_logits",scaled_logits,{1,bh,BL,BL},"B,H,N,N","divide");
        main_ops.add(prefix+".attn.masked_logits",masked_logits,{1,bh,BL,BL},"B,H,N,N","masked_fill");
        main_ops.add(prefix+".attn.softmax.probs",prob,{1,bh,BL,BL},"B,H,N,N","softmax");
        main_ops.add_softmax_diagnostics(
            prefix+".attn.softmax",masked_logits,{1,bh,BL,BL},"B,H,N,N",model_bf16);
        main_ops.add(prefix+".attn.context_heads",context_heads,{1,bh,BL,bhd},"B,H,N,Dh","matmul");
        main_ops.add(prefix+".attn.context_transpose",context_trace,{1,BL,bh,bhd},"B,N,H,Dh","transpose","exhaustive");
        main_ops.add(prefix+".attn.context_merged",context_trace,{1,BL,bd},"B,N,D","reshape");
        ggml_tensor * a_mat=ggml_mul_mat(c,w.wo,context);ggml_mul_mat_set_prec(a_mat,GGML_PREC_F32);
        ggml_tensor * a=turbovla_cuda_fused_linear
            ? bf16_round(c,linear(c,w.wo,w.bo,context),model_bf16)
            : bf16_round(c,ggml_add(c,a_mat,w.bo),model_bf16);
        main_ops.add(prefix+".attn.output.matmul",a_mat,{1,BL,bd},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".attn.output.bias_add",a,{1,BL,bd},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".attn.output",a,{1,BL,bd},"B,N,D","linear","layer");
        main_ops.add(prefix+".residual_1.left",text,{1,BL,bd},"B,N,D","identity","exhaustive");
        main_ops.add(prefix+".residual_1.right",a,{1,BL,bd},"B,N,D","identity","exhaustive");
        ggml_tensor * residual_1=bf16_round(c,ggml_add(c,text,a),model_bf16);
        main_ops.add(prefix+".residual_1.sum",residual_1,{1,BL,bd},"B,N,D","add");
        main_ops.add_layer_norm_diagnostics(
            prefix+".norm_1",residual_1,{1,BL,bd},"B,N,D","B,N,1",1e-12f,model_bf16);
        text=bf16_round(c,ln(c,residual_1,w.n1w,w.n1b,1e-12f,model_bf16),model_bf16);
        main_ops.add(prefix+".norm_1.output",text,{1,BL,bd},"B,N,D","layer_norm","exhaustive");
        ggml_tensor * f1_mat=ggml_mul_mat(c,w.upw,text);ggml_mul_mat_set_prec(f1_mat,GGML_PREC_F32);
        ggml_tensor * f1=turbovla_cuda_fused_linear
            ? bf16_round(c,linear(c,w.upw,w.upb,text),model_bf16)
            : bf16_round(c,ggml_add(c,f1_mat,w.upb),model_bf16);
        ggml_tensor * activation=bf16_round(c,ggml_gelu_erf(c,f1),model_bf16);
        ggml_tensor * f2_mat=ggml_mul_mat(c,w.downw,activation);ggml_mul_mat_set_prec(f2_mat,GGML_PREC_F32);
        ggml_tensor * f=turbovla_cuda_fused_linear
            ? bf16_round(c,linear(c,w.downw,w.downb,activation),model_bf16)
            : bf16_round(c,ggml_add(c,f2_mat,w.downb),model_bf16);
        main_ops.add(prefix+".ffn.linear_1.matmul",f1_mat,{1,BL,bi},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".ffn.linear_1.bias_add",f1,{1,BL,bi},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".ffn.gelu.input",f1,{1,BL,bi},"B,N,F","gelu","exhaustive");
        main_ops.add(prefix+".ffn.gelu.output",activation,{1,BL,bi},"B,N,F","gelu");
        main_ops.add(prefix+".ffn.linear_2.matmul",f2_mat,{1,BL,bd},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".ffn.linear_2.bias_add",f,{1,BL,bd},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".residual_2.left",text,{1,BL,bd},"B,N,D","identity","exhaustive");
        main_ops.add(prefix+".residual_2.right",f,{1,BL,bd},"B,N,D","identity","exhaustive");
        ggml_tensor * residual_2=bf16_round(c,ggml_add(c,text,f),model_bf16);
        main_ops.add(prefix+".residual_2.sum",residual_2,{1,BL,bd},"B,N,D","add");
        main_ops.add_layer_norm_diagnostics(
            prefix+".norm_2",residual_2,{1,BL,bd},"B,N,D","B,N,1",1e-12f,model_bf16);
        text=bf16_round(c,ln(c,residual_2,w.n2w,w.n2b,1e-12f,model_bf16),model_bf16);
        main_ops.add(prefix+".norm_2.output",text,{1,BL,bd},"B,N,D","layer_norm","exhaustive");
        if(trace.includes_layer())bert_layers_trace.push_back(text);
    }
    ggml_tensor * bert_unpadded_trace=text;
    ggml_tensor * bert_pad=nullptr;
    if(BL<L){bert_pad=ggml_new_tensor_2d(c,GGML_TYPE_F32,bd,L-BL);ggml_set_input(bert_pad);text=ggml_concat(c,text,bert_pad,1);}
    ggml_tensor * bert_padded_trace=text;
    main_ops.add("text.projection.input",bert_padded_trace,{1,L,bd},"B,N,D","input");
    ggml_tensor * text_projection_mat=diagnostic_matmul(
        c,text_pw,text,trace.includes_exhaustive());
    text=bf16_round(c,linear(c,text_pw,text_pb,text),mt==GGML_TYPE_BF16);
    main_ops.add("text.projection.matmul",text_projection_mat,{1,L,d},"B,N,D","matmul","exhaustive");
    main_ops.add("text.projection.bias_add",text,{1,L,d},"B,N,D","add","exhaustive");
    main_ops.add("text.projection.before_zero_fill",text,{1,L,d},"B,N,D","identity","exhaustive");
    ggml_tensor * text_zero_fill_mask=nullptr;
    if(trace.includes_exhaustive()){
        text_zero_fill_mask=ggml_new_tensor_1d(c,GGML_TYPE_F32,L);
        ggml_set_input(text_zero_fill_mask);
    }
    main_ops.add("text.projection.zero_fill_mask",text_zero_fill_mask,{1,L},"B,N","mask","exhaustive");
    ggml_tensor * visual=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,visual_n);ggml_set_input(visual);
    ggml_tensor * visual_input_debug=visual;
    ggml_tensor * bert_text_debug=text;
    ggml_tensor * first_fusion_visual_debug=nullptr,*first_fusion_text_debug=nullptr;
    std::vector<ggml_tensor*> fusion_visual_inputs,fusion_text_inputs;
    std::vector<ggml_tensor*> fusion_visual_outputs,fusion_text_outputs;
    int fusion_index=0;
    for(const auto & w:fusion){
        const std::string prefix="interaction.layer_"+two_digit((size_t)fusion_index);
        if(trace.active()){fusion_visual_inputs.push_back(visual);fusion_text_inputs.push_back(text);}
        ggml_tensor * visual_residual_source=visual,* text_residual_source=text;
        main_ops.add(prefix+".visual.residual_source",visual_residual_source,{1,visual_n,d},"B,V,D","identity");
        main_ops.add(prefix+".text.residual_source",text_residual_source,{1,L,d},"B,L,D","identity");
        // Each source feeds three independent normalized branches below.
        ggml_set_output(visual); ggml_set_output(text);
        // Build independent normalized/projection branches. Keeping each
        // branch single-consumer avoids backend buffer aliasing in this
        // bidirectional DAG while preserving the exact PyTorch equations.
        ggml_tensor * vn_v=bf16_round(c,ln(c,visual,w.nvw,w.nvb,1e-5f,model_bf16),model_bf16);
        ggml_tensor * tn_v=bf16_round(c,ln(c,text,w.nlw,w.nlb,1e-5f,model_bf16),model_bf16);
        main_ops.add_layer_norm_diagnostics(
            prefix+".visual.norm",visual,{1,visual_n,d},"B,N,D","B,N,1",1e-5f,model_bf16);
        main_ops.add_layer_norm_diagnostics(
            prefix+".text.norm",text,{1,L,d},"B,N,D","B,N,1",1e-5f,model_bf16);
        const bool fusion_exhaustive=trace.includes_exhaustive();
        ggml_tensor * qv_mat=diagnostic_matmul(c,w.qvw,vn_v,fusion_exhaustive);
        ggml_tensor * kt_mat=diagnostic_matmul(c,w.ktw,tn_v,fusion_exhaustive);
        ggml_tensor * vv_mat=diagnostic_matmul(c,w.vvw,vn_v,fusion_exhaustive);
        ggml_tensor * vt_mat=diagnostic_matmul(c,w.vtw,tn_v,fusion_exhaustive);
        ggml_tensor * qv=bf16_round(c,linear(c,w.qvw,w.qvb,vn_v),model_bf16);
        ggml_tensor * kt=bf16_round(c,linear(c,w.ktw,w.ktb,tn_v),model_bf16);
        ggml_tensor * vv=bf16_round(c,linear(c,w.vvw,w.vvb,vn_v),model_bf16);
        ggml_tensor * vt=bf16_round(c,linear(c,w.vtw,w.vtb,tn_v),model_bf16);
        main_ops.add(prefix+".visual.norm.output",vn_v,{1,visual_n,d},"B,N,D","layer_norm","exhaustive");
        main_ops.add(prefix+".text.norm.output",tn_v,{1,L,d},"B,N,D","layer_norm","exhaustive");
        main_ops.add(prefix+".cross.q_visual.bias_add",qv,{1,visual_n,fusion_inner},"B,N,D","linear_projection","exhaustive");
        main_ops.add(prefix+".cross.k_text.bias_add",kt,{1,L,fusion_inner},"B,N,D","linear_projection","exhaustive");
        main_ops.add(prefix+".cross.v_visual.bias_add",vv,{1,visual_n,fusion_inner},"B,N,D","linear_projection","exhaustive");
        main_ops.add(prefix+".cross.v_text.bias_add",vt,{1,L,fusion_inner},"B,N,D","linear_projection","exhaustive");
        main_ops.add(prefix+".cross.q_visual.matmul",qv_mat,{1,visual_n,fusion_inner},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".cross.k_text.matmul",kt_mat,{1,L,fusion_inner},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".cross.v_visual.matmul",vv_mat,{1,visual_n,fusion_inner},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".cross.v_text.matmul",vt_mat,{1,L,fusion_inner},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".cross.q_visual.reshape",qv,{1,visual_n,4,256},"B,N,H,Dh","reshape","exhaustive");
        main_ops.add(prefix+".cross.k_text.reshape",kt,{1,L,4,256},"B,N,H,Dh","reshape","exhaustive");
        main_ops.add(prefix+".cross.v_visual.reshape",vv,{1,visual_n,4,256},"B,N,H,Dh","reshape","exhaustive");
        main_ops.add(prefix+".cross.v_text.reshape",vt,{1,L,4,256},"B,N,H,Dh","reshape","exhaustive");
        ggml_tensor * qvh_trace=trace.includes_op()?heads(c,qv,256,4):nullptr;
        ggml_tensor * qv_scaled=bf16_round(c,ggml_scale(c,qv,1.0f/16.0f),model_bf16);
        ggml_tensor * qvh=heads(c,qv_scaled,256,4),*kth=heads(c,kt,256,4),*vth=values(c,vt,256,4);
        ggml_tensor * vvh=values(c,vv,256,4);
        ggml_tensor * vth_trace=trace.includes_op()?heads(c,vt,256,4):nullptr;
        ggml_tensor * vvh_trace=trace.includes_op()?heads(c,vv,256,4):nullptr;
        ggml_tensor * vscore_mat=autocast_matmul(c,kth,qvh,model_bf16);
        ggml_tensor * vscore=bf16_round(c,vscore_mat,model_bf16);
        main_ops.add_fusion_diagnostics(
            prefix,vscore,cross_mask,4,visual_n,L,model_bf16);
        ggml_tensor * vscore_masked=trace.includes_op()
            ? bf16_round(c,ggml_add(c,vscore,cross_mask),model_bf16) : nullptr;
        ggml_tensor * vprob=pytorch_softmax(c,vscore,cross_mask,model_bf16,1);
        ggml_tensor * vctx_heads=bf16_round(c,autocast_matmul(c,vth,vprob,model_bf16),model_bf16);
        ggml_tensor * vctx=merge_heads(c,vctx_heads,fusion_inner,visual_n);
        ggml_tensor * vctx_trace=trace.includes_op()?ggml_dup(c,vctx):nullptr;
        ggml_tensor * dv_mat=diagnostic_matmul(c,w.ovw,vctx,fusion_exhaustive);
        ggml_tensor * dv=bf16_round(c,linear(c,w.ovw,w.ovb,vctx),model_bf16);
        main_ops.add(prefix+".cross.q_visual.transpose",qvh_trace,{1,4,visual_n,256},"B,H,N,Dh","reshape_transpose","exhaustive");
        main_ops.add(prefix+".cross.k_text.transpose",kth,{1,4,L,256},"B,H,N,Dh","reshape_transpose","exhaustive");
        main_ops.add(prefix+".cross.v_visual.transpose",vvh_trace,{1,4,visual_n,256},"B,H,N,Dh","reshape_transpose","exhaustive");
        main_ops.add(prefix+".cross.v_text.transpose",vth_trace,{1,4,L,256},"B,H,N,Dh","reshape_transpose","exhaustive");
        main_ops.add(prefix+".cross.visual_to_text.qk_matmul",vscore,{4,visual_n,L},"BH,V,L","bmm");
        main_ops.add_subtract_global_max(
            prefix+".cross.visual_to_text.masked_logits", vscore_masked,
            {4,visual_n,L}, "BH,V,L", "masked_fill", "op", vscore);
        main_ops.add(prefix+".cross.visual_to_text.softmax.probs",vprob,{4,visual_n,L},"BH,V,L","softmax");
        main_ops.add(prefix+".cross.visual_to_text.context_heads",vctx_heads,{4,visual_n,256},"BH,V,Dh","bmm");
        main_ops.add(prefix+".cross.visual_to_text.context_transpose",vctx_trace,{1,visual_n,4,256},"B,V,H,Dh","transpose","exhaustive");
        main_ops.add(prefix+".cross.visual_to_text.context_merged",vctx_trace,{1,visual_n,fusion_inner},"B,V,D","reshape");
        main_ops.add(prefix+".cross.visual_to_text.output.matmul",dv_mat,{1,visual_n,d},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".cross.visual_to_text.output.bias_add",dv,{1,visual_n,d},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".cross.visual_to_text.output",dv,{1,visual_n,d},"B,V,D","linear","layer");
        ggml_tensor * tscore=ggml_cont(c,ggml_transpose(c,vscore));
        ggml_tensor * tprob=pytorch_softmax(c,tscore,nullptr,model_bf16,2);
        ggml_tensor * tctx_heads=bf16_round(c,autocast_matmul(c,vvh,tprob,model_bf16),model_bf16);
        ggml_tensor * tctx=merge_heads(c,tctx_heads,fusion_inner,L);
        ggml_tensor * tctx_trace=trace.includes_op()?ggml_dup(c,tctx):nullptr;
        ggml_tensor * dt_mat=diagnostic_matmul(c,w.otw,tctx,fusion_exhaustive);
        ggml_tensor * dt=bf16_round(c,linear(c,w.otw,w.otb,tctx),model_bf16);
        main_ops.add_subtract_global_max(
            prefix+".cross.text_to_visual.logits_transposed",tscore,
            {4,L,visual_n},"BH,L,V","transpose","exhaustive");
        main_ops.add_subtract_row_max(
            prefix+".cross.text_to_visual.masked_logits",tscore,
            {4,L,visual_n},"BH,L,V","masked_fill");
        main_ops.add(prefix+".cross.text_to_visual.softmax.probs",tprob,{4,L,visual_n},"BH,L,V","softmax");
        main_ops.add(prefix+".cross.text_to_visual.context_heads",tctx_heads,{4,L,256},"BH,L,Dh","bmm");
        main_ops.add(prefix+".cross.text_to_visual.context_transpose",tctx_trace,{1,L,4,256},"B,L,H,Dh","transpose","exhaustive");
        main_ops.add(prefix+".cross.text_to_visual.context_merged",tctx_trace,{1,L,fusion_inner},"B,L,D","reshape");
        main_ops.add(prefix+".cross.text_to_visual.output.matmul",dt_mat,{1,L,d},"B,N,D","matmul","exhaustive");
        main_ops.add(prefix+".cross.text_to_visual.output.bias_add",dt,{1,L,d},"B,N,D","add","exhaustive");
        main_ops.add(prefix+".cross.text_to_visual.output",dt,{1,L,d},"B,L,D","linear","layer");
        ggml_tensor * vn_res=bf16_round(c,ln(c,visual,w.nvw,w.nvb,1e-5f,model_bf16),model_bf16);
        ggml_tensor * tn_res=bf16_round(c,ln(c,text,w.nlw,w.nlb,1e-5f,model_bf16),model_bf16);
        ggml_tensor * visual_scaled=bf16_round(c,ggml_mul(c,dv,w.gamma_v),model_bf16);
        ggml_tensor * text_scaled=bf16_round(c,ggml_mul(c,dt,w.gamma_l),model_bf16);
        visual=bf16_round(c,ggml_add(c,vn_res,visual_scaled),mt==GGML_TYPE_BF16);
        text=bf16_round(c,ggml_add(c,tn_res,text_scaled),mt==GGML_TYPE_BF16);
        main_ops.add(prefix+".visual.gamma",w.gamma_v,{d},"D","parameter");
        main_ops.add(prefix+".text.gamma",w.gamma_l,{d},"D","parameter");
        main_ops.add(prefix+".visual.delta",dv,{1,visual_n,d},"B,N,D","attention");
        main_ops.add(prefix+".text.delta",dt,{1,L,d},"B,N,D","attention");
        main_ops.add(prefix+".visual.scaled_delta",visual_scaled,{1,visual_n,d},"B,N,D","multiply","exhaustive");
        main_ops.add(prefix+".text.scaled_delta",text_scaled,{1,L,d},"B,N,D","multiply","exhaustive");
        main_ops.add(prefix+".visual.drop_path_output",visual_scaled,{1,visual_n,d},"B,N,D","drop_path","exhaustive");
        main_ops.add(prefix+".text.drop_path_output",text_scaled,{1,L,d},"B,N,D","drop_path","exhaustive");
        main_ops.add(prefix+".visual.residual_sum",visual,{1,visual_n,d},"B,V,D","add");
        main_ops.add(prefix+".text.residual_sum",text,{1,L,d},"B,L,D","add");
        main_ops.add(prefix+".text.after_fusion",text,{1,L,d},"B,L,D","identity","layer");
        const std::string enhancer=prefix+".text_enhancer";
        main_ops.add(enhancer+".input",text,{1,L,d},"B,N,D","layer_input","layer");
        ggml_tensor * da=packed_attention(c,text,text,w.inw,w.inb,w.iow,w.iob,d,4,model_bf16,self_mask,&main_ops,enhancer+".attn");
        ggml_tensor * text_residual_1=ggml_add(c,text,da);
        main_ops.add(enhancer+".residual_1.left",text,{1,L,d},"B,N,D","identity","exhaustive");
        main_ops.add(enhancer+".residual_1.right",da,{1,L,d},"B,N,D","identity","exhaustive");
        main_ops.add(enhancer+".residual_1.sum",text_residual_1,{1,L,d},"B,N,D","add");
        main_ops.add_layer_norm_diagnostics(
            enhancer+".norm_1",text_residual_1,{1,L,d},"B,N,D","B,N,1",1e-5f,model_bf16);
        text=bf16_round(c,ln(c,text_residual_1,w.t1w,w.t1b,1e-5f,model_bf16),model_bf16);
        main_ops.add(enhancer+".norm_1.output",text,{1,L,d},"B,N,D","layer_norm","exhaustive");
        ggml_tensor * f1_mat=diagnostic_matmul(c,w.f1w,text,fusion_exhaustive);
        ggml_tensor * f1=linear(c,w.f1w,w.f1b,text),*activation=ggml_relu(c,f1);
        ggml_tensor * f2_mat=diagnostic_matmul(c,w.f2w,activation,fusion_exhaustive);
        ggml_tensor * f=linear(c,w.f2w,w.f2b,activation);
        main_ops.add(enhancer+".ffn.linear_1.matmul",f1_mat,{1,L,fusion_inner},"B,N,D","matmul","exhaustive");
        main_ops.add(enhancer+".ffn.linear_1.bias_add",f1,{1,L,fusion_inner},"B,N,D","linear_projection","exhaustive");
        main_ops.add(enhancer+".ffn.activation.input",f1,{1,L,fusion_inner},"B,N,F","activation","exhaustive");
        main_ops.add(enhancer+".ffn.activation.output",activation,{1,L,fusion_inner},"B,N,F","relu");
        main_ops.add(enhancer+".ffn.linear_2.matmul",f2_mat,{1,L,d},"B,N,D","matmul","exhaustive");
        main_ops.add(enhancer+".ffn.linear_2.bias_add",f,{1,L,d},"B,N,D","linear_projection","exhaustive");
        ggml_tensor * text_residual_2=ggml_add(c,text,f);
        main_ops.add(enhancer+".residual_2.left",text,{1,L,d},"B,N,D","identity","exhaustive");
        main_ops.add(enhancer+".residual_2.right",f,{1,L,d},"B,N,D","identity","exhaustive");
        main_ops.add(enhancer+".residual_2.sum",text_residual_2,{1,L,d},"B,N,D","add");
        main_ops.add_layer_norm_diagnostics(
            enhancer+".norm_2",text_residual_2,{1,L,d},"B,N,D","B,N,1",1e-5f,model_bf16);
        text=bf16_round(c,ln(c,text_residual_2,w.t2w,w.t2b,1e-5f,model_bf16),model_bf16);
        main_ops.add(enhancer+".norm_2.output",text,{1,L,d},"B,N,D","layer_norm","exhaustive");
        if(trace.includes_layer()){fusion_visual_outputs.push_back(visual);fusion_text_outputs.push_back(text);}
        if(fusion_index++==0){first_fusion_visual_debug=visual;first_fusion_text_debug=text;}
    }
    ggml_tensor * condition=ggml_concat(c,visual,text,1);
    main_ops.add_host_f32("condition.concat_axis",{1.0f},{},"","constant");
    ggml_tensor * st=ggml_new_tensor_1d(c,GGML_TYPE_F32,state_d);ggml_set_input(st);
    ggml_tensor * state_norm=ln(c,st,state_nw,state_nb,1e-5f);
    main_ops.add_layer_norm_diagnostics(
        "state_projection.norm",st,{1,state_d},"B,D","B,1",1e-5f,true);
    ggml_tensor * state_linear_1_mat=diagnostic_matmul(c,state_w1,state_norm,trace.includes_exhaustive());
    ggml_tensor * state_linear_1=linear(c,state_w1,state_b1,state_norm);
    ggml_tensor * state_activation=ggml_gelu_erf(c,state_linear_1);
    ggml_tensor * state_linear_2_mat=diagnostic_matmul(c,state_w2,state_activation,trace.includes_exhaustive());
    ggml_tensor * state_linear_2=linear(c,state_w2,state_b2,state_activation);
    ggml_tensor * state_reshaped=ggml_reshape_2d(c,state_linear_2,d,2);
    ggml_tensor * state_reshaped_trace=trace.includes_op()?ggml_dup(c,state_reshaped):nullptr;
    ggml_tensor * state_positioned=ggml_add(c,state_reshaped,state_pos);
    ggml_tensor * stok=bf16_round(c,ln(c,state_positioned,state_ow,state_ob,1e-5f),mt==GGML_TYPE_BF16);
    main_ops.add_layer_norm_diagnostics(
        "state_projection.output_norm",state_positioned,{1,2,d},"B,S,D","B,S,1",1e-5f,true);
    main_ops.add("state_projection.norm.output",state_norm,{1,state_d},"B,D","layer_norm","exhaustive");
    main_ops.add("state_projection.linear_1.matmul",state_linear_1_mat,{1,d},"B,F","matmul","exhaustive");
    main_ops.add("state_projection.linear_1.bias_add",state_linear_1,{1,d},"B,F","linear_projection","exhaustive");
    main_ops.add("state_projection.gelu.input",state_linear_1,{1,d},"B,F","gelu","exhaustive");
    main_ops.add("state_projection.gelu.output",state_activation,{1,d},"B,F","gelu");
    main_ops.add("state_projection.linear_2.matmul",state_linear_2_mat,{1,2*d},"B,SxD","matmul","exhaustive");
    main_ops.add("state_projection.linear_2.bias_add",state_linear_2,{1,2*d},"B,SxD","linear_projection","exhaustive");
    main_ops.add("state_projection.before_reshape",state_linear_2,{1,2*d},"B,SxD","identity","exhaustive");
    main_ops.add("state_projection.after_reshape",state_reshaped_trace,{1,2,d},"B,S,D","reshape");
    main_ops.add("state_projection.position.after_add",state_positioned,{1,2,d},"B,S,D","add");
    main_ops.add("state_projection.position.before_add",state_reshaped_trace,{1,2,d},"B,S,D","identity","exhaustive");
    main_ops.add("state_projection.output_norm.output",stok,{1,2,d},"B,S,D","layer_norm","exhaustive");
    ggml_tensor * memory=ggml_concat(c,condition,stok,1),* act=action_queries;ggml_set_output(memory);
    main_ops.add_host_f32("action.memory.concat_axis",{1.0f},{},"","constant");
    main_ops.add("action.queries.weight",action_queries,{action_n,d},"T,D","parameter");
    main_ops.add("action.queries.expanded",action_queries,{1,action_n,d},"B,T,D","expand");
    std::vector<ggml_tensor*> decoder_layers_trace;
    for(size_t layer_index=0;layer_index<decoder.size();++layer_index){
        const auto & w=decoder[layer_index];
        const std::string prefix="action.decoder.layer_"+two_digit(layer_index);
        main_ops.add(prefix+".input",act,{1,action_n,d},"B,T,D","layer_input","layer");
        ggml_tensor * self_residual_left=act;
        main_ops.add(prefix+".self_attn.norm_input",self_residual_left,{1,action_n,d},"B,T,D","identity");
        ggml_tensor * z=ln(c,act,w.n1w,w.n1b,1e-5f);
        main_ops.add_layer_norm_diagnostics(
            prefix+".self_attn.norm",act,{1,action_n,d},"B,T,D","B,T,1",1e-5f,true);
        main_ops.add(prefix+".self_attn.norm.output",z,{1,action_n,d},"B,T,D","layer_norm","exhaustive");
        ggml_tensor * self_delta=packed_attention(c,z,z,w.siw,w.sib,w.sow,w.sob,d,8,model_bf16,nullptr,&main_ops,prefix+".self_attn");
        act=bf16_round(c,ggml_add(c,act,self_delta),mt==GGML_TYPE_BF16);ggml_set_output(act);
        main_ops.add(prefix+".self_attn.residual.left",self_residual_left,{1,action_n,d},"B,T,D","identity");
        main_ops.add(prefix+".self_attn.residual.right",self_delta,{1,action_n,d},"B,T,D","identity");
        main_ops.add(prefix+".self_attn.residual.sum",act,{1,action_n,d},"B,T,D","add");
        ggml_tensor * cross_residual_left=act;
        main_ops.add(prefix+".cross_attn.norm_input",cross_residual_left,{1,action_n,d},"B,T,D","identity");
        z=ln(c,act,w.n2w,w.n2b,1e-5f);
        main_ops.add_layer_norm_diagnostics(
            prefix+".cross_attn.norm",act,{1,action_n,d},"B,T,D","B,T,1",1e-5f,true);
        main_ops.add(prefix+".cross_attn.norm.output",z,{1,action_n,d},"B,T,D","layer_norm","exhaustive");
        ggml_tensor * cross_delta=packed_attention(c,z,memory,w.ciw,w.cib,w.cow,w.cob,d,8,model_bf16,nullptr,&main_ops,prefix+".cross_attn");
        act=bf16_round(c,ggml_add(c,act,cross_delta),mt==GGML_TYPE_BF16);ggml_set_output(act);
        main_ops.add(prefix+".cross_attn.residual.left",cross_residual_left,{1,action_n,d},"B,T,D","identity");
        main_ops.add(prefix+".cross_attn.residual.right",cross_delta,{1,action_n,d},"B,T,D","identity");
        main_ops.add(prefix+".cross_attn.residual.sum",act,{1,action_n,d},"B,T,D","add");
        ggml_tensor * ffn_residual_left=act;
        main_ops.add(prefix+".ffn.norm_input",ffn_residual_left,{1,action_n,d},"B,T,D","identity");
        z=ln(c,act,w.n3w,w.n3b,1e-5f);
        main_ops.add_layer_norm_diagnostics(
            prefix+".ffn.norm",act,{1,action_n,d},"B,T,D","B,T,1",1e-5f,true);
        main_ops.add(prefix+".ffn.norm.output",z,{1,action_n,d},"B,T,D","layer_norm","exhaustive");
        ggml_tensor * decoder_f1_mat=diagnostic_matmul(c,w.f1w,z,trace.includes_exhaustive());
        ggml_tensor * decoder_f1=linear(c,w.f1w,w.f1b,z);
        ggml_tensor * decoder_activation=ggml_relu(c,decoder_f1);
        ggml_tensor * decoder_f2_mat=diagnostic_matmul(c,w.f2w,decoder_activation,trace.includes_exhaustive());
        ggml_tensor * decoder_f2=linear(c,w.f2w,w.f2b,decoder_activation);
        main_ops.add(prefix+".ffn.linear_1.matmul",decoder_f1_mat,{1,action_n,2048},"B,T,F","matmul","exhaustive");
        main_ops.add(prefix+".ffn.linear_1.bias_add",decoder_f1,{1,action_n,2048},"B,T,F","linear_projection","exhaustive");
        main_ops.add(prefix+".ffn.activation.input",decoder_f1,{1,action_n,2048},"B,T,F","activation","exhaustive");
        main_ops.add(prefix+".ffn.activation.output",decoder_activation,{1,action_n,2048},"B,T,F","relu");
        main_ops.add(prefix+".ffn.linear_2.matmul",decoder_f2_mat,{1,action_n,d},"B,T,D","matmul","exhaustive");
        main_ops.add(prefix+".ffn.linear_2.bias_add",decoder_f2,{1,action_n,d},"B,T,D","linear_projection","exhaustive");
        act=bf16_round(c,ggml_add(c,act,decoder_f2),mt==GGML_TYPE_BF16);ggml_set_output(act);
        main_ops.add(prefix+".ffn.residual.left",ffn_residual_left,{1,action_n,d},"B,T,D","identity");
        main_ops.add(prefix+".ffn.residual.right",decoder_f2,{1,action_n,d},"B,T,D","identity");
        main_ops.add(prefix+".ffn.residual.sum",act,{1,action_n,d},"B,T,D","add");
        if(trace.includes_layer())decoder_layers_trace.push_back(act);
    }
    ggml_tensor * decoder_hidden=act;
    const bool dump=std::getenv("VLA_TURBOVLA_DUMP_DIR")!=nullptr;
    if(dump){ggml_set_output(visual_input_debug);ggml_set_output(bert_text_debug);ggml_set_output(first_fusion_visual_debug);ggml_set_output(first_fusion_text_debug);ggml_set_output(stok);ggml_set_output(condition);ggml_set_output(memory);ggml_set_output(decoder_hidden);}
    if(trace.active()){
        ggml_set_output(bert_unpadded_trace);ggml_set_output(bert_padded_trace);ggml_set_output(bert_text_debug);
        for(ggml_tensor * tensor:bert_layers_trace)ggml_set_output(tensor);
        for(ggml_tensor * tensor:fusion_visual_inputs)ggml_set_output(tensor);
        for(ggml_tensor * tensor:fusion_text_inputs)ggml_set_output(tensor);
        for(ggml_tensor * tensor:fusion_visual_outputs)ggml_set_output(tensor);
        for(ggml_tensor * tensor:fusion_text_outputs)ggml_set_output(tensor);
        for(ggml_tensor * tensor:decoder_layers_trace)ggml_set_output(tensor);
        ggml_set_output(visual);ggml_set_output(text);ggml_set_output(stok);
        ggml_set_output(condition);ggml_set_output(memory);
    }
    main_ops.add("action.mlp.input",act,{1,action_n,d},"B,T,D","input","layer");
    ggml_tensor * action_linear_0_mat=diagnostic_matmul(c,action_w1,act,trace.includes_exhaustive());
    ggml_tensor * action_linear_0=linear(c,action_w1,action_b1,act);
    main_ops.add("action.mlp.linear_0.matmul",action_linear_0_mat,{1,action_n,512},"B,T,D","matmul","exhaustive");
    main_ops.add("action.mlp.relu_0.input",action_linear_0,{1,action_n,512},"B,T,D","relu","exhaustive");
    act=ggml_relu(c,action_linear_0);
    main_ops.add("action.mlp.linear_0.bias_add",action_linear_0,{1,action_n,512},"B,T,D","linear_projection","exhaustive");
    main_ops.add("action.mlp.relu_0.output",act,{1,action_n,512},"B,T,D","relu");
    ggml_tensor * action_linear_1_mat=diagnostic_matmul(c,action_w2,act,trace.includes_exhaustive());
    ggml_tensor * action_linear_1=linear(c,action_w2,action_b2,act);
    main_ops.add("action.mlp.linear_1.matmul",action_linear_1_mat,{1,action_n,512},"B,T,D","matmul","exhaustive");
    main_ops.add("action.mlp.relu_1.input",action_linear_1,{1,action_n,512},"B,T,D","relu","exhaustive");
    act=ggml_relu(c,action_linear_1);
    main_ops.add("action.mlp.linear_1.bias_add",action_linear_1,{1,action_n,512},"B,T,D","linear_projection","exhaustive");
    main_ops.add("action.mlp.relu_1.output",act,{1,action_n,512},"B,T,D","relu");
    ggml_tensor * action_linear_2_mat=diagnostic_matmul(c,action_w3,act,trace.includes_exhaustive());
    ggml_tensor * before_tanh_trace=linear(c,action_w3,action_b3,act);
    main_ops.add("action.mlp.linear_2.matmul",action_linear_2_mat,{1,action_n,action_d},"B,T,D","matmul","exhaustive");
    main_ops.add("action.mlp.linear_2.bias_add",before_tanh_trace,{1,action_n,action_d},"B,T,D","linear_projection","exhaustive");
    ggml_tensor * output=ggml_tanh(c,before_tanh_trace);ggml_set_output(output);
    if(trace.active())ggml_set_output(before_tanh_trace);
    ggml_cgraph * graph=ggml_new_graph_custom(c,65536,false);ggml_build_forward_expand(graph,output);
    main_ops.expand_graph(graph);
    if(!main_alloc)
        main_alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_t alloc=main_alloc;
    if(!alloc||!ggml_gallocr_alloc_graph(alloc,graph)){std::fprintf(stderr,"vla(turbovla): main graph allocation failed\n");return{};}
    const auto ti_graph_ready=clock::now();
    ggml_backend_tensor_set(ids,in.lang_tokens,0,BL*sizeof(int32_t));int32_t z0=0;ggml_backend_tensor_set(zero,&z0,0,sizeof(z0));
    if(bert_pad){std::vector<float>zp((size_t)bd*(L-BL),0.0f);ggml_backend_tensor_set(bert_pad,zp.data(),0,zp.size()*sizeof(float));}
    std::vector<int32_t> positions(BL,0),attention(L,0);
    if(in.attention_mask&&in.attention_mask_n==BL)for(int i=0;i<BL;++i)attention[i]=in.attention_mask[i]?1:0;
    else for(int i=0;i<BL;++i) attention[i]=in.lang_tokens[i]!=0?1:0;
    std::vector<uint8_t> bert_allowed((size_t)BL*BL,0);for(int i=0;i<BL;++i)bert_allowed[i*BL+i]=1;int previous=0;
    for(int col=0;col<BL;++col){int id=in.lang_tokens[col];if(id==101||id==102||id==1012||id==1029){if(col==0||col==BL-1){positions[col]=0;}else{for(int q=previous+1;q<=col;++q){positions[q]=q-(previous+1);for(int k=previous+1;k<=col;++k)bert_allowed[q*BL+k]=1;}}previous=col;}}
    ggml_backend_tensor_set(pids,positions.data(),0,BL*sizeof(int32_t));const float ni=-3.3895313892515355e38f;
    const float neg_inf=-std::numeric_limits<float>::infinity();
    std::vector<float> bm((size_t)BL*BL),sm((size_t)L*L,neg_inf),cm((size_t)L*visual_n);
    for(int q=0;q<BL;++q)for(int k=0;k<BL;++k)bm[q*BL+k]=bert_allowed[q*BL+k]?0.0f:ni;
    for(int q=0;q<L;++q)sm[q*L+q]=0.0f;
    for(int q=0;q<BL;++q)for(int k=0;k<BL;++k)sm[q*L+k]=bert_allowed[q*BL+k]?0.0f:neg_inf;
    for(int q=0;q<visual_n;++q)for(int k=0;k<L;++k)cm[q*L+k]=attention[k]?0.0f:neg_inf;
    ggml_backend_tensor_set(bmask,bm.data(),0,bm.size()*sizeof(float));ggml_backend_tensor_set(self_mask,sm.data(),0,sm.size()*sizeof(float));ggml_backend_tensor_set(cross_mask,cm.data(),0,cm.size()*sizeof(float));
    if(text_zero_fill_mask){
        std::vector<float> zero_fill((size_t)L,1.0f);
        for(int i=0;i<BL;++i)zero_fill[(size_t)i]=attention[(size_t)i]?0.0f:1.0f;
        ggml_backend_tensor_set(text_zero_fill_mask,zero_fill.data(),0,zero_fill.size()*sizeof(float));
    }
    ggml_backend_tensor_copy(visual_device_output,visual_input_debug);
    ggml_backend_tensor_set(st,in.state,0,state_d*sizeof(float));
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS){std::fprintf(stderr,"vla(turbovla): main compute failed\n");return{};}
    std::vector<float> out((size_t)action_n*action_d);ggml_backend_tensor_get(output,out.data(),0,out.size()*sizeof(float));
    const auto ti_compute_ready=clock::now();
    if(trace.active()){
        trace.i32("text.bert.embeddings.input_ids",in.lang_tokens,BL,{1,BL},"B,N","input","op");
        trace.i32("text.layout.input_ids_padded",in.lang_tokens,BL,{1,BL},"B,N","padding");
        std::vector<uint8_t> attention_u8(BL),key_padding_u8(BL),self_u8((size_t)BL*BL);
        for(int i=0;i<BL;++i){attention_u8[i]=(uint8_t)attention[i];key_padding_u8[i]=(uint8_t)!attention[i];}
        std::copy(bert_allowed.begin(),bert_allowed.end(),self_u8.begin());
        trace.u8("text.layout.attention_mask_padded",attention_u8.data(),attention_u8.size(),{1,BL},"B,N","padding");
        trace.u8("text.layout.key_padding_mask_padded",key_padding_u8.data(),key_padding_u8.size(),{1,BL},"B,N","invert","op");
        trace.u8("text.layout.self_attention_mask_padded",self_u8.data(),self_u8.size(),{1,BL,BL},"B,N,N","padding");
        trace.i32("text.layout.position_ids_padded",positions.data(),positions.size(),{1,BL},"B,N","padding");
        std::vector<uint8_t> bert_mask_heads((size_t)bh*BL*BL);
        for(int head=0;head<bh;++head)
            for(int q=0;q<BL;++q)for(int k=0;k<BL;++k)
                bert_mask_heads[((size_t)head*BL+q)*BL+k]=(uint8_t)!bert_allowed[(size_t)q*BL+k];
        for(size_t layer=0;layer<bert.size();++layer)
            trace.u8("text.bert.layer_"+two_digit(layer)+".attn.mask",bert_mask_heads.data(),bert_mask_heads.size(),{1,bh,BL,BL},"B,H,N,N","mask","op");
        for(size_t layer=0;layer<bert_layers_trace.size();++layer)
            trace.tensor_f32("text.bert.layer_"+two_digit(layer)+".output",backend,bert_layers_trace[layer],{1,BL,bd},"B,N,D","layer_output","layer");
        trace.tensor_f32("text.bert.last_hidden_state_unpadded",backend,bert_unpadded_trace,{1,BL,bd},"B,N,D","bert_output");
        trace.tensor_f32("text.bert.hidden_padded",backend,bert_padded_trace,{1,L,bd},"B,N,D","padding");
        trace.tensor_f32("text.projection.output",backend,bert_text_debug,{1,L,d},"B,N,D","linear_projection");
        std::vector<uint8_t> enhancer_mask((size_t)L*L),enhancer_mask_repeated((size_t)4*L*L);
        for(int q=0;q<L;++q)for(int k=0;k<L;++k){
            const uint8_t masked=(uint8_t)(sm[(size_t)q*L+k] < -1.0e30f);
            enhancer_mask[(size_t)q*L+k]=masked;
            for(int head=0;head<4;++head)
                enhancer_mask_repeated[((size_t)head*L+q)*L+k]=masked;
        }
        for(size_t layer=0;layer<fusion_visual_inputs.size();++layer){
            const std::string prefix="interaction.layer_"+two_digit(layer);
            trace.u8(prefix+".text_enhancer.mask.original",enhancer_mask.data(),enhancer_mask.size(),{1,L,L},"B,N,N","mask","op");
            trace.u8(prefix+".text_enhancer.mask.repeated",enhancer_mask_repeated.data(),enhancer_mask_repeated.size(),{4,L,L},"BH,N,N","repeat","op");
            trace.tensor_f32(prefix+".visual.input",backend,fusion_visual_inputs[layer],{1,visual_n,d},"B,VxN,D","layer_input");
            trace.tensor_f32(prefix+".text.input",backend,fusion_text_inputs[layer],{1,L,d},"B,N,D","layer_input");
            if(layer<fusion_visual_outputs.size()){
                trace.tensor_f32(prefix+".visual.after_fusion",backend,fusion_visual_outputs[layer],{1,visual_n,d},"B,V,D","identity","layer");
                trace.tensor_f32(prefix+".text_enhancer.output",backend,fusion_text_outputs[layer],{1,L,d},"B,N,D","layer_output","layer");
            }
        }
        trace.tensor_f32("condition.visual_tokens",backend,visual,{1,visual_n,d},"B,VxN,D","split");
        trace.tensor_f32("condition.text_tokens",backend,text,{1,L,d},"B,N,D","split");
        trace.tensor_f32("condition.concatenated",backend,condition,{1,visual_n+L,d},"B,VxN+N,D","concat");
        std::vector<float> state_model(in.state,in.state+state_d);
        if(mt==GGML_TYPE_BF16)for(float & value:state_model)value=round_bf16(value);
        trace.f32("state.normalized_f32",in.state,state_d,{1,state_d},"B,D","divide");
        trace.f32("state.normalized_model_dtype",state_model.data(),state_model.size(),{1,state_d},"B,D","cast");
        trace.f32("state_projection.input_raw",in.state,state_d,{1,state_d},"B,D","input");
        trace.f32("state_projection.input_normalized",in.state,state_d,{1,state_d},"B,D","input");
        trace.tensor_f32("state_projection.position_embedding",backend,state_pos,{1,2,d},"1,S,D","parameter");
        trace.tensor_f32("state_projection.output",backend,stok,{1,2,d},"B,S,D","layer_norm");
        trace.tensor_f32("action.memory.condition",backend,condition,{1,visual_n+L,d},"B,VxN+N,D","split");
        trace.tensor_f32("action.memory.state_tokens",backend,stok,{1,2,d},"B,S,D","state_projection");
        trace.tensor_f32("action.memory.concatenated",backend,memory,{1,visual_n+L+2,d},"B,VxN+N+S,D","concat");
        for(size_t layer=0;layer<decoder_layers_trace.size();++layer)
            trace.tensor_f32("action.decoder.layer_"+two_digit(layer)+".output",backend,decoder_layers_trace[layer],{1,action_n,d},"B,T,D","layer_output","layer");
        trace.tensor_f32("action.before_tanh",backend,before_tanh_trace,{1,action_n,action_d},"B,T,A","mlp");
        trace.f32("action.normalized",out.data(),out.size(),{1,action_n,action_d},"B,T,A","tanh");
        main_ops.dump(backend);
    }
    if(dump){const char*ddir=std::getenv("VLA_TURBOVLA_DUMP_DIR");auto dt=[&](const char*n,ggml_tensor*t){std::vector<float>x((size_t)ggml_nelements(t));ggml_backend_tensor_get(t,x.data(),0,x.size()*sizeof(float));dump_f32(ddir,n,x.data(),x.size());};dt("vision.main_input.f32",visual_input_debug);dt("text.projected.f32",bert_text_debug);dt("interaction.layer0.visual.f32",first_fusion_visual_debug);dt("interaction.layer0.text.f32",first_fusion_text_debug);dt("state.tokens.f32",stok);dt("condition.f32",condition);dt("memory.f32",memory);dt("action.decoder_hidden.f32",decoder_hidden);dump_f32(ddir,"action.output.f32",out.data(),out.size());}
    stats.ms_inference=std::chrono::duration<float,std::milli>(clock::now()-ti).count();stats.ms_total=std::chrono::duration<float,std::milli>(clock::now()-start).count();
    if(std::getenv("VLA_TURBOVLA_PROFILE"))std::fprintf(stderr,
        "vla(turbovla): main profile graph_build_alloc=%.3f graph_compute_readback=%.3f cleanup=%.3f ms\n",
        std::chrono::duration<float,std::milli>(ti_graph_ready-ti).count(),
        std::chrono::duration<float,std::milli>(ti_compute_ready-ti_graph_ready).count(),
        std::chrono::duration<float,std::milli>(clock::now()-ti_compute_ready).count());
    return out;
}

} // namespace vla
