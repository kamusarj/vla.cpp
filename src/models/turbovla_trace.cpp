// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "models/turbovla_trace.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

namespace vla {
namespace {

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string safe_name(std::string value) {
    for (char & c : value) {
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!safe) c = '_';
    }
    return value;
}

std::string shape_json(const std::vector<int64_t> & shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::string stage_dir(const std::string & name) {
    const std::string stage = name.substr(0, name.find('.'));
    if (stage == "input") return "00_input";
    if (stage == "text") return "10_text_encoder";
    if (stage == "vision") return "20_vision_encoder";
    if (stage == "vision_projection") return "30_vision_projection";
    if (stage == "interaction") return "40_interaction";
    if (stage == "condition") return "45_condition";
    if (stage == "state" || stage == "state_projection") return "50_state_projection";
    if (stage == "action") return "60_action_decoder";
    return stage;
}

int trace_level(const char * value) {
    const std::string level = value ? value : "boundary";
    if (level == "boundary") return 0;
    if (level == "layer") return 1;
    if (level == "op") return 2;
    if (level == "exhaustive") return 3;
    std::fprintf(stderr,
                 "vla(turbovla): unsupported C++ trace level '%s'; using exhaustive\n",
                 level.c_str());
    return 3;
}

size_t shape_numel(const std::vector<int64_t> & shape) {
    size_t result = 1;
    for (const int64_t dim : shape) result *= static_cast<size_t>(dim);
    return result;
}

float round_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

TurboVlaTrace::TurboVlaTrace(const char * root, uint64_t forward_index) {
    if (!root || !*root) return;
    level_ = trace_level(std::getenv("VLA_TURBOVLA_TRACE_LEVEL"));

    std::ostringstream dirname;
    dirname << "forward_" << std::setw(6) << std::setfill('0') << forward_index
            << "_rank_0";
    trace_dir_ = std::filesystem::path(root) / dirname.str();
    if (std::filesystem::exists(trace_dir_)) {
        std::fprintf(stderr,
                     "vla(turbovla): trace directory already exists: %s\n",
                     trace_dir_.c_str());
        return;
    }

    std::filesystem::create_directories(trace_dir_ / "tensors");
    manifest_.open(trace_dir_ / "manifest.jsonl", std::ios::out);
    summary_.open(trace_dir_ / "summary.csv", std::ios::out);
    if (!manifest_ || !summary_) {
        std::fprintf(stderr, "vla(turbovla): failed to open trace at %s\n",
                     trace_dir_.c_str());
        return;
    }
    summary_ << "trace_id,semantic_name,call_index,required_level,shape,layout,"
                "storage_dtype,numel,min,max,mean,std,abs_max,l2_norm,nan_count,"
                "inf_count,file\n";
    active_ = true;
    static const char * levels[]={"boundary","boundary+layer","op","exhaustive"};
    std::fprintf(stderr, "vla(turbovla): tracing %s tensors to %s\n",
                 levels[level_], trace_dir_.c_str());
}

TurboVlaTrace::~TurboVlaTrace() {
    finish();
}

bool TurboVlaTrace::accepts(const char * required_level) const {
    if (!active_) return false;
    const std::string level=required_level?required_level:"boundary";
    const int required=level=="boundary"?0:level=="layer"?1:level=="op"?2:3;
    return level_>=required;
}

bool TurboVlaTrace::includes(const char * required_level) const {
    return accepts(required_level);
}

void TurboVlaTrace::f32(const std::string & name, const float * data, size_t count,
                        std::initializer_list<int64_t> shape, const char * layout,
                        const char * operation, const char * required_level) {
    write(name, data, count, std::vector<int64_t>(shape), layout, operation,
          required_level, "torch.float32", "float32", "f32le");
}

void TurboVlaTrace::f32(const std::string & name, const float * data, size_t count,
                        const std::vector<int64_t> & shape, const char * layout,
                        const char * operation, const char * required_level) {
    write(name,data,count,shape,layout,operation,required_level,
          "torch.float32","float32","f32le");
}

void TurboVlaTrace::i32(const std::string & name, const int32_t * data, size_t count,
                        std::initializer_list<int64_t> shape, const char * layout,
                        const char * operation, const char * required_level) {
    write(name, data, count, std::vector<int64_t>(shape), layout, operation,
          required_level, "torch.int32", "int32", "i32");
}

void TurboVlaTrace::u8(const std::string & name, const uint8_t * data, size_t count,
                       std::initializer_list<int64_t> shape, const char * layout,
                       const char * operation, const char * required_level) {
    write(name, data, count, std::vector<int64_t>(shape), layout, operation,
          required_level, "torch.uint8", "uint8", "u8");
}

void TurboVlaTrace::tensor_f32(const std::string & name, ggml_backend_t,
                               ggml_tensor * tensor,
                               std::initializer_list<int64_t> shape,
                               const char * layout, const char * operation,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor) return;
    const size_t count=static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> data(count);
    if(tensor->type==GGML_TYPE_BF16){
        std::vector<uint16_t> native(count);
        ggml_backend_tensor_get(tensor,native.data(),0,count*sizeof(uint16_t));
        for(size_t i=0;i<count;++i){
            uint32_t bits=static_cast<uint32_t>(native[i])<<16;
            std::memcpy(&data[i],&bits,sizeof(bits));
        }
    }else{
        ggml_backend_tensor_get(tensor,data.data(),0,count*sizeof(float));
    }
    write(name,data.data(),data.size(),std::vector<int64_t>(shape),layout,
          operation,required_level,
          tensor->type==GGML_TYPE_BF16?"torch.bfloat16":"torch.float32",
          "float32","f32le");
}

void TurboVlaTrace::tensor_f32(const std::string & name, ggml_backend_t,
                               ggml_tensor * tensor,
                               const std::vector<int64_t> & shape,
                               const char * layout, const char * operation,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor) return;
    const size_t count=static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> data(count);
    if(tensor->type==GGML_TYPE_BF16){
        std::vector<uint16_t> native(count);
        ggml_backend_tensor_get(tensor,native.data(),0,count*sizeof(uint16_t));
        for(size_t i=0;i<count;++i){
            uint32_t bits=static_cast<uint32_t>(native[i])<<16;
            std::memcpy(&data[i],&bits,sizeof(bits));
        }
    }else{
        ggml_backend_tensor_get(tensor,data.data(),0,count*sizeof(float));
    }
    write(name,data.data(),data.size(),shape,layout,operation,required_level,
          tensor->type==GGML_TYPE_BF16?"torch.bfloat16":"torch.float32",
          "float32","f32le");
}

void TurboVlaTrace::tensor_numeric_f32(
                               const std::string & name, ggml_backend_t,
                               ggml_tensor * tensor,
                               const std::vector<int64_t> & shape,
                               const char * layout, const char * operation,
                               bool round_to_bf16,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor) return;
    const size_t count = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> data(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor,data.data(),0,count*sizeof(float));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<uint16_t> native(count);
        ggml_backend_tensor_get(tensor,native.data(),0,count*sizeof(uint16_t));
        for (size_t index=0;index<count;++index) {
            uint32_t bits=static_cast<uint32_t>(native[index])<<16;
            std::memcpy(&data[index],&bits,sizeof(bits));
        }
    } else {
        return;
    }
    if (round_to_bf16) for (float & value:data)value=round_bf16(value);
    write(name,data.data(),data.size(),shape,layout,operation,required_level,
          round_to_bf16||tensor->type==GGML_TYPE_BF16?"torch.bfloat16":"torch.float32",
          "float32","f32le");
}

void TurboVlaTrace::tensor_f32_subtract_global_max(
                               const std::string & name, ggml_backend_t,
                               ggml_tensor * tensor,
                               ggml_tensor * max_tensor,
                               const std::vector<int64_t> & shape,
                               const char * layout, const char * operation,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor) return;
    std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    if (!data.empty()) {
        if (!max_tensor) max_tensor = tensor;
        std::vector<float> max_data(static_cast<size_t>(ggml_nelements(max_tensor)));
        ggml_backend_tensor_get(max_tensor, max_data.data(), 0,
                                max_data.size() * sizeof(float));
        const float global_max = *std::max_element(max_data.begin(), max_data.end());
        for (float & value : data) value = round_bf16(value - global_max);
    }
    write(name,data.data(),data.size(),shape,layout,operation,required_level,
          "torch.float32","float32","f32le");
}

void TurboVlaTrace::tensor_f32_subtract_row_max(
                               const std::string & name, ggml_backend_t,
                               ggml_tensor * tensor,
                               const std::vector<int64_t> & shape,
                               const char * layout, const char * operation,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor || shape.empty()) return;
    std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    const size_t row_size = static_cast<size_t>(shape.back());
    if (row_size > 0) {
        for (size_t offset = 0; offset < data.size(); offset += row_size) {
            const auto begin = data.begin() + static_cast<std::ptrdiff_t>(offset);
            const auto end = begin + static_cast<std::ptrdiff_t>(row_size);
            const float row_max = *std::max_element(begin, end);
            for (auto it = begin; it != end; ++it) {
                *it = round_bf16(*it - row_max);
            }
        }
    }
    write(name,data.data(),data.size(),shape,layout,operation,required_level,
          "torch.float32","float32","f32le");
}

void TurboVlaTrace::tensor_f32_layer_norm_diagnostics(
                               const std::string & prefix, ggml_backend_t,
                               ggml_tensor * tensor,
                               const std::vector<int64_t> & shape,
                               const char * layout, const char * stat_layout,
                               float eps, bool bf16,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor || shape.empty()) return;
    std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    const size_t row_size = static_cast<size_t>(shape.back());
    if (row_size == 0 || data.size() % row_size != 0) return;
    const size_t rows = data.size() / row_size;
    std::vector<float> mean(rows), variance(rows), inv_std(rows);
    std::vector<float> normalized(data.size());
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        float total = 0.0f;
        for (size_t col = 0; col < row_size; ++col) total += data[offset + col];
        mean[row] = total / static_cast<float>(row_size);
        if (bf16) mean[row] = round_bf16(mean[row]);

        float square_total = 0.0f;
        for (size_t col = 0; col < row_size; ++col) {
            float centered = data[offset + col] - mean[row];
            if (bf16) centered = round_bf16(centered);
            float square = centered * centered;
            if (bf16) square = round_bf16(square);
            square_total += square;
        }
        variance[row] = square_total / static_cast<float>(row_size);
        if (bf16) variance[row] = round_bf16(variance[row]);
        float variance_eps = variance[row] + eps;
        if (bf16) variance_eps = round_bf16(variance_eps);
        inv_std[row] = 1.0f / std::sqrt(variance_eps);
        if (bf16) inv_std[row] = round_bf16(inv_std[row]);
        for (size_t col = 0; col < row_size; ++col) {
            float centered = data[offset + col] - mean[row];
            if (bf16) centered = round_bf16(centered);
            float value = centered * inv_std[row];
            normalized[offset + col] = bf16 ? round_bf16(value) : value;
        }
    }
    std::vector<int64_t> stat_shape = shape;
    stat_shape.back() = 1;
    const char * source_dtype = bf16 ? "torch.bfloat16" : "torch.float32";
    write(prefix + ".mean", mean.data(), mean.size(), stat_shape, stat_layout,
          "mean", required_level, source_dtype, "float32", "f32le");
    write(prefix + ".variance", variance.data(), variance.size(), stat_shape,
          stat_layout, "variance", required_level, source_dtype, "float32", "f32le");
    write(prefix + ".inv_std", inv_std.data(), inv_std.size(), stat_shape,
          stat_layout, "rsqrt", required_level, source_dtype, "float32", "f32le");
    write(prefix + ".normalized", normalized.data(), normalized.size(), shape,
          layout, "multiply", required_level, source_dtype, "float32", "f32le");
}

void TurboVlaTrace::tensor_f32_softmax_diagnostics(
                               const std::string & prefix, ggml_backend_t,
                               ggml_tensor * tensor,
                               const std::vector<int64_t> & shape,
                               const char * layout, bool bf16,
                               const char * required_level) {
    if (!accepts(required_level) || !tensor || shape.empty()) return;
    std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    const size_t row_size = static_cast<size_t>(shape.back());
    if (row_size == 0 || data.size() % row_size != 0) return;
    const size_t rows = data.size() / row_size;
    std::vector<float> max_value(rows), shifted(data.size()), exponent(data.size()), total(rows);
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        float maximum = data[offset];
        for (size_t col = 1; col < row_size; ++col) maximum = std::max(maximum, data[offset + col]);
        max_value[row] = maximum;
        float sum = 0.0f;
        for (size_t col = 0; col < row_size; ++col) {
            float value = data[offset + col] - maximum;
            shifted[offset + col] = bf16 ? round_bf16(value) : value;
            exponent[offset + col] = std::exp(shifted[offset + col]);
            sum += exponent[offset + col];
        }
        total[row] = sum;
    }
    std::vector<int64_t> stat_shape = shape;
    stat_shape.back() = 1;
    const char * source_dtype = bf16 ? "torch.bfloat16" : "torch.float32";
    write(prefix + ".max", max_value.data(), max_value.size(), stat_shape, layout,
          "max", required_level, source_dtype, "float32", "f32le");
    write(prefix + ".shifted", shifted.data(), shifted.size(), shape, layout,
          "subtract", required_level, source_dtype, "float32", "f32le");
    write(prefix + ".exp", exponent.data(), exponent.size(), shape, layout,
          "exp", required_level, "torch.float32", "float32", "f32le");
    write(prefix + ".sum", total.data(), total.size(), stat_shape, layout,
          "sum", required_level, "torch.float32", "float32", "f32le");
}

void TurboVlaTrace::tensor_f32_fusion_diagnostics(
                               const std::string & prefix, ggml_backend_t,
                               ggml_tensor * raw_logits, ggml_tensor * text_mask,
                               int64_t heads, int64_t visual_tokens,
                               int64_t text_tokens, bool bf16,
                               const char * required_level) {
    if (!accepts(required_level) || !raw_logits || heads <= 0 ||
        visual_tokens <= 0 || text_tokens <= 0) return;
    const size_t h = static_cast<size_t>(heads);
    const size_t v = static_cast<size_t>(visual_tokens);
    const size_t l = static_cast<size_t>(text_tokens);
    std::vector<float> raw(h*v*l);
    ggml_backend_tensor_get(raw_logits, raw.data(), 0, raw.size()*sizeof(float));
    std::vector<float> additive_mask(v*l, 0.0f);
    if (text_mask) {
        ggml_backend_tensor_get(text_mask, additive_mask.data(), 0,
                                additive_mask.size()*sizeof(float));
    }
    const auto bf16_value = [bf16](float value) {
        return bf16 ? round_bf16(value) : value;
    };
    const float global_max = *std::max_element(raw.begin(), raw.end());
    std::vector<float> visual_shifted(raw.size()), visual_masked(raw.size());
    std::vector<float> visual_mask(h*l);
    for (size_t head = 0; head < h; ++head) {
        for (size_t row = 0; row < v; ++row) {
            for (size_t col = 0; col < l; ++col) {
                const size_t index = (head*v + row)*l + col;
                float shifted = bf16_value(raw[index] - global_max);
                shifted = std::clamp(shifted, -50000.0f, 50000.0f);
                visual_shifted[index] = shifted;
                const float mask_value = additive_mask[row*l + col];
                const float masked = std::isinf(mask_value) && mask_value < 0.0f ? 1.0f : 0.0f;
                visual_mask[head*l + col] = masked;
                visual_masked[index] = masked != 0.0f
                    ? -std::numeric_limits<float>::infinity()
                    : bf16_value(shifted + mask_value);
            }
        }
    }

    std::vector<float> transposed(h*l*v), row_max(h*l), text_shifted(h*l*v);
    std::vector<float> text_mask_values(h*l*v, 0.0f);
    for (size_t head = 0; head < h; ++head) {
        for (size_t text = 0; text < l; ++text) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t visual = 0; visual < v; ++visual) {
                const size_t source = (head*v + visual)*l + text;
                const size_t target = (head*l + text)*v + visual;
                transposed[target] = visual_shifted[source];
                maximum = std::max(maximum, transposed[target]);
            }
            row_max[head*l + text] = maximum;
            for (size_t visual = 0; visual < v; ++visual) {
                const size_t target = (head*l + text)*v + visual;
                text_shifted[target] = std::clamp(
                    bf16_value(transposed[target] - maximum),
                    -50000.0f, 50000.0f);
            }
        }
    }

    const char * bf16_source = bf16 ? "torch.bfloat16" : "torch.float32";
    write(prefix + ".cross.visual_to_text.logits_raw", raw.data(), raw.size(),
          {heads,visual_tokens,text_tokens}, "BH,V,L", "identity",
          required_level, bf16_source, "float32", "f32le");
    write(prefix + ".cross.visual_to_text.global_max", &global_max, 1, {}, "",
          "max", required_level, bf16_source, "float32", "f32le");
    write(prefix + ".cross.visual_to_text.logits_global_shifted",
          visual_shifted.data(), visual_shifted.size(),
          {heads,visual_tokens,text_tokens}, "BH,V,L", "subtract",
          required_level, bf16_source, "float32", "f32le");
    write(prefix + ".cross.visual_to_text.logits_clamped",
          visual_shifted.data(), visual_shifted.size(),
          {heads,visual_tokens,text_tokens}, "BH,V,L", "clamp",
          required_level, bf16_source, "float32", "f32le");
    write(prefix + ".cross.text_to_visual.row_max", row_max.data(), row_max.size(),
          {heads,text_tokens,1}, "BH,L,1", "max", required_level,
          bf16_source, "float32", "f32le");
    write(prefix + ".cross.text_to_visual.logits_shifted", text_shifted.data(),
          text_shifted.size(), {heads,text_tokens,visual_tokens}, "BH,L,V",
          "subtract", required_level, bf16_source, "float32", "f32le");
    write(prefix + ".cross.text_to_visual.logits_clamped", text_shifted.data(),
          text_shifted.size(), {heads,text_tokens,visual_tokens}, "BH,L,V",
          "clamp", required_level, bf16_source, "float32", "f32le");
    write(prefix + ".cross.text_to_visual.mask", text_mask_values.data(),
          text_mask_values.size(), {heads,text_tokens,visual_tokens}, "BH,L,V",
          "mask", required_level, "torch.bool", "float32", "f32le");
    write(prefix + ".cross.visual_to_text.mask", visual_mask.data(),
          visual_mask.size(), {heads,1,text_tokens}, "BH,V,L",
          "mask", required_level, "torch.bool", "float32", "f32le");

    const auto emit_softmax = [&](const std::string & direction,
                                  const std::vector<float> & logits,
                                  size_t rows, size_t cols,
                                  const std::vector<int64_t> & full_shape,
                                  const char * layout) {
        std::vector<float> maxima(rows), shifted(logits.size()), exponent(logits.size()), totals(rows);
        for (size_t row = 0; row < rows; ++row) {
            const size_t offset = row*cols;
            float maximum = logits[offset];
            for (size_t col = 1; col < cols; ++col) maximum = std::max(maximum, logits[offset+col]);
            maxima[row] = maximum;
            float sum = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                shifted[offset+col] = bf16_value(logits[offset+col] - maximum);
                exponent[offset+col] = std::exp(shifted[offset+col]);
                sum += exponent[offset+col];
            }
            totals[row] = sum;
        }
        std::vector<int64_t> stat_shape = full_shape;
        stat_shape.back() = 1;
        const std::string base = prefix + ".cross." + direction + ".softmax";
        write(base+".max",maxima.data(),maxima.size(),stat_shape,layout,"max",
              required_level,bf16_source,"float32","f32le");
        write(base+".shifted",shifted.data(),shifted.size(),full_shape,layout,"subtract",
              required_level,bf16_source,"float32","f32le");
        write(base+".exp",exponent.data(),exponent.size(),full_shape,layout,"exp",
              required_level,"torch.float32","float32","f32le");
        write(base+".sum",totals.data(),totals.size(),stat_shape,layout,"sum",
              required_level,"torch.float32","float32","f32le");
    };
    emit_softmax("visual_to_text",visual_masked,h*v,l,
                 {heads,visual_tokens,text_tokens},"BH,V,L");
    emit_softmax("text_to_visual",text_shifted,h*l,v,
                 {heads,text_tokens,visual_tokens},"BH,L,V");
}

template <typename T>
void TurboVlaTrace::write(const std::string & name, const T * data, size_t count,
                          const std::vector<int64_t> & shape, const char * layout,
                          const char * operation, const char * required_level,
                          const char * source_dtype, const char * storage_dtype,
                          const char * suffix) {
    if (!accepts(required_level)) return;
    if (shape_numel(shape) != count) {
        std::fprintf(stderr,
                     "vla(turbovla): trace shape mismatch for %s (%zu != %zu)\n",
                     name.c_str(), shape_numel(shape), count);
        active_ = false;
        return;
    }

    const std::filesystem::path subdir =
        std::filesystem::path("tensors") / stage_dir(name);
    std::filesystem::create_directories(trace_dir_ / subdir);
    std::ostringstream base;
    base << std::setw(6) << std::setfill('0') << trace_id_ << "__"
         << safe_name(name) << "__call_00." << suffix << ".bin";
    const std::filesystem::path relative = subdir / base.str();
    std::ofstream raw(trace_dir_ / relative, std::ios::binary);
    raw.write(reinterpret_cast<const char *>(data),
              static_cast<std::streamsize>(count * sizeof(T)));
    raw.close();

    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double sum = 0.0, sum_sq = 0.0, abs_max = 0.0, l1 = 0.0;
    uint64_t zero_count = 0, nan_count = 0, inf_count = 0, finite_count = 0;
    for (size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(data[i]);
        if (std::isnan(value)) { ++nan_count; continue; }
        if (std::isinf(value)) { ++inf_count; continue; }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += value;
        sum_sq += value * value;
        abs_max = std::max(abs_max, std::abs(value));
        l1 += std::abs(value);
        zero_count += value == 0.0;
        ++finite_count;
    }
    if (!finite_count) min_value = max_value = 0.0;
    const double mean = finite_count ? sum / finite_count : 0.0;
    const double variance = finite_count
        ? std::max(0.0, sum_sq / finite_count - mean * mean) : 0.0;
    const double stddev = std::sqrt(variance);
    const double l2 = std::sqrt(sum_sq);
    const uint64_t timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    const std::string shape_text = shape_json(shape);
    const std::string stage = name.substr(0, name.find('.'));
    manifest_ << std::setprecision(17)
              << "{\"trace_id\":" << trace_id_
              << ",\"semantic_name\":\"" << json_escape(name)
              << "\",\"call_index\":0,\"module_path\":null,\"operation\":\""
              << json_escape(operation ? operation : "unknown")
              << "\",\"io\":\"intermediate\",\"stage\":\"" << json_escape(stage)
              << "\",\"required_level\":\"" << json_escape(required_level)
              << "\",\"shape\":" << shape_text
              << ",\"layout\":\"" << json_escape(layout ? layout : "")
              << "\",\"strides\":[],\"source_dtype\":\"" << source_dtype
              << "\",\"storage_dtype\":\"" << storage_dtype
              << "\",\"native_storage_dtype\":\"" << storage_dtype
              << "\",\"endianness\":\"little\",\"contiguous\":true,\"numel\":" << count
              << ",\"file\":\"" << relative.generic_string()
              << "\",\"native_file\":null,\"min\":" << min_value
              << ",\"max\":" << max_value << ",\"mean\":" << mean
              << ",\"std\":" << stddev << ",\"abs_max\":" << abs_max
              << ",\"l1_norm\":" << l1 << ",\"l2_norm\":" << l2
              << ",\"zero_count\":" << zero_count
              << ",\"zero_ratio\":" << (count ? static_cast<double>(zero_count) / count : 0.0)
              << ",\"nan_count\":" << nan_count << ",\"inf_count\":" << inf_count
              << ",\"first_values\":[],\"last_values\":[],\"sha256_f32\":null,"
                 "\"sha256_native\":\"\",\"timestamp_ns\":" << timestamp
              << ",\"raw_bf16_file\":null,\"pt_file\":null}\n";
    manifest_.flush();

    summary_ << trace_id_ << ",\"" << name << "\",0," << required_level
             << ",\"" << shape_text << "\",\"" << (layout ? layout : "")
             << "\"," << storage_dtype << ',' << count << ',' << min_value << ','
             << max_value << ',' << mean << ',' << stddev << ',' << abs_max << ','
             << l2 << ',' << nan_count << ',' << inf_count << ",\""
             << relative.generic_string() << "\"\n";
    summary_.flush();

    std::ostringstream tree;
    tree << std::setw(6) << std::setfill('0') << trace_id_ << ' ' << name
         << " call=0 " << shape_text;
    tree_.push_back(tree.str());
    ++trace_id_;
}

void TurboVlaTrace::finish() {
    if (!manifest_.is_open() && !summary_.is_open()) return;
    manifest_.close();
    summary_.close();
    if (active_) {
        std::ofstream tree(trace_dir_ / "trace_tree.txt");
        for (const std::string & line : tree_) tree << line << '\n';
        std::ofstream environment(trace_dir_ / "environment.json");
        environment << "{\n  \"producer\": \"vla.cpp\",\n"
                       "  \"architecture\": \"turbovla\",\n"
                       "  \"storage\": \"little-endian raw tensors\"\n}\n";
        std::fprintf(stderr, "vla(turbovla): wrote %llu trace tensors to %s\n",
                     static_cast<unsigned long long>(trace_id_), trace_dir_.c_str());
    }
}

template void TurboVlaTrace::write<float>(
    const std::string &, const float *, size_t, const std::vector<int64_t> &,
    const char *, const char *, const char *, const char *, const char *, const char *);
template void TurboVlaTrace::write<int32_t>(
    const std::string &, const int32_t *, size_t, const std::vector<int64_t> &,
    const char *, const char *, const char *, const char *, const char *, const char *);
template void TurboVlaTrace::write<uint8_t>(
    const std::string &, const uint8_t *, size_t, const std::vector<int64_t> &,
    const char *, const char *, const char *, const char *, const char *, const char *);

} // namespace vla
