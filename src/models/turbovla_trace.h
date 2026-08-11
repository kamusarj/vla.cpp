// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

namespace vla {

// Writes a TurboVLA trace in the same manifest/raw-tensor format as the
// reference PyTorch TraceContext. Tracing is intentionally opt-in.
class TurboVlaTrace {
public:
    TurboVlaTrace(const char * root, uint64_t forward_index);
    ~TurboVlaTrace();

    TurboVlaTrace(const TurboVlaTrace &) = delete;
    TurboVlaTrace & operator=(const TurboVlaTrace &) = delete;

    bool active() const { return active_; }
    bool includes_layer() const { return level_ >= 1; }
    bool includes_op() const { return level_ >= 2; }
    bool includes_exhaustive() const { return level_ >= 3; }
    bool includes(const char * required_level) const;

    void f32(const std::string & name, const float * data, size_t count,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation, const char * required_level = "boundary");
    void f32(const std::string & name, const float * data, size_t count,
             const std::vector<int64_t> & shape, const char * layout,
             const char * operation, const char * required_level = "boundary");
    void i32(const std::string & name, const int32_t * data, size_t count,
             std::initializer_list<int64_t> shape, const char * layout,
             const char * operation, const char * required_level = "boundary");
    void u8(const std::string & name, const uint8_t * data, size_t count,
            std::initializer_list<int64_t> shape, const char * layout,
            const char * operation, const char * required_level = "boundary");

    void tensor_f32(const std::string & name, ggml_backend_t backend,
                    ggml_tensor * tensor, std::initializer_list<int64_t> shape,
                    const char * layout, const char * operation,
                    const char * required_level = "boundary");
    void tensor_numeric_f32(const std::string & name, ggml_backend_t backend,
                    ggml_tensor * tensor, const std::vector<int64_t> & shape,
                    const char * layout, const char * operation,
                    bool round_to_bf16,
                    const char * required_level = "exhaustive");
    void tensor_f32(const std::string & name, ggml_backend_t backend,
                    ggml_tensor * tensor, const std::vector<int64_t> & shape,
                    const char * layout, const char * operation,
                    const char * required_level = "boundary");
    void tensor_f32_subtract_global_max(
                    const std::string & name, ggml_backend_t backend,
                    ggml_tensor * tensor, ggml_tensor * max_tensor,
                    const std::vector<int64_t> & shape,
                    const char * layout, const char * operation,
                    const char * required_level = "boundary");
    void tensor_f32_subtract_row_max(
                    const std::string & name, ggml_backend_t backend,
                    ggml_tensor * tensor, const std::vector<int64_t> & shape,
                    const char * layout, const char * operation,
                    const char * required_level = "boundary");
    void tensor_f32_layer_norm_diagnostics(
                    const std::string & prefix, ggml_backend_t backend,
                    ggml_tensor * tensor, const std::vector<int64_t> & shape,
                    const char * layout, const char * stat_layout,
                    float eps, bool bf16,
                    const char * required_level = "exhaustive");
    void tensor_f32_softmax_diagnostics(
                    const std::string & prefix, ggml_backend_t backend,
                    ggml_tensor * tensor, const std::vector<int64_t> & shape,
                    const char * layout, bool bf16,
                    const char * required_level = "exhaustive");
    void tensor_f32_fusion_diagnostics(
                    const std::string & prefix, ggml_backend_t backend,
                    ggml_tensor * raw_logits, ggml_tensor * text_mask,
                    int64_t heads, int64_t visual_tokens, int64_t text_tokens,
                    bool bf16, const char * required_level = "exhaustive");

private:
    template <typename T>
    void write(const std::string & name, const T * data, size_t count,
               const std::vector<int64_t> & shape, const char * layout,
               const char * operation, const char * required_level,
               const char * source_dtype, const char * storage_dtype,
               const char * suffix);

    bool accepts(const char * required_level) const;
    void finish();

    bool active_ = false;
    int level_ = 0;
    uint64_t trace_id_ = 0;
    std::filesystem::path trace_dir_;
    std::ofstream manifest_;
    std::ofstream summary_;
    std::vector<std::string> tree_;
};

} // namespace vla
