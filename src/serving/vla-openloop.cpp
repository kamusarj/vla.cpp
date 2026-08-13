// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

// Persistent batch inference runner for deterministic recorded-dataset
// open-loop evaluation. Python prepares processor-exact tensors and raw RGB
// frames once; this process loads the GGUF once and evaluates every request.

#include "model.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace vla;

namespace {

template <typename T>
bool read_exact(const std::filesystem::path & path, size_t count, std::vector<T> & out) {
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "vla-openloop: cannot open %s\n", path.c_str()); return false; }
    if (std::fseek(fp, 0, SEEK_END) != 0) { std::fclose(fp); return false; }
    const long bytes = std::ftell(fp);
    if (bytes < 0 || (size_t) bytes != count * sizeof(T) || std::fseek(fp, 0, SEEK_SET) != 0) {
        std::fprintf(stderr, "vla-openloop: %s has %ld bytes, expected %zu\n",
                     path.c_str(), bytes, count * sizeof(T));
        std::fclose(fp); return false;
    }
    out.resize(count);
    const bool ok = count == 0 || std::fread(out.data(), sizeof(T), count, fp) == count;
    std::fclose(fp);
    if (!ok) std::fprintf(stderr, "vla-openloop: short read from %s\n", path.c_str());
    return ok;
}

void usage(const char * program) {
    std::fprintf(stderr, "usage: %s --ckpt model.gguf --fixture DIR --actions actions.f32\n", program);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string ckpt_path, fixture_path, action_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "vla-openloop: %s needs a value\n", name); std::exit(1); }
            return argv[++i];
        };
        if      (arg == "--ckpt")    ckpt_path = need("--ckpt");
        else if (arg == "--fixture") fixture_path = need("--fixture");
        else if (arg == "--actions") action_path = need("--actions");
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "vla-openloop: unknown argument %s\n", arg.c_str()); usage(argv[0]); return 1; }
    }
    if (ckpt_path.empty() || fixture_path.empty() || action_path.empty()) { usage(argv[0]); return 1; }

    const std::filesystem::path fixture(fixture_path);
    FILE * meta = std::fopen((fixture / "meta.txt").c_str(), "r");
    long long request_count=0, token_count=0, view_count=0, image_h=0, image_w=0;
    long long state_dim=0, action_horizon=0, action_dim=0;
    if (!meta || std::fscanf(meta, "%lld %lld %lld %lld %lld %lld %lld %lld",
                            &request_count, &token_count, &view_count, &image_h, &image_w,
                            &state_dim, &action_horizon, &action_dim) != 8) {
        std::fprintf(stderr, "vla-openloop: invalid %s\n", (fixture / "meta.txt").c_str());
        if (meta) std::fclose(meta);
        return 1;
    }
    std::fclose(meta);
    if (request_count <= 0 || token_count <= 0 || view_count <= 0 || image_h <= 0 || image_w <= 0 ||
        state_dim <= 0 || action_horizon <= 0 || action_dim <= 0) {
        std::fprintf(stderr, "vla-openloop: fixture dimensions must be positive\n"); return 1;
    }

    const size_t requests = (size_t) request_count, tokens_per_request = (size_t) token_count;
    const size_t views = (size_t) view_count, pixels_per_view = (size_t) image_h * (size_t) image_w * 3;
    const size_t states_per_request = (size_t) state_dim;
    const size_t actions_per_request = (size_t) action_horizon * (size_t) action_dim;
    std::vector<int32_t> tokens;
    std::vector<float> states, noise;
    std::vector<uint8_t> images;
    if (!read_exact(fixture / "tokens.i32", requests * tokens_per_request, tokens) ||
        !read_exact(fixture / "states.f32", requests * states_per_request, states) ||
        !read_exact(fixture / "noise.f32", requests * actions_per_request, noise) ||
        !read_exact(fixture / "images.u8", requests * views * pixels_per_view, images)) return 1;
    for (float value : states) if (!std::isfinite(value)) { std::fprintf(stderr, "vla-openloop: non-finite state\n"); return 1; }
    for (float value : noise)  if (!std::isfinite(value)) { std::fprintf(stderr, "vla-openloop: non-finite noise\n"); return 1; }

    Model * model = model_load("", ckpt_path, "");
    if (!model) { std::fprintf(stderr, "vla-openloop: model_load failed\n"); return 1; }
    const Config & config = model_config(model);
    if (config.max_state_dim != state_dim || config.n_suffix != action_horizon || config.max_action_dim != action_dim) {
        std::fprintf(stderr,
            "vla-openloop: fixture/model mismatch: state=%lld/%lld horizon=%lld/%lld action=%lld/%lld\n",
            state_dim, (long long) config.max_state_dim, action_horizon, (long long) config.n_suffix,
            action_dim, (long long) config.max_action_dim);
        model_free(model); return 1;
    }

    FILE * actions = std::fopen(action_path.c_str(), "wb");
    const std::filesystem::path timing_path = fixture / "timings.csv";
    FILE * timings = std::fopen(timing_path.c_str(), "w");
    if (!actions || !timings) {
        std::fprintf(stderr, "vla-openloop: cannot create output files\n");
        if (actions) std::fclose(actions);
        if (timings) std::fclose(timings);
        model_free(model);
        return 1;
    }
    std::fprintf(timings, "request,total_ms,vision_ms,inference_ms\n");

    std::vector<ImageView> image_views(views);
    double total_ms = 0.0;
    for (size_t request = 0; request < requests; ++request) {
        for (size_t view = 0; view < views; ++view) {
            const size_t offset = (request * views + view) * pixels_per_view;
            image_views[view] = ImageView{images.data() + offset, (int) image_w, (int) image_h, PixelFormat::U8};
        }
        Inputs input{};
        input.images = image_views.data(); input.n_images = (int) views;
        input.lang_tokens = tokens.data() + request * tokens_per_request; input.n_lang = (int) tokens_per_request;
        input.state = states.data() + request * states_per_request;
        input.noise = noise.data() + request * actions_per_request;
        input.timing_detail = TimingDetail::PHASE;
        std::vector<float> prediction = predict(model, input);
        if (prediction.size() != actions_per_request ||
            std::fwrite(prediction.data(), sizeof(float), prediction.size(), actions) != prediction.size()) {
            std::fprintf(stderr, "vla-openloop: request %zu failed\n", request);
            std::fclose(actions); std::fclose(timings); model_free(model); return 2;
        }
        const Stats & stats = last_stats(model);
        total_ms += stats.ms_total;
        std::fprintf(timings, "%zu,%.6f,%.6f,%.6f\n", request, stats.ms_total, stats.ms_vision, stats.ms_inference);
        std::printf("vla-openloop: request %zu/%zu total=%.1f ms vision=%.1f ms inference=%.1f ms\n",
                    request + 1, requests, stats.ms_total, stats.ms_vision, stats.ms_inference);
        std::fflush(stdout);
    }
    std::fclose(actions); std::fclose(timings);
    model_free(model);
    std::printf("vla-openloop: done requests=%zu mean_total=%.1f ms actions=%s timings=%s\n",
                requests, total_ms / (double) requests, action_path.c_str(), timing_path.c_str());
    return 0;
}
