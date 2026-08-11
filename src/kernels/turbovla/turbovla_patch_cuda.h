#pragma once

#include <cstddef>

struct TurboVlaPatchCudaContext;

// Persistent fixed-shape TurboVLA/DINOv3 BF16 patch projection. Resources
// that are expensive to create (cuDNN handle/descriptors and CUDA workspace)
// live for the model lifetime rather than one camera view.
TurboVlaPatchCudaContext * turbovla_patch_embed_cuda_create(
    const float * bias_f32_device);
void turbovla_patch_embed_cuda_destroy(TurboVlaPatchCudaContext * context);

// Input contains two host F32 NCHW views back-to-back. Each output is a device
// F32 ggml input tensor in token-major [256, 768] order. The call synchronizes
// before returning so the following ggml graph can safely use both outputs.
bool turbovla_patch_embed_cuda_two_views(
    TurboVlaPatchCudaContext * context,
    const float * input_host,
    const void * weight_bf16_device,
    float * output_view_0_device,
    float * output_view_1_device);
