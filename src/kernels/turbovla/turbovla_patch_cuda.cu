#include "turbovla_patch_cuda.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>

namespace {

constexpr int kChannels = 3;
constexpr int kImage = 256;
constexpr int kPatch = 16;
constexpr int kOutputChannels = 768;
constexpr int kOutputSide = 16;
constexpr int kPatches = kOutputSide*kOutputSide;

__global__ void f32_to_bf16(const float * input, __nv_bfloat16 * output, size_t count) {
    const size_t index = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2bfloat16(input[index]);
    }
}

__global__ void add_bias_bf16(__nv_bfloat16 * output, const __nv_bfloat16 * bias) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    constexpr int count = kOutputChannels*kPatches;
    if (index < count) {
        const int channel = index/kPatches;
        output[index] = __float2bfloat16(
            __bfloat162float(output[index]) + __bfloat162float(bias[channel]));
    }
}

__global__ void nchw_bf16_to_token_f32(const __nv_bfloat16 * input, float * output) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    constexpr int count = kOutputChannels*kPatches;
    if (index < count) {
        const int channel = index/kPatches;
        const int patch = index - channel*kPatches;
        output[patch*kOutputChannels + channel] = __bfloat162float(input[index]);
    }
}

bool check_cuda(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "vla(turbovla): %s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool check_cudnn(cudnnStatus_t status, const char * operation) {
    if (status == CUDNN_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "vla(turbovla): %s failed: %s\n", operation,
                 cudnnGetErrorString(status));
    return false;
}

} // namespace

struct TurboVlaPatchCudaContext {
    float * input_f32 = nullptr;
    __nv_bfloat16 * input_bf16 = nullptr;
    __nv_bfloat16 * bias_bf16 = nullptr;
    __nv_bfloat16 * output_bf16 = nullptr;
    void * workspace = nullptr;
    size_t workspace_size = 0;
    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
};

void turbovla_patch_embed_cuda_destroy(TurboVlaPatchCudaContext * context) {
    if (!context) return;
    if (context->workspace) cudaFree(context->workspace);
    if (context->conv_desc) cudnnDestroyConvolutionDescriptor(context->conv_desc);
    if (context->filter_desc) cudnnDestroyFilterDescriptor(context->filter_desc);
    if (context->output_desc) cudnnDestroyTensorDescriptor(context->output_desc);
    if (context->input_desc) cudnnDestroyTensorDescriptor(context->input_desc);
    if (context->handle) cudnnDestroy(context->handle);
    if (context->output_bf16) cudaFree(context->output_bf16);
    if (context->bias_bf16) cudaFree(context->bias_bf16);
    if (context->input_bf16) cudaFree(context->input_bf16);
    if (context->input_f32) cudaFree(context->input_f32);
    delete context;
}

TurboVlaPatchCudaContext * turbovla_patch_embed_cuda_create(
    const float * bias_f32_device) {
    constexpr size_t input_count = size_t(kChannels)*kImage*kImage;
    constexpr size_t output_count = size_t(kOutputChannels)*kPatches;
    constexpr int threads = 256;
    constexpr cudnnConvolutionFwdAlgo_t algorithm =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    auto * context = new TurboVlaPatchCudaContext();

#define CUDA_TRY(expr, label) do { if (!check_cuda((expr), (label))) goto fail; } while (0)
#define CUDNN_TRY(expr, label) do { if (!check_cudnn((expr), (label))) goto fail; } while (0)

    CUDA_TRY(cudaMalloc(&context->input_f32, input_count*sizeof(float)),
             "cudaMalloc patch input F32");
    CUDA_TRY(cudaMalloc(&context->input_bf16, input_count*sizeof(__nv_bfloat16)),
             "cudaMalloc patch input BF16");
    CUDA_TRY(cudaMalloc(&context->bias_bf16, kOutputChannels*sizeof(__nv_bfloat16)),
             "cudaMalloc patch bias BF16");
    CUDA_TRY(cudaMalloc(&context->output_bf16, output_count*sizeof(__nv_bfloat16)),
             "cudaMalloc patch output BF16");
    f32_to_bf16<<<(kOutputChannels + threads - 1)/threads, threads>>>(
        bias_f32_device, context->bias_bf16, kOutputChannels);
    CUDA_TRY(cudaGetLastError(), "convert patch bias to BF16");

    CUDNN_TRY(cudnnCreate(&context->handle), "cudnnCreate");
    CUDNN_TRY(cudnnCreateTensorDescriptor(&context->input_desc), "create input descriptor");
    CUDNN_TRY(cudnnCreateTensorDescriptor(&context->output_desc), "create output descriptor");
    CUDNN_TRY(cudnnCreateFilterDescriptor(&context->filter_desc), "create filter descriptor");
    CUDNN_TRY(cudnnCreateConvolutionDescriptor(&context->conv_desc), "create convolution descriptor");
    CUDNN_TRY(cudnnSetTensor4dDescriptor(
        context->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
        1, kChannels, kImage, kImage),
        "set input descriptor");
    CUDNN_TRY(cudnnSetFilter4dDescriptor(
        context->filter_desc, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW,
        kOutputChannels, kChannels, kPatch, kPatch), "set filter descriptor");
    CUDNN_TRY(cudnnSetConvolution2dDescriptor(
        context->conv_desc, 0, 0, kPatch, kPatch, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "set convolution descriptor");
    CUDNN_TRY(cudnnSetConvolutionMathType(context->conv_desc, CUDNN_TENSOR_OP_MATH),
              "set convolution tensor-op math");
    CUDNN_TRY(cudnnSetTensor4dDescriptor(
        context->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
        1, kOutputChannels, kOutputSide, kOutputSide), "set output descriptor");

    CUDNN_TRY(cudnnGetConvolutionForwardWorkspaceSize(
        context->handle, context->input_desc, context->filter_desc,
        context->conv_desc, context->output_desc, algorithm,
        &context->workspace_size), "query convolution workspace");
    if (context->workspace_size != 0) {
        CUDA_TRY(cudaMalloc(&context->workspace, context->workspace_size),
                 "cudaMalloc convolution workspace");
    }
    CUDA_TRY(cudaDeviceSynchronize(), "initialize patch context");
    return context;

fail:
    turbovla_patch_embed_cuda_destroy(context);
    return nullptr;

#undef CUDA_TRY
#undef CUDNN_TRY
}

bool turbovla_patch_embed_cuda_two_views(
    TurboVlaPatchCudaContext * context,
    const float * input_host,
    const void * weight_bf16_device,
    float * output_view_0_device,
    float * output_view_1_device) {
    if (!context || !input_host || !weight_bf16_device ||
        !output_view_0_device || !output_view_1_device) return false;
    constexpr size_t input_count = size_t(kChannels)*kImage*kImage;
    constexpr size_t output_count = size_t(kOutputChannels)*kPatches;
    constexpr int threads = 256;
    constexpr cudnnConvolutionFwdAlgo_t algorithm =
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float * outputs[2] = {output_view_0_device, output_view_1_device};

    for (int view = 0; view < 2; ++view) {
        if (!check_cuda(cudaMemcpy(
                context->input_f32, input_host + size_t(view)*input_count,
                input_count*sizeof(float), cudaMemcpyHostToDevice),
                "copy patch input")) return false;
        f32_to_bf16<<<(input_count + threads - 1)/threads, threads>>>(
            context->input_f32, context->input_bf16, input_count);
        if (!check_cuda(cudaGetLastError(), "convert patch input to BF16")) return false;
        if (!check_cudnn(cudnnConvolutionForward(
                context->handle, &alpha, context->input_desc, context->input_bf16,
                context->filter_desc, weight_bf16_device, context->conv_desc,
                algorithm, context->workspace, context->workspace_size, &beta,
                context->output_desc, context->output_bf16),
                "cudnnConvolutionForward")) return false;
        add_bias_bf16<<<(output_count + threads - 1)/threads, threads>>>(
            context->output_bf16, context->bias_bf16);
        nchw_bf16_to_token_f32<<<(output_count + threads - 1)/threads, threads>>>(
            context->output_bf16, outputs[view]);
        if (!check_cuda(cudaGetLastError(), "patch bias/transpose kernels")) return false;
    }
    return check_cuda(cudaDeviceSynchronize(), "complete two-view patch projection");
}
