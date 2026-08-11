cmake_minimum_required(VERSION 3.22)
project(vla_cpp LANGUAGES C CXX)

option(VLA_BUILD_SERVERS "Build ZeroMQ/Protobuf VLA and VLM servers" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
# llama.cpp fetched + pinned at configure; bump = one-line GIT_TAG change.
include(FetchContent)
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        b9866
    GIT_SHALLOW    TRUE
    PATCH_COMMAND  git apply --whitespace=nowarn
                   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/llama-bf16-f32-accumulator.patch
)
FetchContent_MakeAvailable(llama)

add_library(vla_core
    src/model.cpp
    src/models/smolvla.cpp
    src/models/pi0.cpp
    src/models/pi05.cpp
    src/models/evo1.cpp
    src/models/gr00tn1d5.cpp
    src/models/gr00tn1d6.cpp
    src/models/gr00tn1d7.cpp
    src/models/bitvla.cpp
    src/models/vla_adapter.cpp
    src/models/openvla_oft.cpp
    src/models/vla_jepa.cpp
    src/models/turbovla.cpp
    src/models/turbovla_trace.cpp
)
target_include_directories(vla_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/vendor
)
target_link_libraries(vla_core PUBLIC llama ggml)

if(GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_CUDA)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Keep the PyTorch-parity LayerNorm/softmax translation units explicit
    # about precise division, sqrt and denormal handling. The vendored patch
    # also removes target-wide fast math so fused CUDA paths share this policy.
    foreach(turbovla_precise_cuda_source norm.cu softmax.cu)
        set_property(SOURCE ${llama_SOURCE_DIR}/ggml/src/ggml-cuda/${turbovla_precise_cuda_source}
            DIRECTORY ${llama_SOURCE_DIR}/ggml/src/ggml-cuda
            APPEND PROPERTY COMPILE_OPTIONS
            --ftz=false --prec-div=true --prec-sqrt=true)
    endforeach()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Emit SASS per supported GPU (real) and PTX (virtual) only for the newest
        # arch, so future cards can JIT without shipping PTX for every target.
        # Blackwell sm_100/sm_120 require CUDA >= 12.8.
        set(_vla_cuda_archs 80-real 86-real 87-real 89-real 90-real)
        set(_vla_cuda_ptx 90-virtual)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND _vla_cuda_archs 100-real 120-real)
            set(_vla_cuda_ptx 120-virtual)
        else()
            message(WARNING
                "CUDA ${CUDAToolkit_VERSION} < 12.8: omitting Blackwell sm_100/sm_120. "
                "Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly to override.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${_vla_cuda_archs} ${_vla_cuda_ptx}
            CACHE STRING "" FORCE)
    endif()
    add_library(bitvla_cuda_kernels STATIC
        src/kernels/bitvla/bitnet_kernels.cu
        src/kernels/bitvla/bitvla_lm_cuda.cu
        src/kernels/bitvla/bitvla_vit_cuda.cu
        src/kernels/bitvla/bitvla_fp32head_cuda.cu
    )

    # PyTorch dispatches TurboVLA's fixed BF16 patch projection through
    # cuDNN's implicit-precomputed-GEMM convolution.  The NVIDIA pip/conda
    # package does not install cuDNN into the CUDA toolkit prefix, so permit
    # its package root to be supplied explicitly.
    set(VLA_CUDNN_ROOT "" CACHE PATH
        "Root containing include/cudnn.h and lib/libcudnn.so for TurboVLA parity")
    find_path(VLA_CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${VLA_CUDNN_ROOT} $ENV{CUDNN_ROOT}
        PATH_SUFFIXES include)
    find_library(VLA_CUDNN_LIBRARY
        NAMES cudnn libcudnn.so.9 libcudnn.so.8
        HINTS ${VLA_CUDNN_ROOT} $ENV{CUDNN_ROOT}
        PATH_SUFFIXES lib)
    if(VLA_CUDNN_INCLUDE_DIR AND VLA_CUDNN_LIBRARY)
        target_sources(bitvla_cuda_kernels PRIVATE
            src/kernels/turbovla/turbovla_patch_cuda.cu)
        target_include_directories(bitvla_cuda_kernels PRIVATE
            ${VLA_CUDNN_INCLUDE_DIR})
        target_link_libraries(bitvla_cuda_kernels PUBLIC ${VLA_CUDNN_LIBRARY})
        target_compile_definitions(vla_core PUBLIC VLA_TURBOVLA_CUDNN)
        get_filename_component(VLA_CUDNN_LIBRARY_DIR ${VLA_CUDNN_LIBRARY} DIRECTORY)
        set_property(TARGET bitvla_cuda_kernels APPEND PROPERTY
            BUILD_RPATH ${VLA_CUDNN_LIBRARY_DIR})
        message(STATUS "vla: TurboVLA cuDNN parity kernel enabled (${VLA_CUDNN_LIBRARY})")
    else()
        message(STATUS
            "vla: cuDNN not found; TurboVLA patch projection uses ggml fallback "
            "(set VLA_CUDNN_ROOT to enable PyTorch parity)")
    endif()
    set_target_properties(bitvla_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(bitvla_cuda_kernels PRIVATE cxx_std_17)
    target_compile_options(bitvla_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-O3>
    )
    target_link_libraries(bitvla_cuda_kernels PUBLIC CUDA::cublas CUDA::cudart)
    target_link_libraries(vla_core PRIVATE bitvla_cuda_kernels)
    target_compile_definitions(vla_core PUBLIC VLA_BITVLA_CUDA_KERNELS)
    target_include_directories(vla_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(GGML_METAL AND NOT GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_METAL)
endif()

if(VLA_BUILD_SERVERS)
add_library(vlm_core
    src/vlm/engine.cpp
)
target_include_directories(vlm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE
        ${llama_SOURCE_DIR}/common
        ${llama_SOURCE_DIR}/tools/mtmd
        ${llama_SOURCE_DIR}/vendor
)
target_link_libraries(vlm_core PUBLIC llama ggml mtmd llama-common)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

# cppzmq (zmq.hpp) is the header-only C++ binding, packaged separately from the
# libzmq C library (Debian/Ubuntu: cppzmq-dev). The servers #include <zmq.hpp>.
find_path(CPPZMQ_INCLUDE_DIR NAMES zmq.hpp)
if(NOT CPPZMQ_INCLUDE_DIR)
    message(FATAL_ERROR
        "cppzmq (zmq.hpp) not found. Install 'cppzmq-dev' (Debian/Ubuntu) or set "
        "CMAKE_PREFIX_PATH to your cppzmq install prefix.")
endif()

set(VLA_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/serving/vla.proto)
set(VLA_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLA_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLA_PROTO_GEN_DIR}/vla.pb.h
            ${VLA_PROTO_GEN_DIR}/vla.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/serving
            --cpp_out=${VLA_PROTO_GEN_DIR}
            ${VLA_PROTO_SRC}
    DEPENDS ${VLA_PROTO_SRC}
    VERBATIM
)

add_executable(vla-server
    src/serving/server.cpp
    ${VLA_PROTO_GEN_DIR}/vla.pb.cc
)
target_include_directories(vla-server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${llama_SOURCE_DIR}/vendor/stb
    ${CPPZMQ_INCLUDE_DIR}
)
target_link_libraries(vla-server PRIVATE
    vla_core
    protobuf::libprotobuf
    PkgConfig::ZeroMQ
)

set(VLM_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/serving/vlm.proto)
set(VLM_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLM_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLM_PROTO_GEN_DIR}/vlm.pb.h
            ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/serving
            --cpp_out=${VLM_PROTO_GEN_DIR}
            ${VLM_PROTO_SRC}
    DEPENDS ${VLM_PROTO_SRC}
    VERBATIM
)

add_executable(vlm-server
    src/serving/vlm-server.cpp
    ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
)
target_include_directories(vlm-server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${CPPZMQ_INCLUDE_DIR}
)
target_link_libraries(vlm-server PRIVATE
    vlm_core
    protobuf::libprotobuf
    PkgConfig::ZeroMQ
)
endif()

# One-shot inference CLI: image + tokens -> action, no server or simulator.
add_executable(vla-cli
    src/serving/vla-cli.cpp
)
target_include_directories(vla-cli PRIVATE
    ${llama_SOURCE_DIR}/vendor/stb
)
target_link_libraries(vla-cli PRIVATE vla_core)

# --- First-party build hygiene (never applied to the vendored llama.cpp subtree) --
set(VLA_FIRST_PARTY_TARGETS vla_core vla-cli)
if(VLA_BUILD_SERVERS)
    list(APPEND VLA_FIRST_PARTY_TARGETS vlm_core vla-server vlm-server)
endif()

foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
    # Warn on our own C++ only; nvcc device code keeps its own diagnostics.
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endforeach()

# Link-time optimization on Release builds when the toolchain supports it.
include(CheckIPOSupported)
check_ipo_supported(RESULT VLA_IPO_OK OUTPUT VLA_IPO_MSG)
if(VLA_IPO_OK AND CMAKE_BUILD_TYPE STREQUAL "Release")
    foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
        set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    endforeach()
else()
    message(STATUS "vla: IPO/LTO disabled (${VLA_IPO_MSG})")
endif()

option(VLA_BUILD_TESTS "Build vla.cpp test harnesses and unit tests" OFF)
if(VLA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
