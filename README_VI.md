# vla.cpp — hướng dẫn build và sử dụng bằng tiếng Việt

[English](README.md) | **Tiếng Việt**

![logo](assets/logo_vlacpp_white.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
[![Built on llama.cpp](https://img.shields.io/badge/built%20on-llama.cpp-lightgrey)](https://github.com/ggml-org/llama.cpp)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20models-Hugging%20Face-yellow)](https://huggingface.co/vrfai)
[![arXiv](https://img.shields.io/badge/arXiv-2606.08094-b31b1b.svg)](http://arxiv.org/abs/2606.08094)
[![Docs](https://img.shields.io/badge/docs-Learn%20vla.cpp-brightgreen)](https://fai-modelopt-tech.github.io/learn-vla-cpp/)

`vla.cpp` là engine suy luận C++ dành cho các mô hình **Vision-Language-Action
(VLA)**, được xây dựng trên [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
Dự án chạy nhiều policy VLA mở như SmolVLA, π0, BitVLA, Evo-1, GR00T
N1.5/1.6/1.7 và các mô hình khác trong cùng một runtime. Mỗi mô hình được đóng
gói thành một file GGUF độc lập, không cần Python hoặc PyTorch khi suy luận.

Các binary có thể điều khiển robot bằng CPU, Apple Silicon hoặc CUDA, từ GPU
phổ thông đến các bo mạch NVIDIA Jetson. Tài liệu
[Learn vla.cpp](https://fai-modelopt-tech.github.io/learn-vla-cpp/) giải thích
thiết kế engine và cách triển khai từng policy trên ggml.

---

## 1. Build vla.cpp

### 1.1. Yêu cầu

- CMake 3.22 trở lên.
- Trình biên dịch hỗ trợ C++17: GCC 11+ hoặc Clang 14+.
- Git và kết nối Internet trong lần cấu hình đầu tiên. CMake tự tải phiên bản
  `llama.cpp` đã được cố định trong [`CMakeLists.txt`](CMakeLists.txt).
- ZeroMQ, cppzmq và Protobuf.
- CUDA 12.x chỉ cần khi build cho GPU NVIDIA.

Trên Ubuntu/Debian, cài đầy đủ công cụ build:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libzmq3-dev cppzmq-dev \
    libprotobuf-dev protobuf-compiler
```

Kiểm tra phiên bản:

```bash
cmake --version
g++ --version
protoc --version
```

### 1.2. Build cho CPU

Từ thư mục gốc của repo:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
```

Các chương trình chính sau khi build:

```text
build-cpu/vla-cli
build-cpu/vla-server
build-cpu/vlm-server
```

### 1.3. Build CUDA cho GPU NVIDIA

CUDA build yêu cầu cả NVIDIA driver và CUDA Toolkit có `nvcc`:

```bash
nvidia-smi
nvcc --version
```

Chọn compute capability phù hợp:

| Dòng GPU | Ví dụ | `CMAKE_CUDA_ARCHITECTURES` |
|---|---|---:|
| Ampere (Jetson) | Orin Nano, Orin NX | `87` |
| Ampere (phổ thông) | RTX 30 series, A40 | `86` |
| Ada Lovelace | RTX 40 series, L40 | `89` |
| Hopper | H100, H200 | `90` |
| Blackwell (phổ thông) | RTX 50 series | `120` |
| Blackwell (datacenter) | B100, B200, GB200 | `100` |

Ví dụ với RTX 30 series:

```bash
export CUDA_ARCHITECTURE=86

cmake -S . -B build-cuda \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"
cmake --build build-cuda -j"$(nproc)"
```

Nếu CMake không tìm thấy CUDA:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
```

Sau đó cấu hình lại thư mục `build-cuda`. Nếu trước đó thư mục này đã được cấu
hình sai, có thể tạo một thư mục build mới, ví dụ `build-cuda-new`, để tránh dùng
lại cache CMake cũ.

CUDA 12.8 trở lên cần thiết cho Blackwell `sm_100` và `sm_120`. Tài liệu build
trên WSL2 nằm tại [`docs/backend/wsl.md`](docs/backend/wsl.md).

### 1.4. Build trên macOS/Apple Silicon

Cài phụ thuộc:

```bash
brew install cmake protobuf zeromq cppzmq pkg-config
```

Metal được bật mặc định:

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j"$(sysctl -n hw.ncpu)"
```

Muốn tắt Metal khi build, thêm `-DGGML_METAL=OFF`. Xem thêm
[`docs/backend/metal.md`](docs/backend/metal.md).

### 1.5. Kiểm tra kết quả build

Chọn thư mục binary tương ứng với máy:

```bash
# Chọn đúng một dòng:
# export VLA_BUILD=build-cpu
export VLA_BUILD=build-cuda
# export VLA_BUILD=build-metal

ls -lh "$VLA_BUILD/vla-cli" "$VLA_BUILD/vla-server"
```

---

## 2. Bắt đầu nhanh với SmolVLA

### 2.1. Tải checkpoint GGUF đã được chuyển đổi

Khuyến nghị dùng model GGUF phát hành sẵn. Không cần tự convert từ
`model.safetensors`:

```bash
python3 -m pip install -U "huggingface_hub[cli]" gguf
hf download vrfai/smolvla-libero-gguf --local-dir models/smolvla
```

Kiểm tra file:

```bash
ls -lh models/smolvla/smolvla-libero.gguf
```

### 2.2. Chạy một lần suy luận

Nếu chưa đặt `VLA_BUILD`, hãy đặt nó theo thư mục đã build ở phần trên:

```bash
export VLA_BUILD=build-cuda
export VLA_GGUF=models/smolvla/smolvla-libero.gguf
export VLA_ARCH=smolvla

"$VLA_BUILD/vla-cli" \
    --ckpt "$VLA_GGUF" \
    --image assets/front.jpg \
    --tokens 1,100,200,2 \
    --pretty
```

`vla-cli` nhận model, ảnh và các token của câu lệnh rồi in ra một action chunk:

- `--tokens`: ID token ngôn ngữ do tokenizer phía client tạo ra.
- `--pretty`: in mỗi action trên một dòng.
- `--state`: truyền trạng thái proprioception; mặc định là vector 0.

Nếu build bằng CUDA hoặc Metal, đổi `VLA_BUILD` sang thư mục tương ứng.

---

## 3. Chạy server

`vla-server` tải model một lần khi khởi động rồi xử lý tuần tự các request
ZeroMQ theo mô hình REQ/REP:

Trên RTX 3050 6 GB của máy này, dùng bản BF16 đã kiểm thử:

```bash
export VLA_BUILD=build-cuda
export VLA_GGUF="$PWD/models/smolvla/smolvla-libero-bf16.gguf"

GGML_CUDA_DISABLE_GRAPHS=1 \
    "$VLA_BUILD/vla-server" "$VLA_GGUF" \
    --bind tcp://127.0.0.1:5555 --timing-detail phase
```

Bản này dùng khoảng 1.15 GB VRAM khi server đã nạp xong. Biến
`GGML_CUDA_DISABLE_GRAPHS=1` ưu tiên độ ổn định khi chạy benchmark dài.

Khi sẵn sàng, server sẽ in:

```text
vla-server: bound to tcp://*:5555. ready.
```

Dùng `--bind` để đổi địa chỉ hoặc cổng. Nhấn `Ctrl-C` để dừng server.

---

## 4. Cài simulator

Thư mục [`eval/`](eval/) hỗ trợ hai simulator end-to-end. Mỗi script setup tạo
một môi trường Python 3.10 bằng `uv` và clone repo simulator tương ứng. Cần cài
[`uv`](https://github.com/astral-sh/uv) trước.

### 4.1. LIBERO

```bash
bash eval/sim/libero/setup_libero.sh
```

Script clone LIBERO vào `eval/sim/libero/LIBERO/`, tạo môi trường tại
`eval/sim/libero/libero_uv/.venv/` và cố định các phiên bản tương thích của
PyTorch, LeRobot, Transformers và Gymnasium.

### 4.2. SimplerEnv

```bash
bash eval/sim/simpler/setup_SimplerEnv.sh
```

Script clone SimplerEnv cùng `ManiSkill2_real2sim` vào
`eval/sim/simpler/SimplerEnv/` và tạo môi trường tại
`eval/sim/simpler/simpler_uv/.venv/`.

---

## 5. Chạy client

[`eval/client/`](eval/client/) chứa client benchmark LIBERO kết nối trực tiếp
đến `vla-server` bằng giao thức Protobuf. Hãy khởi động server trước.

### 5.1. LIBERO

Trong một terminal khác:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
export VLA_ARCH=smolvla
export TOKENIZERS_PARALLELISM=false

python eval/client/run_sim_client_direct.py \
    --vla-addr tcp://127.0.0.1:5555 \
    --task libero_object \
    --task-id 0 \
    --n-episodes 1 \
    --n-action-steps 4 \
    --output-dir "$PWD/outputs/libero_outputs" \
    --arch "$VLA_ARCH"
```

Kết quả kiểm thử trên máy này: task 0 của `libero_object` hoàn thành sau 142
bước, reward 1.0, success 1/1 và thời gian inference trung bình 58.67 ms/bước.

#### Xem dữ liệu client gửi lên server

Thêm hai tham số sau vào lệnh client để lưu một số request trước khi gửi:

```bash
    --dump-request-dir "$PWD/outputs/request_dumps/smolvla" \
    --dump-request-limit 5
```

Mỗi thư mục `request_XXXXXX/` chứa `request.pb` (đúng byte Protobuf gửi qua
ZeroMQ), `metadata.json`, `state.npy`, `lang_tokens.npy`, cùng `image_N.npy` và
`image_N.png`. File PNG dùng để xem nhanh; file NPY giữ chính xác tensor ảnh
float32 mà server nhận. Chế độ này mặc định tắt để không làm chậm benchmark.

Các model GR00T cần thêm:

- Phía client: `--stats-json "/home/linh/Desktop/vla.cpp/models/vrfai/gr00tn1d6-libero-gguf/dataset_statistics.json"`
  cho checkpoint N1.6 LIBERO cài cục bộ theo cấu trúc của repo.
- Phía server: đặt `VLA_GR00T_EMBODIMENT` thành `new_embodiment` cho N1.5,
  `libero_panda` cho N1.6 hoặc `libero_sim` cho N1.7.
- Đặt `VLA_GR00T_BF16_WEIGHTS=1` để model phù hợp GPU 8 GB.

Ví dụ chạy GR00T-N1.6 LIBERO với bundle đặt tại `models/vrfai/`:

```bash
# Máy này dùng RTX 3050 6 GB: GR00T-N1.6 CUDA load được weights nhưng không
# đủ VRAM cho graph inference, vì vậy dùng CPU build. GPU từ 8 GB trở lên có
# thể đổi lại thành build-cuda.
export VLA_BUILD=build-cpu
export GR00T_DIR=/home/linh/Desktop/vla.cpp/models/vrfai/gr00tn1d6-libero-gguf
export GR00T_N1D6_GGUF="$GR00T_DIR/gr00tn1d6-libero.gguf"
export VLA_STATS_JSON="$GR00T_DIR/dataset_statistics.json"
```

Server:

```bash
VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=libero_panda \
    "$VLA_BUILD/vla-server" "$GR00T_N1D6_GGUF" \
    --bind 'tcp://127.0.0.1:5555' --timing-detail phase
```

Client (terminal khác, chạy từ thư mục gốc repo):

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate

python eval/client/run_sim_client_direct.py \
    --vla-addr tcp://localhost:5555 \
    --arch gr00t_n1_6 \
    --tokenizer "$GR00T_DIR" \
    --stats-json "$VLA_STATS_JSON" \
    --bitvla-unnorm-key libero_panda \
    --task libero_object \
    --task-id 0 \
    --n-episodes 1 \
    --n-action-steps 16 \
    --output-dir /tmp/libero_outputs
```

#### GR00T-N1.5 trên GPU 6 GB

GR00T-N1.6 không đủ VRAM cho graph inference trên RTX 3050 6 GB, nhưng N1.5
có weights resident khoảng 4.48 GiB và chạy CUDA được. Bundle local:

```bash
export GR00T_N15_DIR=/home/linh/Desktop/vla.cpp/models/vrfai/gr00tn1d5-libero-object-gguf
```

Server:

```bash
VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=new_embodiment \
    ./build-cuda/vla-server \
    "$GR00T_N15_DIR/gr00tn1d5-libero-object.gguf" \
    --bind 'tcp://127.0.0.1:5555' --timing-detail phase
```

Client:

> **Quan trọng:** terminal chạy server và terminal chạy client không dùng chung
> biến môi trường. Vì vậy, cần `export GR00T_N15_DIR` lại trong chính terminal
> chạy client. Nếu biến này chưa được đặt, Bash sẽ biến đường dẫn
> `"$GR00T_N15_DIR/dataset_statistics.json"` thành
> `/dataset_statistics.json` và client sẽ báo `FileNotFoundError`.

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export GR00T_N15_DIR="$PWD/models/vrfai/gr00tn1d5-libero-object-gguf"

# Kiểm tra file thống kê tồn tại trước khi chạy mô phỏng.
test -f "$GR00T_N15_DIR/dataset_statistics.json" || {
    echo "Không tìm thấy $GR00T_N15_DIR/dataset_statistics.json"
    exit 1
}

python eval/client/run_sim_client_direct.py \
    --vla-addr tcp://localhost:5555 \
    --arch gr00t_n1_5 \
    --tokenizer lerobot/eagle2hg-processor-groot-n1p5 \
    --stats-json "$GR00T_N15_DIR/dataset_statistics.json" \
    --bitvla-unnorm-key new_embodiment \
    --task libero_object \
    --task-id 0 \
    --n-episodes 1 \
    --n-action-steps 16 \
    --output-dir /tmp/libero_outputs
```

Nếu đang đứng ngoài thư mục gốc của repository, hãy thay `$PWD` bằng đường dẫn
tuyệt đối tới `vla.cpp`, ví dụ:

```bash
export GR00T_N15_DIR=/home/linh/Desktop/vla.cpp/models/vrfai/gr00tn1d5-libero-object-gguf
```

### 5.2. SimplerEnv

Hiện chỉ GR00T-N1.6 được nối sẵn với checkpoint `gr00t-n1d6-bridge` và
embodiment `oxe_widowx`. Khởi động server trên cổng 5566:

```bash
VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=oxe_widowx \
    "$VLA_BUILD/vla-server" "$GR00T_N1D6_GGUF" --bind 'tcp://*:5566'
```

Sau đó chạy client từ môi trường SimplerEnv:

```bash
source eval/sim/simpler/simpler_uv/.venv/bin/activate

python eval/client/run_simpler_client_direct.py \
    --arch gr00t_n1_6 \
    --task-id oxe_widowx/widowx_spoon_on_towel \
    --n-episodes 1 \
    --embodiment oxe_widowx \
    --image-size 252 \
    --stats-json "$VLA_STATS_JSON"
```

---

## 6. Model: chuyển đổi và lượng tử hóa

### 6.1. Khi nào cần convert?

Nếu đã tải file `models/smolvla/smolvla-libero.gguf` ở phần bắt đầu nhanh, bạn
**không cần convert**. Converter chỉ dùng khi có checkpoint Hugging Face gốc,
ví dụ một thư mục chứa ít nhất:

```text
smolvla-libero/
├── model.safetensors
├── config.json
└── policy_*processor.json
```

Các lệnh dưới đây dùng đường dẫn thật của repository trên máy này. Khai báo một
lần để có thể chạy lệnh từ bất kỳ thư mục nào:

```bash
export VLA_REPO=/home/linh/Desktop/vla.cpp
```

Tạo môi trường converter:

```bash
python3 -m venv "$VLA_REPO/.venv-converter"
source "$VLA_REPO/.venv-converter/bin/activate"
python -m pip install -U pip
python -m pip install -e "$VLA_REPO[convert]"
```

Tải checkpoint Hugging Face gốc vào một thư mục riêng với các file GGUF:

```bash
hf download HuggingFaceVLA/smolvla_libero \
    --local-dir "$VLA_REPO/models/huggingface/smolvla-libero"
```

Convert checkpoint SmolVLA gốc:

```bash
export SMOLVLA_HF_DIR="$VLA_REPO/models/huggingface/smolvla-libero"

test -f "$SMOLVLA_HF_DIR/model.safetensors" || {
    echo "Không tìm thấy $SMOLVLA_HF_DIR/model.safetensors"
    exit 1
}

python "$VLA_REPO/scripts/convert_smolvla_to_gguf.py" \
    --ckpt "$SMOLVLA_HF_DIR" \
    --out "$VLA_REPO/models/smolvla/smolvla-libero-bf16.gguf"
```

Dùng `--help` để xem đầy đủ tùy chọn của converter:

```bash
python "$VLA_REPO/scripts/convert_smolvla_to_gguf.py" --help
```

### 6.2. Quantize GGUF

Các file GGUF phát hành sẵn dùng BF16. Script quantize chỉ nén các ma trận
trọng số phù hợp của LM backbone; embedding, output head, norm, action expert
và mặc định cả vision tower vẫn giữ kiểu float.

Với checkpoint đã tải trong hướng dẫn này, dùng đúng đường dẫn thật:

```bash
export VLA_REPO=/home/linh/Desktop/vla.cpp
source "$VLA_REPO/.venv-converter/bin/activate"

python "$VLA_REPO/scripts/quantize_gguf.py" \
    --in "$VLA_REPO/models/smolvla/smolvla-libero.gguf" \
    --out "$VLA_REPO/models/smolvla/smolvla-libero-q8_0.gguf" \
    --type Q8_0
```

Kiểm tra kết quả:

```bash
ls -lh "$VLA_REPO"/models/smolvla/*.gguf
```

Sau đó có thể thử model Q8_0:

```bash
export VLA_BUILD="$VLA_REPO/build-cuda"
export VLA_GGUF="$VLA_REPO/models/smolvla/smolvla-libero-q8_0.gguf"

"$VLA_BUILD/vla-cli" --ckpt "$VLA_GGUF" \
    --image "$VLA_REPO/assets/front.jpg" --tokens 1,100,200,2 --pretty
```

> Lưu ý trên build CUDA hiện tại của máy này: file
> `smolvla-libero-q8_0.gguf` báo `unsupported dtype 8` ở lớp connector. Hãy dùng
> `smolvla-libero-bf16.gguf` để chạy LIBERO; bản BF16 vẫn chỉ dùng khoảng 1.15
> GB VRAM khi server đã nạp xong.

- `Q8_0`: gần như không suy giảm chất lượng và giảm đáng kể dung lượng LM.
- `Q4_0`: nhỏ hơn nữa nhưng có nguy cơ giảm độ chính xác nhiều hơn.
- Thêm `--vision` để quantize cả vision tower; file nhỏ hơn nhưng có thể làm
  giảm độ chính xác.

Không dùng `--in model-bf16.gguf` nếu file đó không thật sự tồn tại. Có thể tìm
các GGUF đang có bằng:

```bash
find "$VLA_REPO/models" -type f -name '*.gguf' -print
```

---

## 7. Xử lý lỗi thường gặp

### `FileNotFoundError: model-bf16.gguf`

Tên `model-bf16.gguf` trong README gốc là ví dụ. Với SmolVLA đã tải theo hướng
dẫn này, đầu vào đúng là:

```text
models/smolvla/smolvla-libero.gguf
```

### `missing .../model.safetensors`

`--ckpt` phải trỏ đến thư mục checkpoint Hugging Face gốc có
`model.safetensors`, không phải đường dẫn minh họa và cũng không phải thư mục chỉ
chứa GGUF. Nếu đã có GGUF phát hành sẵn thì bỏ qua converter.

### CMake không tìm thấy `zmq.hpp`

```bash
sudo apt-get install -y libzmq3-dev cppzmq-dev pkg-config
```

Sau đó cấu hình CMake lại.

### CMake không tìm thấy CUDA hoặc chương trình chạy bằng CPU

```bash
which nvcc
nvcc --version
grep -E 'GGML_CUDA|CMAKE_CUDA_ARCHITECTURES' build-cuda/CMakeCache.txt
```

Đảm bảo `GGML_CUDA=ON`, kiến trúc GPU đúng và `nvcc` nằm trong `PATH` ngay từ
lúc chạy lệnh `cmake -S ...`.

### Build hết RAM hoặc bị dừng

Giảm số tiến trình build, ví dụ:

```bash
cmake --build build-cpu -j2
```

---

## 8. Hiệu năng tham khảo

Độ trễ tính bằng mili giây, gồm suy luận và truyền dữ liệu, đo phía client:

| Model | RTX 3090 | Jetson AGX Orin | Orin Nano 8 GB | Apple M4 |
|---|---:|---:|---:|---:|
| `smolvla` | 86 | 262 | 567 | 888 |
| `pi0` | 264 | 893 | 1955 | 1135 |
| `gr00t_n1_5` | 109 | 461 | 1356 | - |
| `gr00t_n1_7` | 102 | 429 | - | 755 |
| `bitvla` | 145 | 809 | 2845 | - |
| `evo1` | 238 | 1048 | 3671 | - |

---

## 9. Mức độ hỗ trợ model và nền tảng

Ký hiệu: `Y` là đã phát hành và benchmark, `~` là đang phát triển, `-` là dự
kiến hỗ trợ.

| Model | CPU (x86-64/ARM) | CUDA | Metal | OpenVINO | Hexagon |
|---|:---:|:---:|:---:|:---:|:---:|
| [SmolVLA](https://hf.co/vrfai/smolvla-libero-gguf) | Y | Y | Y | - | - |
| [π0](https://hf.co/vrfai/pi0-libero-finetuned-v044-gguf) | Y | Y | Y | - | - |
| [π0.5](https://hf.co/vrfai/pi05-libero-gguf) | Y | Y | ~ | - | - |
| [GR00T N1.5](https://hf.co/vrfai/gr00tn1d5-libero-object-gguf) | Y | Y | ~ | - | - |
| [GR00T N1.6](https://hf.co/vrfai/gr00tn1d6-libero-gguf) | Y | Y | ~ | - | - |
| [GR00T N1.7](https://hf.co/vrfai/gr00tn1d7-libero-gguf) | Y | Y | Y | - | - |
| [BitVLA](https://hf.co/vrfai/bitvla-libero-gguf) | Y | Y | ~ | - | - |
| [Evo-1](https://hf.co/vrfai/evo1-libero-gguf)* | Y | Y | ~ | - | - |
| [VLA-Adapter](https://hf.co/vrfai/vla-adapter-libero-gguf) | Y | Y | ~ | - | - |
| [OpenVLA-OFT](https://hf.co/vrfai/openvla-oft-libero-gguf) | Y | Y | ~ | - | - |
| [VLA-JEPA](https://hf.co/vrfai/vla-jepa-libero) | Y | Y | ~ | - | - |

\* Evo-1 tải và chạy được, nhưng GGUF đã phát hành đạt 0% trên `libero_object`
thay vì 94,5% như báo cáo. Chưa nên dùng model này cho tác vụ yêu cầu tỷ lệ
thành công thực tế.

Các lỗi và lưu ý riêng cho từng model nằm trong
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

---

## 10. Người đóng góp

- [Khanh Dang Nguyen](https://github.com/khanhnd61-vr)
- [Hung Thinh Ho](https://github.com/hungho77)
- [Chinh Truong Nguyen](https://github.com/nguyentruongchinh04z)
- [An Thai Le](https://github.com/anindex)

---

## 11. Giấy phép

Dự án được phát hành theo [Apache License, Version 2.0](LICENSE.md).

---

## 12. Ghi nhận

Các model VLA được hỗ trợ:

- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — nhóm Hugging Face LeRobot.
- [π0, π0.5](https://github.com/Physical-Intelligence/openpi) — Physical Intelligence.
- [BitVLA](https://github.com/ustcwhy/BitVLA) — Hongyu Wang và cộng sự.
- [Evo-1](https://github.com/MINT-SJTU/Evo-1/tree/main) — Tao Lin và cộng sự.
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) — Yihao Wang và cộng sự.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft) — Moo Jin Kim và cộng sự.
- [GR00T N1.x](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA Isaac.
- [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) — Jingwen Sun và cộng sự.

Dự án được xây dựng trên:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — engine suy luận LLM bằng C/C++.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — bộ benchmark đánh giá tỷ lệ thành công.
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) — simulator thứ hai trong hệ thống đánh giá.
