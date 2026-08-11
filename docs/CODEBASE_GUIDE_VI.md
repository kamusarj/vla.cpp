# Hướng dẫn đọc hiểu codebase và technical report vla.cpp

Tài liệu này giải thích kiến trúc của `vla.cpp`, ánh xạ các ý tưởng trong
[technical report](https://arxiv.org/abs/2606.08094) vào source code, và chỉ ra
những khác biệt giữa report, tài liệu trong repo và implementation hiện tại.

> Phiên bản code được đọc: commit `ac4a468` trên nhánh `main`.

## 1. vla.cpp là gì?

`vla.cpp` là runtime inference C++ cho các mô hình Vision-Language-Action
(VLA), xây dựng trên `ggml` và `llama.cpp`.

Nó không phải framework để train VLA. Nhiệm vụ của runtime là biến nhiều kiến
trúc VLA khác nhau thành cùng một giao diện:

```text
camera images + language instruction + robot state
                         |
                         v
                 vla::predict(...)
                         |
                         v
              action chunk [H, D]
```

Một VLA khác LLM thông thường ở chỗ output không phải token tiếp theo. Runtime
phải lấy hidden states của backbone, chạy một action head riêng, đôi khi tích
phân qua nhiều bước flow matching hoặc diffusion, rồi trả về cả một đoạn hành
động.

## 2. Mental model tổng thể

Đường inference end-to-end có thể hiểu như sau:

```text
Python simulator/client
  |- resize và chuẩn hóa ảnh
  |- tokenize instruction
  |- pad/normalize robot state
  `- tạo Protobuf request
          |
          | ZeroMQ REQ/REP
          v
C++ vla-server
  |- kiểm tra request
  |- decode ảnh
  |- tạo vla::Inputs
  `- gọi vla::predict()
          |
          v
src/model.cpp
  |- đọc metadata của GGUF
  |- nhận diện architecture
  `- dispatch tới implementation tương ứng
          |
          v
src/models/<architecture>.cpp
  |- vision encoder
  |- multimodal prefix
  |- action head / solver
  `- action chunk
          |
          v
Python client
  |- reshape [chunk_size, action_dim]
  |- postprocess nếu model yêu cầu
  `- thực thi N action rồi replan
```

Các điểm vào quan trọng:

- [`src/model.h`](../src/model.h): public C++ API.
- [`src/arch.h`](../src/arch.h): danh sách architecture và interface chung.
- [`src/model.cpp`](../src/model.cpp): detect architecture và dispatch.
- [`src/models/`](../src/models): implementation của từng model.
- [`src/serving/vla.proto`](../src/serving/vla.proto): wire protocol.
- [`src/serving/server.cpp`](../src/serving/server.cpp): inference server.
- [`eval/client/vla_cpp_client.py`](../eval/client/vla_cpp_client.py): client và
  preprocessing theo từng architecture.

## 3. Các tầng của codebase

### 3.1 Public API

[`src/model.h`](../src/model.h) định nghĩa bốn khái niệm chính:

- `Config`: shape và hyperparameter đã đọc từ checkpoint.
- `Inputs`: ảnh, token ngôn ngữ, robot state, noise và attention mask.
- `Stats`: latency của vision, prefill, denoise và toàn request.
- `Model`: opaque handle của model đã load.

API chính:

```cpp
Model * model_load(...);
const Config & model_config(const Model * model);
std::vector<float> predict(Model * model, const Inputs & inputs);
const Stats & last_stats(const Model * model);
void model_free(Model * model);
```

`Model` chỉ là wrapper sở hữu một `ModelArchBase`. `predict()` được chuyển tiếp
đến virtual method của architecture thực tế.

### 3.2 Architecture dispatch

[`src/model.cpp`](../src/model.cpp) đọc metadata như
`general.architecture`, `smolvla.architecture`, `evo1.architecture`, ... từ
GGUF. Sau đó nó gọi factory tương ứng:

```text
smolvla      -> smolvla_create(...)
pi0          -> pi0_create(...)
evo1         -> evo1_create(...)
gr00t_n1_7   -> gr00t_n1_7_create(...)
bitvla       -> bitvla_create(...)
...
```

Repo hiện tại hỗ trợ 11 architecture:

- SmolVLA;
- pi0;
- pi0.5;
- Evo-1;
- GR00T N1.5, N1.6 và N1.7;
- BitVLA;
- VLA-Adapter;
- OpenVLA-OFT;
- VLA-JEPA.

Technical report chỉ đánh giá bảy architecture đầu tiên của phiên bản cũ hơn:
SmolVLA, pi0, Evo-1, BitVLA và ba bản GR00T.

### 3.3 Implementation theo architecture

Mỗi file trong [`src/models/`](../src/models) tự sở hữu:

- model weights;
- `ggml_context`;
- CPU/CUDA/Metal backend;
- compute graph;
- vision preprocessing;
- attention mask;
- action solver;
- cache và timing;
- một phần normalization/postprocessing.

Codebase thống nhất interface nhưng chưa trừu tượng hóa sâu implementation. Cách
này giúp port một architecture mới khá trực tiếp, đổi lại các file model lớn và
có một phần logic trùng lặp.

### 3.4 GGUF reader và deployment bundle

[`src/models/gguf_reader.h`](../src/models/gguf_reader.h) cung cấp loader chung
để:

- đọc GGUF metadata;
- đọc tensor F32/BF16;
- giữ tensor quantized ở dạng packed;
- fetch một số embedding row trực tiếp từ file;
- chuyển BF16 sang F32 khi cần.

Các converter trong [`scripts/`](../scripts) biến checkpoint Hugging Face hoặc
safetensors thành GGUF. GGUF có thể chứa:

- model configuration;
- vision weights;
- language-model weights;
- action-head weights;
- normalization statistics;
- embodiment metadata.

Ví dụ [`scripts/convert_smolvla_to_gguf.py`](../scripts/convert_smolvla_to_gguf.py)
đọc cả state/action statistics, kiểm tra shape và ghi vision tower, LM cùng
action expert vào bundle.

## 4. Hai giai đoạn inference chính

Technical report mô hình hóa VLA inference thành hai giai đoạn.

### 4.1 Multimodal prefix

Prefix được tạo từ:

```text
vision tokens + language tokens + state token(s)
```

Vision tower mã hóa một hoặc nhiều camera view. Token ngôn ngữ được lấy từ
embedding table. Robot state được chiếu qua một projection nhỏ. Sau đó backbone
tạo full hidden-state sequence thay vì chỉ logits.

Với architecture dùng cross-attention, K/V sinh từ prefix được tái sử dụng
trong các solver step. Cache này khác với autoregressive self-attention KV cache
của một LLM server thông thường.

### 4.2 Action head

Code hiện tại có ba họ action head.

#### Flow matching

Dùng bởi SmolVLA, pi0 và pi0.5. Action bắt đầu từ noise và được cập nhật qua
nhiều Euler step:

```text
a_0 = noise

for each solver step t:
    velocity = action_expert(a_t, prefix, t)
    a_(t+1) = a_t + dt * velocity
```

#### Diffusion transformer / DiT

Dùng bởi Evo-1, GR00T và VLA-JEPA. Action sequence đi qua một transformer nhỏ,
thường cross-attend vào context do vision-language backbone sinh ra.

#### Single-pass regression

BitVLA, OpenVLA-OFT và VLA-Adapter dự đoán các action position song song trong
một forward pass. Không có solver loop như flow-matching model.

## 5. Deep dive: SmolVLA

Đường chính nằm trong [`src/models/smolvla.cpp`](../src/models/smolvla.cpp).

Một prediction thực hiện:

1. Preprocess từng camera view.
2. Chạy SigLIP vision transformer.
3. Pixel-shuffle feature map và chiếu về LM hidden size.
4. Ghép image embeddings, language embeddings và state embedding.
5. Chạy VLM prefix và tạo K/V theo từng layer.
6. Tạo cross-attention K/V cho action expert.
7. Khởi tạo action noise.
8. Tích phân flow field qua `cfg.num_steps`.
9. Giải chuẩn hóa action bằng `action_mean` và `action_std`.

SmolVLA có hai execution mode:

- graph cache nhanh khi không yêu cầu timing chi tiết;
- tách prefill và denoise để đo latency từng phase.

Khi số camera view không đổi, cấu trúc graph được giữ lại và các input tensor
được cập nhật cho request tiếp theo. Đây là graph reuse, không phải reuse prefix
giữa hai observation khác nhau.

## 6. Deep dive: Evo-1

Implementation nằm tại [`src/models/evo1.cpp`](../src/models/evo1.cpp).

Luồng chính:

1. InternViT tạo image embeddings.
2. Client hoặc model tạo prompt chứa các vị trí image-context token.
3. Image embeddings được splice vào embedding sequence.
4. Qwen backbone tạo context hidden states.
5. State được chiếu thành một context token bổ sung.
6. Mỗi DiT layer tạo K/V từ context một lần.
7. Action chạy qua nhiều solver step và dùng lại các K/V đó.
8. Output được giải chuẩn hóa bằng `action_min/action_max`.

Evo-1 là ví dụ rõ nhất của ý tưởng trong report: phần context lớn chạy một lần,
trong khi action solver nhỏ hơn cross-attend vào context nhiều lần.

## 7. Deep dive: BitVLA

Implementation nằm tại [`src/models/bitvla.cpp`](../src/models/bitvla.cpp).

BitVLA khác hai họ trên:

1. BitSigLIP xử lý ảnh.
2. BitNet xử lý prefix và các action query token.
3. Runtime lấy hidden states tại action positions.
4. MLP action head dự đoán toàn bộ chunk trong một pass.
5. Output được giải chuẩn hóa bằng q01/q99.

CUDA build thêm kernel riêng từ [`src/kernels/bitvla/`](../src/kernels/bitvla).
Các kernel này dùng packed ternary weights và tensor-core paths thay vì chỉ dựa
vào generic ggml matmul.

## 8. Serving và control loop

### 8.1 C++ server

[`src/serving/server.cpp`](../src/serving/server.cpp) là ZeroMQ REP server đồng
bộ. Nó:

- load một model khi khởi động;
- nhận một request tại một thời điểm;
- kiểm tra số ảnh, token, state và noise;
- hỗ trợ JPEG, RGB U8 và RGB float trong `[0, 1]`;
- gọi `vla::predict()`;
- trả action chunk cùng phase timing.

Server hiện không có batching, multi-model scheduling hay authentication. Nếu
bind ra interface mạng, nó in cảnh báo vì bất kỳ máy nào truy cập được endpoint
đều có thể gửi inference request.

### 8.2 Python client

[`eval/client/vla_cpp_client.py`](../eval/client/vla_cpp_client.py) giữ phần phụ
thuộc architecture:

- tokenizer;
- prompt template;
- image transformation;
- state normalization;
- action postprocessing;
- action replay queue.

Client nhận cả chunk nhưng chỉ thực thi `n_action_steps` trước khi gửi observation
mới. Đây là tham số control quan trọng:

- nhỏ: replan thường xuyên, observation mới hơn;
- lớn: giảm số lần inference nhưng robot chạy open-loop lâu hơn;
- bằng toàn horizon: action throughput cao nhưng dễ drift khỏi scene đã quan sát.

## 9. Những kết luận chính của technical report

### 9.1 Prefix compute-bound, action expert thường memory-bound

Vision và LM prefix xử lý hàng trăm token cùng lúc nên weight reuse cao. Trên
GPU GDDR, phase này thường compute-bound.

Action expert xử lý ít token hơn nhưng lặp nhiều solver step. Arithmetic intensity
thấp hơn nên phase này thường memory-bound. Tỷ trọng action expert tăng theo số
solver step.

Hệ quả: tối ưu prefix và action head cần hai chiến lược khác nhau.

### 9.2 Quantization chủ yếu giải bài toán capacity

Report cho thấy packed ternary BitVLA giảm model từ khoảng 5.6 GiB xuống 1.34
GiB và giảm đáng kể load-time high-water mark. Tuy nhiên latency không tự giảm
nếu kernel vẫn thực hiện cùng lượng phép toán.

Tốc độ chỉ tăng rõ khi W2A8 matmul được chuyển từ CUDA-core `dp4a` sang IMMA
tensor cores. Report đo khoảng 4--4.6 lần nhanh hơn cho kernel tương ứng.

### 9.3 Precision error có thể trở thành lỗi hành vi

Một sai khác FP16/FP32 trong phép tính position index có thể chọn sai hàng của
positional-embedding table. Sai số prefix sau đó tích lũy qua action solver và
có thể làm action lệch khoảng `1.97` trong robot space.

Các phép toán rời rạc cần được kiểm tra parity đặc biệt kỹ:

- index calculation;
- rounding;
- bucketization;
- argmax;
- position lookup.

Phép toán liên tục như interpolation thường chịu reduced precision tốt hơn.

### 9.4 Latency thay đổi closed-loop behavior

Trong thí nghiệm ALOHA với GR00T N1.6, report ghi:

- vla.cpp: khoảng 470 ms mỗi chunk;
- PyTorch: khoảng 620 ms mỗi chunk;
- vla.cpp: 35/40 trial thành công;
- PyTorch: 16/40 trial thành công.

Weights và open-loop action gần tương đương. Khác biệt đến từ việc action của
runtime nhanh hơn được lập kế hoạch trên observation mới hơn khoảng 150 ms.

Do đó latency VLA không chỉ là benchmark throughput; nó trực tiếp ảnh hưởng
đến policy trong closed loop.

## 10. Những điểm report, docs và code chưa đồng nhất

### 10.1 `num_steps` không phải chunk length

Contract thực tế là:

```text
action output shape = [n_suffix, max_action_dim]
solver iterations   = num_steps
```

Một số comment trong `model.h` dùng `[num_steps, max_action_dim]` để mô tả output,
nhưng server trả `chunk_size = cfg.n_suffix`.

### 10.2 Normalization phụ thuộc architecture

Docs nói `predict()` trả normalized actions và caller luôn unnormalize. Code thực
tế không hoàn toàn như vậy:

- SmolVLA, pi0, Evo-1 và BitVLA có unnormalization trong C++;
- một số GR00T và VLA-JEPA postprocess bằng statistics ở Python client;
- server không thêm một bước unnormalize chung.

Không nên bỏ architecture preset và dùng một generic client cho mọi model.

### 10.3 Vision packaging đã thay đổi

`docs/ARCHITECTURE.md` vẫn mô tả SmolVLA/pi0 dùng một mmproj riêng. Converter
SmolVLA hiện tại có thể bake vision tower vào cùng GGUF. API và CLI vẫn giữ
`mmproj_path` để tương thích với cả bundle cũ lẫn mới.

### 10.4 Evo-1 hiện có known issue nghiêm trọng

[`docs/KNOWN_ISSUES.md`](KNOWN_ISSUES.md) ghi released Evo-1 GGUF đạt 0% trên
LIBERO, trái với 94.5% trong report.

Tài liệu đó cũng nói implementation dùng Gaussian `N(0,1)` noise, trong khi code
hiện tại tự sinh `uniform[-1,1)` nếu request không truyền noise. Commit
`2efef28` đã thay đổi hành vi này nhưng known-issues document chưa được cập nhật.

Vì vậy cần phân biệt:

- model load và tạo finite actions;
- numerical parity với reference;
- task success trong simulator/robot.

Smoke test chỉ xác nhận điều đầu tiên.

### 10.5 Public CI chưa thể hiện đầy đủ parity gate trong report

Report mô tả việc gate từng block với reference. Test public hiện tại chủ yếu có:

- vision helper unit test;
- converter key-remap test;
- `predict_check` dùng fixed input/noise nhưng cần GGUF thật và không chạy tự
  động trong CTest.

Do đó không nên hiểu rằng mọi checkpoint phát hành đều đang được CI kiểm tra
end-to-end task success.

## 11. Điểm mạnh và giới hạn

### Điểm mạnh

- Public API nhỏ, dễ nhúng vào ứng dụng C++.
- Một runtime hỗ trợ nhiều họ VLA.
- Server không cần PyTorch.
- CPU, CUDA và Metal dùng chung cấu trúc tổng thể.
- GGUF đóng gói config, weights và nhiều normalization statistics.
- Có validation ở network boundary.
- Có custom low-bit CUDA kernels cho BitVLA.
- Có eval scaffold cho LIBERO và SimplerEnv.

### Giới hạn

- Model implementations lớn và khá monolithic.
- Pre/postprocessing chưa hoàn toàn nằm trong C++.
- Output normalization contract chưa đồng nhất.
- Server không batching và chỉ xử lý request đồng bộ.
- Không có dynamic weight staging cho thiết bị ít RAM/VRAM.
- Low-bit acceleration riêng mới tập trung vào BitVLA.
- Tests tự động chưa bao phủ numerical parity và task success của mọi model.
- Một số tài liệu đang chậm hơn implementation.

## 12. Thứ tự đọc code đề xuất

1. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) để nắm vocabulary.
2. [`src/model.h`](../src/model.h) để hiểu public contract.
3. [`src/model.cpp`](../src/model.cpp) và [`src/arch.h`](../src/arch.h) để hiểu
   dispatch.
4. [`src/serving/vla-cli.cpp`](../src/serving/vla-cli.cpp) để xem inference path
   ngắn nhất.
5. [`src/models/smolvla.cpp`](../src/models/smolvla.cpp) để hiểu flow matching và
   cross-attention cache.
6. [`src/models/evo1.cpp`](../src/models/evo1.cpp) để hiểu DiT.
7. [`src/models/bitvla.cpp`](../src/models/bitvla.cpp) để hiểu single-pass action
   head và low-bit path.
8. [`src/serving/server.cpp`](../src/serving/server.cpp) và
   [`eval/client/vla_cpp_client.py`](../eval/client/vla_cpp_client.py) để hiểu
   serving/control boundary.
9. Converter của model tương ứng để ánh xạ tensor từ checkpoint gốc sang GGUF.
10. Cuối cùng đọc GR00T, vì preprocessing, embodiment và postprocessing phức tạp
    hơn các model còn lại.

## 13. Kết luận

Ý tưởng trung tâm của `vla.cpp` là biểu diễn VLA inference thành:

```text
multimodal prefill + cached context + action generation
```

Phần khó nhất không đơn thuần là viết transformer bằng C++. Runtime còn phải giữ
đúng image preprocessing, attention mask, position encoding, cache lifecycle,
noise schedule, normalization và action-chunk semantics. Một implementation có
thể compile, chạy trên GPU và tạo output hữu hạn nhưng vẫn điều khiển robot sai
nếu chỉ một trong các chi tiết này lệch khỏi reference.
