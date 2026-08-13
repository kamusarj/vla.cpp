# GR00T N1.7 Checkpoint2: PyTorch vs GGUF Open-Loop Evaluation

This report consolidates the recorded-dataset open-loop evaluation results for
the original PyTorch checkpoint and its converted vla.cpp GGUF artifact.

These are two runtime representations of the same trained checkpoint, not two
independently trained models:

- **PyTorch:** Hugging Face checkpoint `Luke99662244/checkpoint2`, revision
  `7c4bdfe13690784aeb59a29f0142c43cd74072c4`.
- **GGUF:** `checkpoint2-f32.gguf`, converted from the checkpoint above for
  inference with vla.cpp. SHA-256:
  `c37d3563df050d8a2307740e29a078968e17d4313b8490ed29593eb5ee3999be`.

## Evaluation protocol

Both evaluations use the protocol from
[Step 4: Open Loop Evaluation in NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md#step-4-open-loop-evaluation).

| Setting | Value |
| --- | --- |
| Dataset | `datasets/ur10e-cup-eval-v2` |
| Trajectory | `0` — `pick up the cup` |
| Frames evaluated | 814 |
| Inference requests | 51, at frames 0, 16, ..., 800 |
| Execution horizon | 16 actions per request |
| Model action horizon | 40 |
| Denoising steps | 4 |
| State/action dimensions | 7 active dimensions: 6 arm joints + 1 gripper |
| Camera views | `side`, `wrist` |
| Processed image size | 256 × 352 per view |
| Embodiment | `new_embodiment`, projector ID 10 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16,311 MiB |
| Isaac-GR00T revision | `376ba890cff8c9de64d71d982772a9c36185fdd7` |

The errors below are calculated after decoding normalized model output back to
the original action units. Predictions are generated once every 16 frames, the
first 16 actions of each predicted chunk are concatenated, and the resulting
814 actions are compared with the recorded ground truth.

## Results

| Metric | PyTorch checkpoint | vla.cpp GGUF | Observed difference |
| --- | ---: | ---: | ---: |
| Unnormalized action MSE | 0.0007250374 | 0.0009261190 | +0.0002010817 (+27.73%) |
| Unnormalized action MAE | 0.0101807602 | 0.0113530281 | +0.0011722679 (+11.51%) |
| Frames | 814 | 814 | Same |
| Inference requests | 51 | 51 | Same |
| Non-finite outputs | None observed | None observed | Same |
| Runtime failure / CUDA OOM | None | None | Same |

The GGUF result is slightly higher in both error metrics in these two recorded
runs. This difference is descriptive, but it is **not a controlled estimate of
conversion error**: the original PyTorch evaluation sampled stochastic initial
noise without recording a fixed seed, whereas the GGUF evaluation used BF16
noise generated with seed `20260813`.

## Trajectory visualizations

The orange curve is the ground-truth action, the green curve is the predicted
action, and the red points mark inference frames spaced 16 frames apart.

### Original PyTorch checkpoint

![PyTorch ground-truth and predicted actions](../openloop-output-checkpoint2/traj_0.jpeg)

### Converted vla.cpp GGUF

![GGUF ground-truth and predicted actions](../gguf-openloop-output-checkpoint2/traj_0.jpeg)

Both plots show that the predicted action trends follow the recorded trajectory
across all six arm joints and the gripper. This verifies recorded-trajectory
tracking; it does not measure closed-loop success on the physical robot.

## GGUF inference performance

The persistent `vla-openloop` process loads the GGUF once and evaluates all 51
requests on the RTX 5060 Ti.

| Timing | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Total per request | 69.21 ms | 66.56 ms | 66.43 ms | 178.79 ms |
| Vision encoder | 38.65 ms | 36.59 ms | 36.56 ms | 133.70 ms |
| Action inference | 28.82 ms | 28.34 ms | 28.31 ms | 41.37 ms |

The first request includes graph and kernel warm-up. After warm-up, requests
remain close to 66.5 ms. The GGUF is stored as F32 on disk and loaded as BF16
resident weights, using approximately 4.13 GiB of GPU memory for model weights.
Equivalent per-request timing was not captured by the earlier PyTorch run, so a
PyTorch/GGUF speedup is not reported here.

## Controlled conversion validation

A separate step-0 parity test used identical camera tensors, language tokens,
normalized state, and BF16 noise for both runtimes. It produced:

| Fixed-input parity metric | Result |
| --- | ---: |
| Cosine similarity, all 132 output dimensions | 0.99999464 |
| MAE, all 132 output dimensions | 0.00229214 |
| Cosine similarity, 7 active action dimensions | 0.99999487 |
| MAE, 7 active action dimensions | 0.00193708 |
| Maximum absolute error, 7 active action dimensions | 0.007459 |

The first GGUF request in the full open-loop batch is byte-identical to the
previous vla.cpp CUDA parity output. These controlled results indicate that the
converter and GGUF runtime reproduce the PyTorch model closely. A strict
full-trajectory conversion comparison would require rerunning PyTorch with the
exact 51 noise tensors saved by the GGUF fixture.

## Conclusion

- Both runtimes complete the full 814-frame evaluation successfully and track
  the trajectory with low unnormalized action error.
- The observed GGUF run has 27.73% higher MSE and 11.51% higher MAE than the
  earlier PyTorch run, but the runs used different stochastic noise.
- Fixed-input parity is very high (`0.99999487` cosine similarity over active
  actions), supporting that checkpoint conversion is correct.
- This one recorded trajectory validates the inference pipeline; it is not a
  substitute for held-out multi-episode or closed-loop robot evaluation.

## Artifacts

PyTorch evaluation:

- [Evaluator log](../openloop-output-checkpoint2/eval.log)
- [Run information](../openloop-output-checkpoint2/run-info.txt)
- [Trajectory visualization](../openloop-output-checkpoint2/traj_0.jpeg)

GGUF evaluation:

- [Metrics](../gguf-openloop-output-checkpoint2/metrics.json)
- [vla.cpp inference log](../gguf-openloop-output-checkpoint2/inference.log)
- [Per-request timings](../gguf-openloop-output-checkpoint2/timings.csv)
- [Decoded predictions](../gguf-openloop-output-checkpoint2/pred_actions.f32)
- [Trajectory visualization](../gguf-openloop-output-checkpoint2/traj_0.jpeg)

Implementation:

- [`vla-openloop` persistent batch runner](../src/serving/vla-openloop.cpp)
- [GGUF open-loop preparation and scoring script](../scripts/eval_gr00t_n1_7_gguf_openloop.py)
- [Detailed checkpoint and conversion report](GR00T_N1D7_UR10E_OPENLOOP_EVALUATION.md)
