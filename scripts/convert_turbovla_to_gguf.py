# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Convert a supported TurboVLA .pth checkpoint to a combined vla.cpp GGUF."""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch


ARCH = "turbovla"


_PREFIXES = (
    ("text_encoder.bert.embeddings.", "t.e."),
    ("text_encoder.bert.encoder.layer.", "t.b."),
    ("text_encoder.bert.pooler.dense.", "t.pool."),
    ("text_encoder.text_projection.", "t.p."),
    ("vision_encoder.backbone.embeddings.", "v.e."),
    ("vision_encoder.backbone.layer.", "v.b."),
    ("vision_encoder.backbone.norm.", "v.n."),
    ("vision_projection.", "vp."),
    ("vision_language_interaction.text_layers.", "i.t."),
    ("vision_language_interaction.fusion_layers.", "i.f."),
    ("action_head.state_projection.", "a.s."),
    ("action_head.decoder.action_queries.", "a.q."),
    ("action_head.decoder.decoder.layers.", "a.d."),
    ("action_head.decoder.action_projection.layers.", "a.p."),
)


def _gguf_name(name: str) -> str:
    for prefix, short in _PREFIXES:
        if name.startswith(prefix):
            name = short + name[len(prefix):]
            break
    if name.endswith(".weight"):
        name = name[:-7] + ".w"
    elif name.endswith(".bias"):
        name = name[:-5] + ".b"
    if len(name.encode("utf-8")) >= 64:
        raise ValueError(f"GGUF tensor name remains too long: {name!r}")
    return name


def _bf16_u16(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.to(torch.bfloat16).contiguous().view(torch.int16)
            .cpu().numpy().view(np.uint16))


def _add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor,
                outtype: str) -> None:
    tensor = tensor.detach().cpu().contiguous()
    if outtype == "f32":
        writer.add_tensor(name, tensor.float().numpy())
    else:
        writer.add_tensor(
            name, _bf16_u16(tensor), raw_shape=list(tensor.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )


def _positive_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        value = default
    value = int(value)
    if value < 1:
        raise ValueError(f"model_config.{key} must be positive, got {value}")
    return value


def _suite_name(blob: dict) -> str:
    if blob.get("suite"):
        return str(blob["suite"])
    data_root = str(blob.get("args", {}).get("data_root", "")).lower()
    for suite in ("aloha", "libero", "robotwin"):
        if suite in data_root:
            return suite
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="TurboVLA PyTorch .pth checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outtype", choices=("f32", "bf16"), default="bf16")
    parser.add_argument(
        "--stats-out", type=Path, default=None,
        help="Normalizer JSON sidecar (default: <out stem>.stats.json)",
    )
    args = parser.parse_args()

    # mmap avoids eagerly faulting optimizer tensors into RAM. Fine-tuning
    # checkpoints can be much larger than the model_state_dict we export.
    try:
        blob = torch.load(
            args.ckpt, map_location="cpu", weights_only=False, mmap=True,
        )
    except TypeError:  # torch < 2.1
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict")
    config = blob.get("model_config", {})
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{args.ckpt} has no non-empty model_state_dict")
    if config.get("name") != "TurboVLA":
        raise ValueError(f"expected model_config.name=TurboVLA, got {config.get('name')!r}")

    text_config = config.get("text", {})
    vision_config = config.get("vision", {})
    interaction_config = config.get("interaction", {})
    action_config = config.get("action", {})
    image_size = _positive_int(vision_config, "image_size", 256)
    image_views = _positive_int(vision_config, "num_views", 2)
    text_max_length = _positive_int(text_config, "max_length", 256)
    padding_length = text_config.get("padding_length")
    text_length = 0 if padding_length is None else int(padding_length)
    state_dim = _positive_int(action_config, "state_dim", 8)
    action_dim = _positive_int(action_config, "action_dim", 7)
    action_horizon = _positive_int(action_config, "horizon", 12)

    # The C++ implementation currently targets the published ViT-B/256,
    # 256-wide interaction and 12x7 action-head family. State width and text
    # padding are checkpoint-specific and are resolved dynamically below.
    supported = {
        "vision.image_size": (image_size, 256),
        "vision.num_views": (image_views, 2),
        "interaction.hidden_dim": (
            _positive_int(interaction_config, "hidden_dim", 256), 256,
        ),
        "interaction.num_layers": (
            _positive_int(interaction_config, "num_layers", 6), 6,
        ),
        "interaction.enhancer_inner_dim": (
            _positive_int(interaction_config, "enhancer_inner_dim", 1024), 1024,
        ),
        "action.action_dim": (action_dim, 7),
        "action.horizon": (action_horizon, 12),
        "action.num_state_tokens": (
            _positive_int(action_config, "num_state_tokens", 2), 2,
        ),
        "action.num_layers": (
            _positive_int(action_config, "num_layers", 3), 3,
        ),
        "action.state_hidden_dim": (
            _positive_int(action_config, "state_hidden_dim", 256), 256,
        ),
        "action.mlp_hidden_dim": (
            _positive_int(action_config, "mlp_hidden_dim", 512), 512,
        ),
    }
    mismatches = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in supported.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "checkpoint uses a TurboVLA variant not yet supported by vla.cpp: "
            + ", ".join(mismatches)
        )
    if text_length < 0 or text_length > text_max_length:
        raise ValueError(
            f"invalid text padding_length={text_length} for max_length={text_max_length}"
        )

    required = (
        "text_encoder.bert.embeddings.word_embeddings.weight",
        "vision_encoder.backbone.embeddings.patch_embeddings.weight",
        "vision_language_interaction.fusion_layers.0.attn.v_proj.weight",
        "action_head.decoder.action_projection.layers.2.weight",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"checkpoint is missing required tensors: {missing}")
    expected_shapes = {
        "text_encoder.bert.embeddings.word_embeddings.weight": (30522, 768),
        "vision_encoder.backbone.embeddings.patch_embeddings.weight":
            (768, 3, 16, 16),
        "action_head.state_projection.net.0.weight": (state_dim,),
        "action_head.state_projection.net.1.weight": (256, state_dim),
        "action_head.decoder.action_queries.weight": (action_horizon, 256),
        "action_head.decoder.action_projection.layers.2.weight":
            (action_dim, 512),
    }
    bad_shapes = [
        f"{name}={tuple(state[name].shape)} (expected {expected})"
        for name, expected in expected_shapes.items()
        if name not in state or tuple(state[name].shape) != expected
    ]
    if bad_shapes:
        raise ValueError("checkpoint tensor shape mismatch: " + ", ".join(bad_shapes))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), ARCH)
    kv = lambda name: f"{ARCH}.{name}"
    writer.add_string(kv("architecture"), ARCH)
    writer.add_string(kv("source_checkpoint"), args.ckpt.name)
    suite = _suite_name(blob)
    writer.add_string(kv("suite"), suite)
    writer.add_string(kv("model_config_json"), json.dumps(config, separators=(",", ":")))
    writer.add_uint32(kv("image_size"), image_size)
    writer.add_uint32(kv("image_views"), image_views)
    # Zero means dynamic/longest: for batch-size-one C++ inference the runtime
    # token count is also the interaction sequence length.
    writer.add_uint32(kv("text_length"), text_length)
    writer.add_uint32(kv("text_max_length"), text_max_length)
    writer.add_uint32(kv("state_dim"), state_dim)
    writer.add_uint32(kv("action_dim"), action_dim)
    writer.add_uint32(kv("action_horizon"), action_horizon)
    writer.add_float32(kv("layer_norm_eps"), 1e-5)
    stats = blob.get("stats", {})
    stats_sidecar = {"suite": suite}
    for name in ("state_mean", "state_std", "action_min", "action_max"):
        values = stats.get(name)
        if values is not None:
            expected = state_dim if name.startswith("state_") else action_dim
            if len(values) != expected:
                raise ValueError(
                    f"stats.{name} has {len(values)} values, expected {expected}"
                )
            serialized = [float(value) for value in values]
            writer.add_array(kv(name), serialized)
            stats_sidecar[name] = serialized
    if stats.get("stats_key"):
        writer.add_string(kv("stats_key"), str(stats["stats_key"]))
        stats_sidecar["stats_key"] = str(stats["stats_key"])

    # Tensor names deliberately match the PyTorch state_dict. This keeps the
    # conversion auditable and makes layer-by-layer parity debugging simple.
    used_names: set[str] = set()
    for index, (name, tensor) in enumerate(state.items(), start=1):
        if not torch.is_tensor(tensor):
            continue
        output_name = _gguf_name(name)
        if output_name in used_names:
            raise ValueError(f"GGUF tensor name collision at {output_name!r}")
        used_names.add(output_name)
        _add_tensor(writer, output_name, tensor, args.outtype)
        if index % 100 == 0:
            print(f"added {index}/{len(state)} tensors", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({args.outtype}, {len(state)} tensors)")
    if len(stats_sidecar) > 1:
        stats_out = args.stats_out or args.out.with_suffix(".stats.json")
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(
            json.dumps(stats_sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {stats_out} (normalizer sidecar)")


if __name__ == "__main__":
    main()
