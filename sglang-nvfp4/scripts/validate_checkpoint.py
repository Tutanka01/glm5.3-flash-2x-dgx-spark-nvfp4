#!/usr/bin/env python3
"""Fail-closed validation for the pinned GLM-5.3-Flash NVFP4 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gib(value: int | float) -> float:
    return float(value) / (1024**3)


def validate_snapshot(model_dir: Path, manifest_path: Path) -> list[str]:
    model_dir = model_dir.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(model_dir.is_dir(), f"snapshot directory does not exist: {model_dir}")
    if failures:
        return failures

    model_spec = manifest["model"]
    quant_spec = manifest["quantization"]
    index_spec = manifest["index"]
    file_hashes: dict[str, str] = manifest["files"]

    expected_static = {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    expected_shards = {
        f"model-{number:05d}-of-{index_spec['shards']:05d}.safetensors"
        for number in range(1, int(index_spec["shards"]) + 1)
    }
    expected_names = expected_static | expected_shards
    actual_names = {entry.name for entry in model_dir.iterdir()}
    missing_names = sorted(expected_names - actual_names)
    unexpected_names = sorted(actual_names - expected_names)
    require(not missing_names, f"missing snapshot files: {', '.join(missing_names[:8])}")
    require(
        not unexpected_names,
        f"unexpected files in pinned snapshot: {', '.join(unexpected_names[:8])}",
    )

    dangerous_suffixes = {".py", ".sh", ".so", ".dll", ".dylib", ".exe"}
    dangerous = sorted(
        entry.name for entry in model_dir.iterdir() if entry.suffix.lower() in dangerous_suffixes
    )
    require(not dangerous, f"executable/code files are not expected: {', '.join(dangerous)}")

    for name in expected_names & actual_names:
        path = model_dir / name
        require(path.exists(), f"broken snapshot link: {name}")
        require(path.is_file(), f"expected a regular file: {name}")

    config_path = model_dir / "config.json"
    if config_path.exists():
        config = _load_json(config_path)
        require(
            config.get("architectures") == [model_spec["architecture"]],
            f"unexpected architecture: {config.get('architectures')!r}",
        )
        require(
            config.get("model_type") == model_spec["model_type"],
            f"unexpected model_type: {config.get('model_type')!r}",
        )

        text_config = config.get("text_config", {})
        require(text_config.get("dtype") == "bfloat16", "text dtype must remain bfloat16")
        require(
            text_config.get("max_position_embeddings")
            == model_spec["max_position_embeddings"],
            "unexpected max_position_embeddings",
        )
        require(text_config.get("num_nextn_predict_layers") == 1, "MTP head is missing")
        require(text_config.get("n_routed_experts") == 288, "unexpected routed expert count")
        require(text_config.get("num_experts_per_tok") == 8, "unexpected experts-per-token")

        quant = config.get("quantization_config", {})
        require(quant.get("quant_method") == quant_spec["quant_method"], "wrong quant_method")
        require(quant.get("quant_algo") == quant_spec["quant_algo"], "wrong quant_algo")
        producer = quant.get("producer", {})
        require(producer.get("name") == quant_spec["producer_name"], "wrong quant producer")
        require(
            producer.get("version") == quant_spec["producer_version"],
            "wrong quant producer version",
        )
        weights = quant.get("config_groups", {}).get("group_0", {}).get("weights", {})
        require(weights.get("num_bits") == quant_spec["num_bits"], "wrong quant bit width")
        require(weights.get("type") == quant_spec["type"], "wrong quant number type")
        require(weights.get("group_size") == quant_spec["group_size"], "wrong group size")
        ignore = quant.get("ignore", [])
        require(isinstance(ignore, list), "quantization ignore must be a list")
        if isinstance(ignore, list):
            require(len(ignore) == quant_spec["ignore_count"], "wrong quantization ignore count")
            require(len(ignore) == len(set(ignore)), "duplicate quantization ignore entries")
            for protected in ("lm_head", "model.language_model.embed_tokens", "model.visual.*"):
                require(protected in ignore, f"protected BF16 path missing from ignore: {protected}")
            for fused_name in quant_spec.get("required_ignore", []):
                require(
                    fused_name in ignore,
                    f"required fused-module ignore path is missing: {fused_name}",
                )

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
        weight_map = index.get("weight_map", {})
        require(isinstance(weight_map, dict), "weight_map must be a JSON object")
        if isinstance(weight_map, dict):
            require(len(weight_map) == index_spec["tensor_entries"], "wrong tensor entry count")
            mapped_shards = set(weight_map.values())
            require(mapped_shards == expected_shards, "index shard set is not exactly 1..120")

            expert_keys = [key for key in weight_map if ".mlp.experts." in key]
            expert_weights = sum(key.endswith(".weight") for key in expert_keys)
            expert_scales = sum(key.endswith(".weight_scale") for key in expert_keys)
            expert_scales_2 = sum(key.endswith(".weight_scale_2") for key in expert_keys)
            non_expert = len(weight_map) - len(expert_keys)
            require(expert_weights == index_spec["expert_weights"], "wrong expert weight count")
            require(
                expert_scales == index_spec["expert_weight_scales"],
                "wrong expert weight_scale count",
            )
            require(
                expert_scales_2 == index_spec["expert_weight_scales_2"],
                "wrong expert weight_scale_2 count",
            )
            require(non_expert == index_spec["non_expert_entries"], "wrong non-expert count")

        payload_bytes = index.get("metadata", {}).get("total_size")
        require(
            payload_bytes == index_spec["tensor_payload_bytes"],
            f"unexpected tensor payload size: {payload_bytes!r}",
        )

    for filename, expected_hash in file_hashes.items():
        path = model_dir / filename
        if path.exists():
            actual_hash = _sha256(path)
            require(
                actual_hash == expected_hash,
                f"SHA-256 mismatch for {filename}: {actual_hash}",
            )

    shard_disk_bytes = 0
    for shard_name in sorted(expected_shards & actual_names):
        shard = model_dir / shard_name
        try:
            shard_size = shard.stat().st_size
        except OSError as exc:
            failures.append(f"cannot stat {shard_name}: {exc}")
            continue
        require(shard_size > 1024 * 1024, f"suspiciously small shard: {shard_name}")
        shard_disk_bytes += shard_size

    if not failures:
        payload_bytes = int(index_spec["tensor_payload_bytes"])
        print(f"Validated checkpoint: {model_spec['id']}@{model_spec['revision']}")
        print(f"  snapshot: {model_dir}")
        print(f"  architecture: {model_spec['model_type']} ({model_spec['architecture']})")
        print(
            "  quantization: "
            f"{quant_spec['quant_algo']} weight-only, group={quant_spec['group_size']}, "
            f"producer={quant_spec['producer_name']} {quant_spec['producer_version']}"
        )
        print(
            f"  expert tensors: {index_spec['expert_weights']:,} weights + two scale tensors each"
        )
        print(
            f"  shards: {index_spec['shards']} "
            f"({shard_disk_bytes:,} on-disk bytes visible in this snapshot)"
        )
        print(
            f"  indexed tensor payload: {_gib(payload_bytes):.2f} GiB; "
            f"ideal TP=2 share: {_gib(payload_bytes) / 2:.2f} GiB/node"
        )
        print("  config/index/tokenizer/template/generation/processor hashes: audited match")
        print("  unexpected executable files: none")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    try:
        failures = validate_snapshot(args.model_dir, args.manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"checkpoint validation error: {exc}", file=sys.stderr)
        return 2

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
