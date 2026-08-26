#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_checkpoint import validate_snapshot  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ValidateCheckpointTest(unittest.TestCase):
    def make_snapshot(self, root: Path) -> tuple[Path, Path]:
        snapshot = root / "snapshot"
        snapshot.mkdir()
        static_content = {
            ".gitattributes": b"lfs\n",
            "LICENSE": b"license\n",
            "README.md": b"readme\n",
            "chat_template.jinja": b"corrected-template\n",
            "generation_config.json": b"{}\n",
            "processor_config.json": b"{}\n",
            "tokenizer.json": b"tokenizer\n",
            "tokenizer_config.json": b"{}\n",
        }
        for name, data in static_content.items():
            (snapshot / name).write_bytes(data)

        config = {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "text_config": {
                "dtype": "bfloat16",
                "max_position_embeddings": 1048576,
                "num_nextn_predict_layers": 1,
                "n_routed_experts": 288,
                "num_experts_per_tok": 8,
            },
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "producer": {"name": "modelopt", "version": "0.45.0"},
                "config_groups": {
                    "group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}
                },
                "ignore": ["lm_head", "model.language_model.embed_tokens", "model.visual.*"],
            },
        }
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")

        shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for shard_name in shard_names:
            with (snapshot / shard_name).open("wb") as handle:
                handle.seek(1024 * 1024)
                handle.write(b"x")
        weight_map = {
            "model.layers.0.mlp.experts.0.weight": shard_names[0],
            "model.layers.0.mlp.experts.0.weight_scale": shard_names[0],
            "model.layers.0.mlp.experts.0.weight_scale_2": shard_names[1],
            "model.embed_tokens.weight": shard_names[1],
        }
        index = {"metadata": {"total_size": 1234}, "weight_map": weight_map}
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )

        manifest = {
            "model": {
                "id": "test/model",
                "revision": "a" * 40,
                "architecture": "Glm5NextForConditionalGeneration",
                "model_type": "glm5_next",
                "max_position_embeddings": 1048576,
            },
            "quantization": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "producer_name": "modelopt",
                "producer_version": "0.45.0",
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
                "ignore_count": 3,
            },
            "index": {
                "tensor_payload_bytes": 1234,
                "tensor_entries": 4,
                "shards": 2,
                "expert_weights": 1,
                "expert_weight_scales": 1,
                "expert_weight_scales_2": 1,
                "non_expert_entries": 1,
            },
            "files": {name: digest(data) for name, data in static_content.items() if name in {
                "chat_template.jinja",
                "generation_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            }},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return snapshot, manifest_path

    def test_valid_snapshot_and_fail_closed_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, manifest = self.make_snapshot(Path(temporary))
            self.assertEqual(validate_snapshot(snapshot, manifest), [])

            (snapshot / "chat_template.jinja").write_text("stale", encoding="utf-8")
            failures = validate_snapshot(snapshot, manifest)
            self.assertTrue(any("SHA-256 mismatch" in failure for failure in failures))

            (snapshot / "remote_code.py").write_text("pass\n", encoding="utf-8")
            failures = validate_snapshot(snapshot, manifest)
            self.assertTrue(any("unexpected files" in failure for failure in failures))
            self.assertTrue(any("executable/code" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
