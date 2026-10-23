#!/usr/bin/env python3
"""Apply SGLang PR #36708's GLM mHC capture to the accepted SM121 model file."""

from __future__ import annotations

import ast
import os
from pathlib import Path


PATH = Path(
    os.environ.get(
        "GLM53_DFLASH_GLM_FILE",
        "/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py",
    )
)
ROOT = PATH.parents[2]
MARKER = "def set_dflash_layers_to_capture"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


text = PATH.read_text(encoding="utf-8")
if MARKER not in text:
    text = replace_once(
        text,
        "from sglang.kernels.ops.layernorm.mhc import hc_post as _hc_post_fn\n",
        "from sglang.kernels.ops.layernorm.mhc import hc_contract\n"
        "from sglang.kernels.ops.layernorm.mhc import hc_post as _hc_post_fn\n",
        "hc_contract import",
    )
    text = replace_once(
        text,
        "class Glm5NextModel(nn.Module):\n"
        "    fall_back_to_pt_during_load = False\n\n"
        "    def __init__(\n"
        "        self,\n"
        "        config: Glm5NextTextConfig,\n"
        "        quant_config: Optional[QuantizationConfig] = None,\n"
        "        prefix: str = \"\",\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "        self.padding_id = config.pad_token_id\n",
        "class Glm5NextModel(nn.Module):\n"
        "    fall_back_to_pt_during_load = False\n\n"
        "    def __init__(\n"
        "        self,\n"
        "        config: Glm5NextTextConfig,\n"
        "        quant_config: Optional[QuantizationConfig] = None,\n"
        "        prefix: str = \"\",\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "        self.padding_id = config.pad_token_id\n",
        "model config",
    )
    text = replace_once(
        text,
        "        self.layers_to_capture = []\n",
        "        self.layers_to_capture = []\n"
        "        self.dflash_capture = False\n",
        "dflash capture state",
    )
    text = replace_once(
        text,
        "    def get_input_embeddings(self) -> torch.Tensor:\n"
        "        return self.embed_tokens\n\n"
        "    def forward(\n"
        "        self,\n"
        "        input_ids: torch.Tensor,\n",
        "    def get_input_embeddings(self) -> torch.Tensor:\n"
        "        return self.embed_tokens\n\n"
        "    def _prepare_aux_hidden_state(\n"
        "        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]\n"
        "    ) -> torch.Tensor:\n"
        "        aux_hidden_state = (\n"
        "            hidden_states if residual is None else hidden_states + residual\n"
        "        )\n"
        "        if self.dflash_capture and self.config.mhc:\n"
        "            aux_hidden_state = hc_contract(aux_hidden_state, self.config.hc_mult)\n"
        "        return aux_hidden_state\n\n"
        "    def forward(\n"
        "        self,\n"
        "        input_ids: torch.Tensor,\n",
        "aux hidden preparation",
    )
    text = replace_once(
        text,
        "        if forward_batch.can_run_tbo:\n",
        "        if forward_batch.can_run_tbo and not self.dflash_capture:\n",
        "TBO gate",
    )
    text = replace_once(
        text,
        "                if i in self.layers_to_capture:\n"
        "                    if self.enable_a2a_moe and i > self.first_k_dense_replace:\n"
        "                        aux_hidden_state = get_parallel().attn_tp_group.all_gather(\n"
        "                            hidden_states + residual, dim=0\n"
        "                        )\n"
        "                        aux_hidden_states.append(aux_hidden_state)\n"
        "                    else:\n"
        "                        aux_hidden_states.append(hidden_states + residual)\n",
        "                if i in self.layers_to_capture:\n"
        "                    aux_hidden_state = self._prepare_aux_hidden_state(\n"
        "                        hidden_states, residual\n"
        "                    )\n"
        "                    if self.enable_a2a_moe and i > self.first_k_dense_replace:\n"
        "                        aux_hidden_state = get_parallel().attn_tp_group.all_gather(\n"
        "                            aux_hidden_state, dim=0\n"
        "                        )\n"
        "                    aux_hidden_states.append(aux_hidden_state)\n",
        "capture loop",
    )
    text = replace_once(
        text,
        "    def prepare_context_parallel_metadata_for_dcp(\n",
        "    def set_dflash_layers_to_capture(self, layer_ids: List[int]):\n"
        "        if not self.pp_group.is_last_rank:\n"
        "            return\n\n"
        "        if layer_ids is None:\n"
        "            raise ValueError(\n"
        "                \"DFLASH requires explicit layer_ids for aux hidden capture.\"\n"
        "            )\n\n"
        "        self.capture_aux_hidden_states = True\n"
        "        self.model.dflash_capture = True\n"
        "        self.model.layers_to_capture = [val + 1 for val in layer_ids]\n\n"
        "    def prepare_context_parallel_metadata_for_dcp(\n",
        "target hook",
    )

ast.parse(text, filename=str(PATH))
PATH.write_text(text, encoding="utf-8")

server_args = (ROOT / "srt/server_args.py").read_text(encoding="utf-8")
required = {
    "generic DFLASH server support": "DFLASH" in server_args,
    "FA4 draft backend": "fa4" in server_args.lower(),
    "DFlash model": (ROOT / "srt/models/dflash.py").is_file(),
    "DFlash worker": (ROOT / "srt/speculative/dflash_worker_v2.py").is_file(),
    "GLM target hook": MARKER in text,
    "mHC contraction": "hc_contract(aux_hidden_state" in text,
}
failed = [name for name, passed in required.items() if not passed]
if failed:
    raise RuntimeError("DFlash2 build prerequisites missing: " + ", ".join(failed))
print("DFlash2 GLM/SM121 source gate: PASS")
