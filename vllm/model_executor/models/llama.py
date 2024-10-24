# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip, AixQkvWeightHelper

from .interfaces import SupportsLoRA
from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers

# ==========================================================================================
# Yocto : support ladder net

# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = torch.nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         return output * self.weight

def dropout_add(x, residual, prob, training):
    # type: (torch.Tensor, torch.Tensor, float, bool) -> torch.Tensor
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out

# @torch.jit.script
def dropout_add_fused_inference(x: torch.Tensor,
                                residual: torch.Tensor,
                                prob: float) -> torch.Tensor:
    return dropout_add(x, residual, prob, False)

class ParallelGatedLinearUnit(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ParallelGatedLinearUnit, self).__init__()

        feed_forward_dim = int(int(out_dim/(3*4)) * 4)

        self.dense_1 = ColumnParallelLinear(
            in_dim,
            feed_forward_dim,
            gather_output=False,
            skip_bias_add=True,
            bias=False,
        )
        self.dense_2 = ColumnParallelLinear(
            in_dim,
            feed_forward_dim,
            gather_output=False,
            skip_bias_add=True,
            bias=False,
        )
        self.dense_3 = RowParallelLinear(
            feed_forward_dim,
            out_dim,
            input_is_parallel=True,
            skip_bias_add=True,
            bias=False,
        )
        self.activation = torch.nn.SiLU()
    
    def forward(self, hidden_state):
        part_one, _ = self.dense_1(hidden_state)
        part_one = self.activation(part_one)
        part_two, _ = self.dense_2(hidden_state)

        hidden_state, _ = self.dense_3(torch.multiply(part_one, part_two))
        return hidden_state

class LlamaLadderLayer(nn.Module):
    
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size,
                                  eps=config.rms_norm_eps)
        self.feed_forward = ParallelGatedLinearUnit(
            in_dim=4096, out_dim=4096
        )
        self.merge_gates = nn.Parameter(torch.ones(1))
    
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states).to(hidden_states.dtype)
        
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = dropout_add_fused_inference(hidden_states, residual=residual, prob=0)
        
        return hidden_states

# ==========================================================================================


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # debug_list.append({"q":q.clone().detach()})
        # debug_list.append({"k":k.clone().detach()})
        # debug_list.append({"v":v.clone().detach()})
        q, k = self.rotary_emb(positions, q, k)
        # debug_list.append({"rotary_emb_q":q.clone().detach()})
        # debug_list.append({"rotary_emb_k":k.clone().detach()})
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        # debug_list.append({"attn_output":attn_output.clone().detach()})
        output, _ = self.o_proj(attn_output)
        # debug_list.append({"o_proj_output":output.clone().detach()})
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # debug_list = []
        # debug_list.append({"layer_norm_input" : hidden_states.clone().detach()})
        # debug_list.append({"residual is None" : residual is None})
        # debug_list.append({"layer_norm_weight" : self.input_layernorm.weight})
        # debug_list.append({"layer_norm_eps" : self.input_layernorm.variance_epsilon})
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        # debug_list.append({"layer_norm_output" : hidden_states.clone().detach()})
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # for item in attn_debug_list:
        #     debug_list.append(item)
        # debug_list.append({"attention_output" : hidden_states.clone().detach()})

        
        # Fully Connected
        # debug_list.append({"post_attention_norm_input" : hidden_states.clone().detach()})
        # debug_list.append({"post_attention_norm_input_res" : residual.clone().detach()})
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        # debug_list.append({"post_attention_norm_output" : hidden_states.clone().detach()})
        hidden_states = self.mlp(hidden_states)
        # debug_list.append({"mlp_output" : hidden_states.clone().detach()})
        # return hidden_states, residual, debug_list
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
        with_ladder: Optional[bool] = False,
        sub_layers_ids: Optional[List[int]] = [],
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
# ==========================================================================================
        # Yocto : support ladder net
        # TODO(Yocto) : remove this hack
        self.with_ladder = with_ladder
        if self.with_ladder:
            self.sub_layers_ids = sub_layers_ids
            self.input_scaler = ParallelGatedLinearUnit(in_dim=4096, out_dim=4096)
            self.output_layer = ParallelGatedLinearUnit(in_dim=4096, out_dim=4096)
            self.ladder_layers = torch.nn.ModuleList([LlamaLadderLayer(self.config) for _ in range(len(self.sub_layers_ids))])
            self.finnal_norm = RMSNorm(4096)
            self.sigmoid = torch.nn.Sigmoid()
            self.activation_func = torch.nn.SiLU()
# ==========================================================================================


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # debug_tensor = {}
        stem_hidden_state = None
        # debug_tensor["input_ids"] = input_ids.clone().detach().int()
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        ladder_hidden_states = None
        # debug_tensor.append({"embd_output" : hidden_states.clone().detach()})
        # debug_tensor.append({"residual" : residual})

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # hidden_list.append({"layer_id" : i})
            # hidden_list.append({"layer" : layer})
            # hidden_list.append({"hidden_states" : hidden_states})
            # hidden_list.append({"residual" : residual})
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
            if self.with_ladder:
                if i in self.sub_layers_ids:
                    # debug_tensor['hidden_states'] = hidden_states.clone().detach()
                    # debug_tensor['residual'] = residual.clone().detach()
                    stem_hidden_state = hidden_states[-1].clone().contiguous() + residual[-1].clone().contiguous()
                    if i == 0:
                        # debug_tensor['layer_norm_input'] = stem_hidden_state.clone().detach()
                        ladder_hidden_states = self.input_scaler(stem_hidden_state)
                        # debug_tensor['layer_norm_output'] = ladder_hidden_states.clone().detach()
                    else:
                        # debug_tensor[f'layer_{i}_input'] = ladder_hidden_states.clone().detach()
                        ladder_layer = self.ladder_layers[self.sub_layers_ids.index(i) - 1]
                        gate_value = self.sigmoid(ladder_layer.merge_gates)
                        # debug_tensor[f'layer_{i}_gate_value'] = gate_value.clone().detach()
                        # debug_tensor[f'layer_{i}_stem_hidden_state'] = stem_hidden_state.clone().detach()
                        merged = torch.add(stem_hidden_state * gate_value, ladder_hidden_states * (1 - gate_value))
                        # debug_tensor[f'layer_{i}_merged'] = merged.clone().detach()
                        ladder_hidden_states = ladder_layer(merged)
                        # debug_tensor[f'layer_{i}_output'] = ladder_hidden_states.clone().detach()

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        # debug_tensor["ladder_hidden_states"] = ladder_hidden_states.clone().detach()

        if self.with_ladder:
            ladder_hidden_states = self.activation_func(ladder_hidden_states)
            ladder_hidden_states = self.output_layer(ladder_hidden_states)
            ladder_hidden_states = self.finnal_norm(ladder_hidden_states)
            
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states, ladder_hidden_states


class LlamaForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        with_ladder: Optional[bool] = False,
        sub_layers_ids:Optional[List[int]] = [],
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                prefix="model",
                                with_ladder=with_ladder,
                                sub_layers_ids=sub_layers_ids)
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output, ladder_hidden_states = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        return model_output, ladder_hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

# ==========================================================================================
    def compute_ladder_logits(
        self,
        ladder_hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        ladder_logits = torch.nn.functional.linear(ladder_hidden_states, self.lm_head.weight)
        return ladder_logits
# ==========================================================================================

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights_with_ladder(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        qkv_weight_helper = AixQkvWeightHelper(self)
        params_dict = dict(self.named_parameters())
        weight_dtype = params_dict[next(iter(params_dict.keys()))]
        for name, loaded_weight in weights:
            # if name == 'tok_embeddings.weight':
            #     source_name = 'model.embed_tokens.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name == 'output.weight':
            #     source_name = 'lm_head.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name == 'norm.weight':
            #     source_name = 'model.norm.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('attention.query_key_value.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.self_attn.qkv_proj.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     loaded_weight = qkv_weight_helper.permute_qkv_weights(source_name, loaded_weight)
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('feed_forward.w1.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.mlp.gate_up_proj.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name][:14464].shape == loaded_weight.shape, f"{name} shape error"
            #     (params_dict[source_name].data)[:14464]  = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('feed_forward.w3.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.mlp.gate_up_proj.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name][14464:].shape == loaded_weight.shape, f"{name} shape error"
            #     (params_dict[source_name].data)[14464:]  = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('attention.wo.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.self_attn.o_proj.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('feed_forward.w2.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.mlp.down_proj.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('attention_norm.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.input_layernorm.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            # elif name.startswith('layers.') and name.endswith('ffn_norm.weight'):
            #     layer_id = name.split('.')[1]
            #     source_name = 'model.layers.' + layer_id + '.post_attention_layernorm.weight'
            #     assert source_name in params_dict.keys(), f"{name} is not in params_dict"
            #     assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
            #     params_dict[source_name].data = loaded_weight.cuda()
            if name == 'merge_gates':
                for layer_id in range(loaded_weight.shape[0]):
                    source_name = 'model.ladder_layers.' + str(layer_id) + '.merge_gates'
                    assert source_name in params_dict.keys(), f"{name} is not in params_dict"
                    assert params_dict[source_name].shape == loaded_weight[layer_id].unsqueeze(0).shape, f"{name} shape error"
                    params_dict[source_name].data = loaded_weight[layer_id].unsqueeze(0).cuda()
            elif name.startswith("input_scaler"):
                source_name = 'model.' + name
                assert source_name in params_dict.keys(), f"{name} is not in params_dict"
                assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
                params_dict[source_name].data = loaded_weight.to(weight_dtype).cuda()
            elif name.startswith('layers.') and 'feed_forward.dense' in name:
                source_name = 'model.ladder_' + name
                assert source_name in params_dict.keys(), f"{name} is not in params_dict"
                assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
                params_dict[source_name].data = loaded_weight.cuda()
            elif name.startswith('layers.') and 'input_norm.weight' in name:
                source_name = 'model.ladder_' + name
                assert source_name in params_dict.keys(), f"{name} is not in params_dict"
                assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
                params_dict[source_name].data = loaded_weight.cuda()
            elif name.startswith('output_layer'):
                source_name = 'model.' + name
                assert source_name in params_dict.keys(), f"{name} is not in params_dict"
                assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
                params_dict[source_name].data = loaded_weight.cuda()
            elif name == 'finnal_norm.weight':
                source_name = 'model.' + name
                assert source_name in params_dict.keys(), f"{name} is not in params_dict"
                assert params_dict[source_name].shape == loaded_weight.shape, f"{name} shape error"
                params_dict[source_name].data = loaded_weight.cuda()
            else:
                raise ValueError(f"key {name} error.")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        
    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")
