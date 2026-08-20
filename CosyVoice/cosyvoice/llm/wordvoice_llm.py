# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
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
import hashlib
import json
import os, queue
import random
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Generator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path
from cosyvoice.utils.losses import DynamicStyleLoss


@dataclass(frozen=True)
class PreparedWordVoicePrefix:
    key: str
    hidden: torch.Tensor
    cache: tuple
    word_embeddings: tuple[torch.Tensor, ...]
    word_index: int
    preceding_duration: int
    durations: tuple[int, ...]
    boundaries: tuple[int, ...]
    tones: tuple[int, ...]
    energies: tuple[int, ...]
    pitches: tuple[int, ...]
    final_duration: int
    final_boundary: int
    prepare_seconds: float
    retained_bytes: int


def _hash_tensor(digest, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode('utf-8'))
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(repr(tuple(value.shape)).encode('ascii'))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _cache_tuple(cache) -> tuple:
    legacy = cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache
    return tuple((key, value) for key, value in legacy)


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = self.speech_token_size
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.long)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        if ignore_eos is True:
            weighted_scores[self.speech_token_size] = -float('inf')
        top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False)
            if top_ids == self.eos_token:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = speech_token_size
        self.fill_token = speech_token_size + 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v2.batch.onnx'))

    def prepare_lm_input_target(self, sos_emb, text_token, text_token_emb, text_token_len, task_id_emb,
                                speech_token, speech_token_emb, speech_token_len, words_token_emb, 
                                words_len, start_times, end_times, boundary_list, tone_list, f0_list, energy_list,
                                instruct_token=None, instruct_token_emb=None, instruct_token_len=None):
        lm_target, lm_input = [], []
        device = text_token.device  # 获取当前所在设备
        
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)
        words_token_emb = unpad_sequence(words_token_emb, words_len.cpu(), batch_first=True)
        
        # NOTE add instruct_token in CosyVoice3
        # if instruct_token is not None and instruct_token_emb is not None and instruct_token_len is not None:
        instruct_token = unpad_sequence(instruct_token, instruct_token_len.cpu(), batch_first=True)
        instruct_token_emb = unpad_sequence(instruct_token_emb, instruct_token_len.cpu(), batch_first=True)

        dur_target_list, pau_target_list, boundary_target_list, tone_target_list, word_lm_positions_list = [], [], [], [], []
        
        for i in range(len(text_token)):
            # 加上 device=device，确保 Target 生成在正确的设备上
            this_lm_target = torch.tensor([IGNORE_ID] * (1 + instruct_token_len[i].item() + text_token_len[i].item()) + speech_token[i].tolist() + [self.eos_token], dtype=torch.long, device=device)
            this_lm_input = torch.concat([sos_emb.squeeze(dim=0), instruct_token_emb[i], text_token_emb[i], task_id_emb.squeeze(dim=0), speech_token_emb[i]], dim=0)
            
            # 【修复点】：增加 device=device，防止过 embedding 时 CPU/GPU 张量冲突
            durations = torch.tensor([end_times[i][k] - start_times[i][k] for k in range(len(start_times[i]))], dtype=torch.long, device=device)
            durations = torch.clamp(durations, 0, self.max_duration - 1)
            pauses = torch.tensor([
                start_times[i][k + 1] - end_times[i][k] if k < len(end_times[i]) - 1 else speech_token_len[i] - end_times[i][k]
                for k in range(len(start_times[i]))
            ], dtype=torch.long, device=device)
            pauses = torch.clamp(pauses, 0, self.max_pause - 1)

            boundary = torch.tensor(boundary_list[i], dtype=torch.long, device=device)
            boundary = torch.clamp(boundary, 0, self.max_boundary - 1)
            tone = torch.tensor(tone_list[i], dtype=torch.long, device=device)
            tone = torch.clamp(tone, 0, self.max_tone - 1)
            f0 = f0_list[i].to(device=device, dtype=torch.long)
            energy = energy_list[i].to(device=device, dtype=torch.long)

            boundary_target_list.append(boundary.clone())
            tone_target_list.append(tone.clone())
            dur_target_list.append(durations.clone())
            pau_target_list.append(pauses.clone())
            durations_input = durations.clone()
            pauses_input = pauses.clone()

            if self.training:
                # 对各个字级输入做随机掩码处理
                mask_prob = 0.3
                total_mask = torch.rand(durations_input.shape, device=device) < 0.1  # 10% 的概率全都掩码掉，增加训练时的鲁棒性
                dur_mask = (torch.rand(durations_input.shape, device=device) < mask_prob) | total_mask
                pau_mask = (torch.rand(pauses_input.shape, device=device) < mask_prob) | total_mask
                bnd_mask = (torch.rand(boundary.shape, device=device) < mask_prob) | total_mask
                tone_mask = (torch.rand(tone.shape, device=device) < mask_prob) | total_mask
                f0_mask = (torch.rand(f0.shape, device=device) < mask_prob) | total_mask
                eng_mask = (torch.rand(energy.shape, device=device) < mask_prob) | total_mask
                # 将掩码位置替换为原先设定的最大长度索引，即代表 <MASK>
                durations_input[dur_mask] = self.max_duration
                pauses_input[pau_mask] = self.max_pause
                boundary[bnd_mask] = self.max_boundary
                tone[tone_mask] = self.max_tone
                f0[f0_mask] = self.max_f0
                energy[eng_mask] = self.max_energy
            
            # 计算嵌入
            dur_emb = self.duration_embedding(durations_input)
            pau_emb = self.pause_embedding(pauses_input)
            bnd_emb = self.boundary_embedding(boundary)
            tone_emb = self.tone_embedding(tone)
            f0_emb = self.f0_embedding(f0)
            eng_emb = self.energy_embedding(energy)
            # style_emb = torch.stack([words_token_emb[i], dur_emb, pau_emb], dim=0).mean(dim=0) # xxh: 停顿时间对比
            style_emb = torch.stack([words_token_emb[i], dur_emb, bnd_emb, tone_emb, f0_emb, eng_emb], dim=0).mean(dim=0)

            # insert word-level duration tokens
            offset = 0
            word_positions = []
            for k, start_idx in enumerate(start_times[i]):
                idx = 1 + instruct_token_len[i].item() + text_token_len[i].item() + start_idx + offset
                
                # target序列
                this_lm_target = torch.cat([
                    this_lm_target[:idx],
                    torch.tensor([self.bound_token, IGNORE_ID], dtype=this_lm_target.dtype, device=device),
                    this_lm_target[idx:]
                ])
                # input序列
                this_lm_input = torch.cat([
                    this_lm_input[:idx+1],
                    self.speech_embedding.weight[self.bound_token].unsqueeze(0),
                    style_emb[k].unsqueeze(0),
                    this_lm_input[idx+1:]
                ], dim=0)
                word_positions.append(idx + 1)
                offset += 2
                
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
            word_lm_positions_list.append(torch.tensor(word_positions, dtype=torch.long, device=device))
            
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.long, device=device)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID).to(device)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)
        dur_target = pad_sequence(dur_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        pau_target = pad_sequence(pau_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        bnd_target = pad_sequence(boundary_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        tone_target = pad_sequence(tone_target_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        f0_target = pad_sequence(f0_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        eng_target = pad_sequence(energy_list, batch_first=True, padding_value=IGNORE_ID).to(device)
        word_lm_positions = pad_sequence(word_lm_positions_list, batch_first=True, padding_value=IGNORE_ID).to(device)

        return lm_target, lm_input, lm_input_len, dur_target, pau_target, bnd_target, tone_target, f0_target, eng_target, word_lm_positions

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        # 1. encode text_token
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        words_token = batch['words'].to(device)
        words_token_emb = self.llm.model.model.embed_tokens(words_token)
        words_len = batch['word_len'].to(device)
        start_times = batch['start']
        end_times = batch['end']
        boundary_list = batch['boundary']
        tone_list = batch['tone']
        f0_list = batch['f0']
        f0_list = [torch.clamp(torch.floor((f0 + 1) / 2 * self.max_f0), min=0, max=self.max_f0 - 1).to(torch.long) for f0 in f0_list]  # 量化为20个区间，并限制最大值为 self.max_f0
        energy_list = batch['energy']
        energy_list = [torch.clamp(torch.floor(energy * self.max_energy), min=0, max=self.max_energy - 1).to(torch.long) for energy in energy_list]  # 量化为20个区间，并限制最大值为 self.max_energy

        # 2. encode speech_token
        if 'speech_token' not in batch:
            speech_token, speech_token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            speech_token = batch['speech_token'].to(device)
            speech_token_len = batch['speech_token_len'].to(device)
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. sos and task_id
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. prepare llm_input/target
        instruct_token = batch['instruct_token'].to(device)
        instruct_token_len = batch['instruct_token_len'].to(device)
        instruct_token_emb = self.llm.model.model.embed_tokens(instruct_token)
        (lm_target, lm_input, lm_input_len, dur_target, pau_target, 
         bnd_target, tone_target, f0_target, eng_target, word_lm_positions) = self.prepare_lm_input_target(
                                                                            sos_emb, text_token, text_token_emb, text_token_len, 
                                                                            task_id_emb, speech_token, speech_token_emb, speech_token_len,
                                                                            words_token_emb, words_len, start_times, end_times, 
                                                                            boundary_list, tone_list, f0_list, energy_list,
                                                                            instruct_token, instruct_token_emb, instruct_token_len)
        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len)
        logits = self.llm_decoder(lm_output)
        speech_loss = self.criterion_ce(logits, lm_target)

        # 5. extract word-level embeddings for duration/pause
        lm_style_hiddens = []
        for b in range(word_lm_positions.size(0)):
            positions = word_lm_positions[b]
            positions = positions[positions != IGNORE_ID]
            lm_style_hiddens.append(lm_output[b, positions, :])

        lm_style_hiddens = pad_sequence(lm_style_hiddens, batch_first=True)

        # 7. style attributions prediction
        dur_pred = self.duration_predictor(lm_style_hiddens)
        # pau_pred = self.pause_predictor(lm_style_hiddens)
        bnd_pred = self.boundary_predictor(lm_style_hiddens)
        tone_pred = self.tone_predictor(lm_style_hiddens)
        f0_pred = self.f0_predictor(lm_style_hiddens)
        eng_pred = self.energy_predictor(lm_style_hiddens)

        # 8. flatten for cross entropy
        torch.set_printoptions(threshold=torch.inf)

        speech_loss = speech_loss / (2 * torch.exp(self.log_sigma_speech)**2) + self.log_sigma_speech
        # dur_loss = F.cross_entropy(dur_pred.view(-1, self.max_duration+1), dur_target.view(-1), ignore_index=IGNORE_ID)
        dur_loss = self.style_loss_module.duration_loss(dur_pred, dur_target)
        dur_loss = dur_loss / (2 * torch.exp(self.log_sigma_dur)**2) + self.log_sigma_dur
        # pau_loss = F.cross_entropy(pau_pred.view(-1, self.max_pause+1), pau_target.view(-1), ignore_index=IGNORE_ID)
        bnd_loss = self.style_loss_module.boundary_loss(bnd_pred, bnd_target)
        bnd_loss = bnd_loss / (2 * torch.exp(self.log_sigma_bnd)**2) + self.log_sigma_bnd
        tone_loss = self.style_loss_module.tone_loss(tone_pred, tone_target)
        tone_loss = tone_loss / (2 * torch.exp(self.log_sigma_tone)**2) + self.log_sigma_tone
        f0_loss = self.style_loss_module.f0_loss(f0_pred, f0_target) * 2
        f0_loss = f0_loss / (2 * torch.exp(self.log_sigma_f0)**2) + self.log_sigma_f0
        eng_loss = self.style_loss_module.energy_loss(eng_pred, eng_target)
        eng_loss = eng_loss / (2 * torch.exp(self.log_sigma_eng)**2) + self.log_sigma_eng

        # total_loss = speech_loss + dur_loss + pau_loss
        total_loss = speech_loss + dur_loss + bnd_loss + tone_loss + f0_loss + eng_loss
        acc = th_accuracy(logits.view(-1, self.llm_decoder.out_features), lm_target, ignore_label=IGNORE_ID)
        
        return {'loss': total_loss, 'speech_loss': speech_loss, 'acc': acc, 'dur_loss': dur_loss, 
                'bnd_loss': bnd_loss, 'tone_loss': tone_loss, 'f0_loss': f0_loss, 'eng_loss': eng_loss}

    def prepared_prefix_key(
            self,
            text: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            word_list: List[torch.Tensor],
            start_list: List[int],
            dur_list: List[int],
            bnd_list: List[int],
            tone_list: List[int],
            eng_list: List[int],
            f0_list: List[int],
            embedding: torch.Tensor,
    ) -> str:
        digest = hashlib.sha256()
        native_decoder = getattr(self.llm, 'native_decoder', None)
        decoder_manifest = getattr(native_decoder, 'manifest', None)
        decoder_identity = {
            'decoder_class': (
                type(native_decoder).__name__ if native_decoder is not None else 'eager'
            ),
            'manifest': decoder_manifest if isinstance(decoder_manifest, dict) else {},
            'sampling': (
                getattr(self.sampling, '__module__', type(self.sampling).__module__),
                getattr(self.sampling, '__qualname__', type(self.sampling).__qualname__),
            ),
            'model_class': type(self.llm.model).__name__,
        }
        digest.update(
            b'prepared-prefix-runtime-identity:'
            + json.dumps(decoder_identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
        )
        for name, tensor in (
                ('text', text),
                ('prompt_text', prompt_text),
                ('prompt_speech_token', prompt_speech_token),
                ('embedding', embedding),
        ):
            _hash_tensor(digest, name, tensor)
        for index, word in enumerate(word_list):
            _hash_tensor(digest, f'word_{index}', word)
        for name, values in (
                ('starts', start_list),
                ('durations', dur_list),
                ('boundaries', bnd_list),
                ('tones', tone_list),
                ('energies', eng_list),
                ('pitches', f0_list),
        ):
            digest.update(name.encode('ascii'))
            digest.update(repr(tuple(values)).encode('ascii'))
        return digest.hexdigest()

    def consume_prepared_prefix_metrics(self):
        metrics = self._prepared_prefix_metrics
        self._prepared_prefix_metrics = None
        return metrics

    @torch.inference_mode()
    def base_inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            word_list: List[torch.Tensor],       # 字 token 列表
            start_list: List[int],               # 字在音频中的起始索引
            dur_list: List[int],                 # 时长物理量
            bnd_list: List[int],
            tone_list: List[int],
            eng_list: List[int],
            f0_list: List[int],
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
            better_infer = True,
            prepared_prefix_key: str = '',
            prepared_prefix_fingerprint_seconds: float = 0.0,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len

        if self.__class__.__name__ == 'CosyVoice3LM':
            assert 151646 in text, '<|endofprompt|> not detected in CosyVoice3 text or prompt_text, check your input!'

        # 常量预分配
        precomputed_bound_emb = self.speech_embedding.weight[self.bound_token].view(1, 1, -1)
        # 注意：这里改为 1D tensor，以便后续 logits[:, forbidden_ids] 索引不报错
        forbidden_stop_ids = torch.tensor([self.eos_token], device=device, dtype=torch.long)
        forbidden_bound_ids = torch.tensor([self.bound_token], device=device, dtype=torch.long)

        reuse_enabled = bool(
            prepared_prefix_key
            and getattr(self.llm, 'native_decoder', None) is not None
        )
        prefix = self._prepared_prefix
        cache_hit = bool(reuse_enabled and prefix is not None and prefix.key == prepared_prefix_key)
        if not cache_hit:
            prepare_started = time.perf_counter()
            text_emb = self.llm.model.model.embed_tokens(text)
            style_embs = []
            word_embs_list = []
            for i in range(len(word_list)):
                w_emb = self.llm.model.model.embed_tokens(word_list[i].to(device))[0,0,:]
                word_embs_list.append(w_emb)
                d = torch.clamp(
                    torch.tensor([dur_list[i]], device=device, dtype=torch.long),
                    0,
                    self.max_duration - 1,
                )
                b = torch.tensor([bnd_list[i]], device=device, dtype=torch.long)
                t = torch.tensor([tone_list[i]], device=device, dtype=torch.long)
                e = torch.tensor([eng_list[i]], device=device, dtype=torch.long)
                f = torch.tensor([f0_list[i]], device=device, dtype=torch.long)
                style_embs.append(torch.stack([
                    w_emb,
                    self.duration_embedding(d).view(-1),
                    self.boundary_embedding(b).view(-1),
                    self.tone_embedding(t).view(-1),
                    self.f0_embedding(f).view(-1),
                    self.energy_embedding(e).view(-1),
                ], dim=0).mean(dim=0).view(1, 1, -1))

            word_idx = 0
            prompt_embs_list = []
            P = prompt_speech_token_len.item() if isinstance(prompt_speech_token_len, torch.Tensor) else prompt_speech_token_len
            pre_dur = 0
            if P > 0:
                p_speech_emb = self.speech_embedding(prompt_speech_token).squeeze(0)
                for i in range(P):
                    if word_idx < len(start_list) and start_list[word_idx] == i:
                        prompt_embs_list.append(self.speech_embedding.weight[self.bound_token])
                        prompt_embs_list.append(style_embs[word_idx].view(-1))
                        if word_idx > 0:
                            dur_list[word_idx-1] = pre_dur
                            pre_dur = 0
                        word_idx += 1
                    if word_idx > 0:
                        pre_dur += 1
                    prompt_embs_list.append(p_speech_emb[i])
                while word_idx < len(start_list) and start_list[word_idx] == P:
                    prompt_embs_list.append(self.speech_embedding.weight[self.bound_token])
                    prompt_embs_list.append(style_embs[word_idx].view(-1))
                    dur_list[word_idx-1] = pre_dur
                    pre_dur = 0
                    word_idx += 1
                prompt_speech_token_emb = torch.stack(prompt_embs_list, dim=0).unsqueeze(0)
            else:
                prompt_speech_token_emb = torch.zeros(
                    1, 0, self.llm_input_size, dtype=text_emb.dtype, device=device
                )

            current_input = torch.concat([
                self.speech_embedding.weight[self.sos].reshape(1, 1, -1),
                text_emb,
                self.speech_embedding.weight[self.task_id].reshape(1, 1, -1),
                prompt_speech_token_emb,
            ], dim=1)
            prefill_mask = torch.tril(torch.ones(
                (1, current_input.shape[1], current_input.shape[1]),
                device=device,
                dtype=torch.bool,
            ))
            hidden, cache = self.llm.forward_one_step(
                current_input,
                masks=prefill_mask,
                cache=None,
            )
            if hidden.is_cuda:
                torch.cuda.current_stream(hidden.device).synchronize()
            immutable_cache = _cache_tuple(cache)
            prepare_seconds = time.perf_counter() - prepare_started
            last_hidden = hidden[:, -1:, :].clone()
            prefix = PreparedWordVoicePrefix(
                key=prepared_prefix_key,
                hidden=last_hidden,
                cache=immutable_cache,
                word_embeddings=tuple(word_embs_list),
                word_index=word_idx,
                preceding_duration=pre_dur,
                durations=tuple(dur_list),
                boundaries=tuple(bnd_list),
                tones=tuple(tone_list),
                energies=tuple(eng_list),
                pitches=tuple(f0_list),
                final_duration=dur_list[word_idx-1],
                final_boundary=bnd_list[word_idx-1],
                prepare_seconds=prepare_seconds,
                retained_bytes=(
                    _tensor_bytes(last_hidden)
                    + _tensor_bytes(immutable_cache)
                    + _tensor_bytes(word_embs_list)
                ),
            )
            if reuse_enabled:
                self._prepared_prefix = prefix
                self._prepared_prefix_misses += 1
        else:
            self._prepared_prefix_hits += 1

        restore_started = time.perf_counter()
        dur_list = list(prefix.durations)
        bnd_list = list(prefix.boundaries)
        tone_list = list(prefix.tones)
        eng_list = list(prefix.energies)
        f0_list = list(prefix.pitches)
        word_embs_list = prefix.word_embeddings
        word_idx = prefix.word_index
        pre_dur = prefix.preceding_duration
        final_dur = prefix.final_duration
        final_bnd = prefix.final_boundary
        cache = prefix.cache
        prefill_hidden = prefix.hidden
        pau_list = [0] * len(word_list)
        restore_seconds = time.perf_counter() - restore_started
        self._prepared_prefix_metrics = {
            'prepared_prefix_cache': 'hit' if cache_hit else ('miss' if reuse_enabled else 'disabled'),
            'prepared_prefix_cache_hit': cache_hit,
            'prepared_prefix_fingerprint_seconds': round(prepared_prefix_fingerprint_seconds, 6),
            'prepared_prefix_prepare_seconds': round(0.0 if cache_hit else prefix.prepare_seconds, 6),
            'prepared_prefix_source_prepare_seconds': round(prefix.prepare_seconds, 6),
            'prepared_prefix_restore_seconds': round(restore_seconds, 6),
            'prepared_prefix_retained_bytes': prefix.retained_bytes if reuse_enabled else 0,
            'prepared_prefix_hits': self._prepared_prefix_hits,
            'prepared_prefix_misses': self._prepared_prefix_misses,
        }

        # 开始推理：Prefill & Decode
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        true_dur = pre_dur
        generate_speech_token = []
        decode_mask = torch.ones((1, 1, 1), device=device, dtype=torch.bool)
        for step in range(max_len):
            if step == 0:
                lm_output = prefill_hidden
            else:
                lm_output, cache = self.llm.forward_one_step(
                    current_input,
                    masks=decode_mask,
                    cache=cache
                )
            # logits = self.llm_decoder(lm_output[:, -1, :]) 
            logits = self.llm_decoder(lm_output[:, -1])
            
            # 控制停止符与边界符
            if word_idx < len(word_list):
                logits[:, forbidden_stop_ids] = -float('inf')
            # else:
            #     logits[:, forbidden_bound_ids] = negative_inf
            
            if better_infer is True:
                if true_dur < final_dur:
                    # 比指定的时长短则强制模型输出有声音的token
                    logits[:, self.silent_tokens] = -float('inf')
                    logits[:, forbidden_bound_ids] = -float('inf')
                elif true_dur == final_dur:
                    # 留一帧缓冲
                    logits[:, forbidden_bound_ids] = -float('inf')
                elif true_dur > final_dur:
                    # 最低静音与最大静音设置
                    pause_len = true_dur - final_dur
                    allows_silence = pause_len < self.max_pause_lens[final_bnd]
                    allows_terminal = pause_len > self.max_pause_lens[final_bnd-1]
                    if allows_silence:
                        mask_indices = self._wordvoice_decode_mask_indices(logits.device)[
                            1 if allows_terminal else 0
                        ]
                    elif allows_terminal:
                        mask_indices = self._wordvoice_decode_mask_indices(logits.device)[2]
                    else:
                        mask_indices = self._wordvoice_decode_mask_indices(logits.device)[3]

                    logits.index_fill_(1, mask_indices, -float('inf'))

            # 采样
            token_id = self.sampling_ids(logits.squeeze(dim=0), generate_speech_token, sampling, ignore_eos=True if step < min_len else False)
            if token_id in self.silent_tokens:
                pau_list[word_idx-1] += 1

            if better_infer is True:
                if true_dur == final_dur and final_bnd == 0:
                    token_id = self.bound_token

            if word_idx == len(word_list):
                if token_id == self.eos_token or token_id == self.bound_token:
                    break

            # 【核心修改区】：动态预测时长与停顿，构建 tot_emb
            if token_id == self.bound_token:
                # 1. 既然输出了 bound_token，我们需要先把它送进网络，拿到上下文 Hidden State
                inner_output, cache = self.llm.forward_one_step(
                    precomputed_bound_emb,
                    masks=decode_mask,
                    cache=cache
                )
                user_dur = dur_list[word_idx]
                user_bnd = bnd_list[word_idx]
                user_tone = tone_list[word_idx]
                user_f0 = f0_list[word_idx]
                user_eng = eng_list[word_idx]

                requested_attributes = (
                    user_dur,
                    user_bnd,
                    user_tone,
                    user_f0,
                    user_eng,
                )
                masked_attributes = (
                    self.max_duration,
                    self.max_boundary,
                    self.max_tone,
                    self.max_f0,
                    self.max_energy,
                )
                if any(
                    requested == masked
                    for requested, masked in zip(requested_attributes, masked_attributes)
                ):
                    hidden_state = inner_output[:, -1, :]
                    predicted_attributes = torch.stack((
                        self.duration_predictor(hidden_state).argmax(dim=-1),
                        self.boundary_predictor(hidden_state).argmax(dim=-1),
                        self.tone_predictor(hidden_state).argmax(dim=-1),
                        self.f0_predictor(hidden_state).argmax(dim=-1),
                        self.energy_predictor(hidden_state).argmax(dim=-1),
                    )).flatten().tolist()
                else:
                    predicted_attributes = requested_attributes

                pred_dur, pred_bnd, pred_tone, pred_f0, pred_eng = predicted_attributes
                final_dur = pred_dur if user_dur == self.max_duration else user_dur # xxh
                final_bnd = pred_bnd if user_bnd == self.max_boundary else user_bnd
                final_tone = pred_tone if user_tone == self.max_tone else user_tone
                final_f0 = pred_f0 if user_f0 == self.max_f0 else user_f0
                final_eng = pred_eng if user_eng == self.max_energy else user_eng

                if better_infer == True:
                    if user_bnd == self.max_boundary:
                        final_bnd = min(final_bnd, 3) # 减少长停顿
                    if user_eng == self.max_energy:
                        final_eng = max(final_eng, min(eng_list)) # 防止极端小声

                bnd_list[word_idx] = final_bnd
                tone_list[word_idx] = final_tone
                f0_list[word_idx] = final_f0
                eng_list[word_idx] = final_eng

                # 安全截断
                final_dur = min(max(final_dur, 0), self.max_duration - 1)
                
                # 4. 动态组装 Style Embedding
                w_emb = word_embs_list[word_idx]
                d_emb = self.duration_embedding.weight[final_dur]
                b_emb = self.boundary_embedding.weight[final_bnd]
                t_emb = self.tone_embedding.weight[final_tone]
                f_emb = self.f0_embedding.weight[final_f0]
                e_emb = self.energy_embedding.weight[final_eng]
                
                dynamic_tot_emb = torch.stack([w_emb, d_emb, b_emb, t_emb, f_emb, e_emb], dim=0).mean(dim=0).view(1, 1, -1)
                
                # 5. 将组装好的字特征作为主循环的下一个输入
                current_input = dynamic_tot_emb[:]

                dur_list[word_idx-1] = true_dur
                true_dur = 0

                bnd_list[word_idx] = final_bnd
                tone_list[word_idx] = final_tone
                f0_list[word_idx] = final_f0
                eng_list[word_idx] = final_eng

                word_idx += 1
            else: # 输出语音token
                true_dur += 1
                generate_speech_token.append(token_id)
                current_input = self.speech_embedding.weight[token_id].reshape(1, 1, -1)
        dur_list[-1] = true_dur
        
        return generate_speech_token, dur_list, bnd_list, tone_list, f0_list, eng_list, pau_list

class WordVoiceLM(Qwen2LM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.bound_token = speech_token_size + 3

        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 200,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.max_duration = 35
        self.max_pause = 200
        self.max_boundary = 5
        self.max_tone = 7
        self.max_f0 = 20
        self.max_energy = 20
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        self.max_pause_lens = [0, 1, 4, 10, 15]
        self._wordvoice_mask_cache_device = None
        self._wordvoice_mask_indices = None

        self.speech_embedding = torch.nn.Embedding(speech_token_size + 200, llm_input_size)
        self.duration_embedding = torch.nn.Embedding(self.max_duration + 1, llm_input_size)
        self.pause_embedding = torch.nn.Embedding(self.max_pause + 1, llm_input_size)
        self.boundary_embedding = torch.nn.Embedding(self.max_boundary + 1, llm_input_size)
        self.tone_embedding = torch.nn.Embedding(self.max_tone + 1, llm_input_size)
        self.f0_embedding = torch.nn.Embedding(self.max_f0 + 1, llm_input_size)
        self.energy_embedding = torch.nn.Embedding(self.max_energy + 1, llm_input_size)
        self.duration_predictor = nn.Linear(llm_output_size, self.max_duration + 1)
        self.pause_predictor = nn.Linear(llm_output_size, self.max_pause + 1)
        self.boundary_predictor = nn.Linear(llm_output_size, self.max_boundary + 1)
        self.tone_predictor = nn.Linear(llm_output_size, self.max_tone + 1)
        self.f0_predictor = nn.Linear(llm_output_size, self.max_f0 + 1)
        self.energy_predictor = nn.Linear(llm_output_size, self.max_energy + 1)

        # 初始化可学习的不确定性参数，初值可为 1.0
        self.log_sigma_speech = nn.Parameter(torch.zeros(()))  # log σ, 避免负值问题
        self.log_sigma_dur = nn.Parameter(torch.zeros(()))
        self.log_sigma_bnd = nn.Parameter(torch.zeros(()))
        self.log_sigma_tone = nn.Parameter(torch.zeros(()))
        self.log_sigma_f0 = nn.Parameter(torch.zeros(()))
        self.log_sigma_eng = nn.Parameter(torch.zeros(()))
        self.style_loss_module = DynamicStyleLoss(
            max_bnd_class = self.max_boundary + 1,
            max_tone_class = self.max_tone + 1,
            max_f0_class = self.max_f0 + 1,
            max_energy_class = self.max_energy + 1,
            max_dur_class = self.max_duration + 1,
            ignore_id = IGNORE_ID
        )

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(200)]
        self.vllm_output_queue = {}
        self._prepared_prefix = None
        self._prepared_prefix_metrics = None
        self._prepared_prefix_hits = 0
        self._prepared_prefix_misses = 0
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))

    def _wordvoice_decode_mask_indices(self, device):
        """Return cached indices for the three WordVoice silence masks."""
        if self._wordvoice_mask_cache_device != device:
            vocab_size = self.llm_decoder.out_features

            def masked_indices(exempt_ids):
                mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
                mask[torch.tensor(exempt_ids, dtype=torch.long, device=device)] = False
                return torch.nonzero(mask, as_tuple=True)[0]

            silent = list(self.silent_tokens)
            self._wordvoice_mask_indices = (
                masked_indices(silent),
                masked_indices(silent + [self.bound_token, self.eos_token]),
                masked_indices([self.bound_token, self.eos_token]),
                masked_indices([self.bound_token]),
            )
            self._wordvoice_mask_cache_device = device
        return self._wordvoice_mask_indices
