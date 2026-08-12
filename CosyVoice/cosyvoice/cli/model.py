# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import os
from typing import Any, Generator, cast
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper

class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=False)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=False)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            if isinstance(text, Generator):
                assert (self.__class__.__name__ != 'CosyVoiceModel') and not hasattr(self.llm, 'vllm'), 'streaming input text is only implemented for CosyVoice2/3 and do not support vllm!'
                token_generator = self.llm.inference_bistream(text=text,
                                                              prompt_text=prompt_text.to(self.device),
                                                              prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                              prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                              prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                              embedding=llm_embedding.to(self.device))
            else:
                token_generator = self.llm.inference(text=text.to(self.device),
                                                     text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device),
                                                     uuid=uuid)
            for i in token_generator:
                if i in self.silent_tokens:
                    cur_silent_token_num += 1
                    if cur_silent_token_num > max_silent_token_num:
                        continue
                else:
                    cur_silent_token_num = 0
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid])

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # NOTE increase token_hop_len incrementally to avoid duplicate inference
        self.token_max_hop_len = 4 * self.token_hop_len
        self.stream_scale_factor = 2
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = []

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            while True:
                time.sleep(0.1)
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False)
                    token_offset += this_token_hop_len
                    self.token_hop_len = min(self.token_max_hop_len, self.token_hop_len * self.stream_scale_factor)
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=token_offset,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


class CosyVoice3Model(CosyVoice2Model):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # NOTE increase token_hop_len incrementally to avoid duplicate inference
        self.token_max_hop_len = 4 * self.token_hop_len
        self.stream_scale_factor = 2
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.tts_word_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.flow_runtime_metrics = {'flow_backend': 'eager'}
        self.flow_amp_dtype = None
        # FSQ silent and breath token
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    def enable_wordvoice_flow_amp(self, dtype):
        if dtype != 'bf16':
            raise ValueError('WordVoice flow AMP supports only bf16')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('WordVoice bf16 flow AMP requires a CUDA bf16-capable GPU')
        parity = self._validate_wordvoice_flow_bf16()
        self.flow_amp_dtype = torch.bfloat16
        self.flow_runtime_metrics = {
            'flow_backend': 'pytorch_amp',
            'flow_precision': 'bf16',
            'flow_parity': parity,
        }

    def _validate_wordvoice_flow_bf16(self):
        estimator = cast(Any, self.flow).decoder.estimator
        if not isinstance(estimator, torch.nn.Module):
            raise RuntimeError('WordVoice bf16 flow AMP requires the PyTorch flow estimator')
        relative_rmse_limit = 0.025
        cosine_similarity_limit = 0.9998
        max_relative_rmse = 0.0
        min_cosine_similarity = 1.0
        was_training = estimator.training
        estimator.eval()
        try:
            for sequence_length in (64, 256, 1024):
                generator = torch.Generator(device=self.device).manual_seed(sequence_length)
                inputs = (
                    torch.randn(2, 80, sequence_length, device=self.device, generator=generator),
                    torch.ones(2, 1, sequence_length, device=self.device),
                    torch.randn(2, 80, sequence_length, device=self.device, generator=generator),
                    torch.tensor([0.25, 0.25], device=self.device),
                    torch.randn(2, 80, device=self.device, generator=generator),
                    torch.randn(2, 80, sequence_length, device=self.device, generator=generator),
                )
                with torch.inference_mode():
                    expected = estimator(*inputs, streaming=False)
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        actual = estimator(*inputs, streaming=False)
                nonfinite = {
                    name: int((~torch.isfinite(output)).sum().item())
                    for name, output in (('expected', expected), ('actual', actual))
                    if not torch.isfinite(output).all()
                }
                if nonfinite:
                    raise RuntimeError(
                        'WordVoice bf16 flow AMP gate failed; '
                        f'stage=parity-nonfinite; sequence_length={sequence_length}; '
                        f'nonfinite={nonfinite}'
                    )
                expected_float = expected.float()
                difference = actual.float() - expected_float
                rmse = torch.sqrt(torch.mean(difference.square()))
                expected_rms = torch.sqrt(torch.mean(expected_float.square()))
                relative_rmse = float((rmse / expected_rms).item())
                cosine_similarity = float(
                    F.cosine_similarity(
                        actual.float().flatten(), expected_float.flatten(), dim=0
                    ).item()
                )
                max_relative_rmse = max(max_relative_rmse, relative_rmse)
                min_cosine_similarity = min(min_cosine_similarity, cosine_similarity)
        finally:
            estimator.train(was_training)
        parity = {
            'sequence_lengths': [64, 256, 1024],
            'max_relative_rmse': round(max_relative_rmse, 6),
            'min_cosine_similarity': round(min_cosine_similarity, 8),
        }
        if (
            max_relative_rmse > relative_rmse_limit
            or min_cosine_similarity < cosine_similarity_limit
        ):
            raise RuntimeError(
                'WordVoice bf16 flow AMP gate failed; stage=parity; '
                f'limits={{"max_relative_rmse": {relative_rmse_limit}, '
                f'"min_cosine_similarity": {cosine_similarity_limit}}}; '
                f'actual={parity}'
            )
        return parity

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # append mel cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel = self.hift_cache_dict[uuid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uuid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech

    def wordvoice_token2wav(self, token, start_id, dur_list, bnd_list, tone_list, f0_list, eng_list, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            flow_started = time.perf_counter()
            flow_autocast = torch.cuda.amp.autocast(
                enabled=self.flow_amp_dtype is not None,
                dtype=self.flow_amp_dtype or torch.float16,
            )
            with flow_autocast:
                tts_mel, flow_timing_events = self.flow.inference_wordvoice(token=token.to(self.device, dtype=torch.int32),
                                                 token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                 prompt_token=prompt_token.to(self.device),
                                                 prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                 start_id=start_id,
                                                 dur_list=dur_list,
                                                 bnd_list=bnd_list,
                                                 tone_list=tone_list,
                                                 f0_list=f0_list,
                                                 eng_list=eng_list,
                                                 prompt_feat=prompt_feat.to(self.device),
                                                 prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                 embedding=embedding.to(self.device),
                                                 streaming=stream,
                                                 finalize=finalize)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            flow_seconds = time.perf_counter() - flow_started
            flow_detail_metrics = {}
            for index in range(len(flow_timing_events) - 1):
                metric_name, started = flow_timing_events[index]
                _, finished = flow_timing_events[index + 1]
                flow_detail_metrics[metric_name] = round(started.elapsed_time(finished) / 1000, 3)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # append mel cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel = self.hift_cache_dict[uuid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uuid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            vocoder_started = time.perf_counter()
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            vocoder_seconds = time.perf_counter() - vocoder_started
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        stage_metrics = {
            'flow_seconds': round(flow_seconds, 3),
            'flow_sequence_frames': int((token.shape[1] + prompt_token.shape[1]) * self.flow.token_mel_ratio),
            'vocoder_seconds': round(vocoder_seconds, 3),
        }
        stage_metrics.update(flow_detail_metrics)
        return tts_speech, stage_metrics

    def wordvoice_llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, word_list, start_list, dur_list, bnd_list, tone_list, eng_list, f0_list, uuid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        fingerprint_started = time.perf_counter()
        prepared_prefix_key = self.llm.prepared_prefix_key(
            text,
            prompt_text,
            llm_prompt_speech_token,
            word_list,
            start_list,
            dur_list,
            bnd_list,
            tone_list,
            eng_list,
            f0_list,
            llm_embedding,
        )
        fingerprint_seconds = time.perf_counter() - fingerprint_started
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            (
                self.tts_speech_token_dict[uuid],
                dur_list,
                bnd_list,
                tone_list,
                f0_list, 
                eng_list,
                pau_list
                ) = self.llm.base_inference(
                    text=text.to(self.device),
                    text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_text=prompt_text.to(self.device),
                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                    word_list=word_list,
                    start_list=start_list,
                    dur_list=dur_list,
                    bnd_list=bnd_list,
                    tone_list=tone_list,
                    eng_list=eng_list,
                    f0_list=f0_list,
                    embedding=llm_embedding.to(self.device),
                    uuid=uuid,
                    prepared_prefix_key=prepared_prefix_key,
                    prepared_prefix_fingerprint_seconds=fingerprint_seconds)
        self.llm_end_dict[uuid] = True

        return dur_list, bnd_list, tone_list, f0_list, eng_list, pau_list

    def wordvoice_tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            word_list=[], start_list=[], dur_list=[], bnd_list=[], tone_list=[], eng_list=[], f0_list=[],
            stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.tts_word_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], [], False
            self.hift_cache_dict[this_uuid] = None

        llm_started = time.perf_counter()
        dur_list, bnd_list, tone_list, f0_list, eng_list, pau_list = self.wordvoice_llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, word_list, start_list, dur_list, bnd_list, tone_list, eng_list, f0_list, this_uuid)
        llm_seconds = time.perf_counter() - llm_started
        native_runtime_metrics = self.llm.llm.consume_native_metrics()
        prepared_prefix_metrics = self.llm.consume_prepared_prefix_metrics()

        prompt_words_len = len(start_list)
        bnd_control = bnd_list[prompt_words_len:]
        tone_control = tone_list[prompt_words_len:]
        dur_control = [40*(d-p) for d,p in zip(dur_list[prompt_words_len:], pau_list[prompt_words_len:])]
        f0_control = [round((f-9.5)/10, 3) for f in f0_list[prompt_words_len:]]
        eng_control = [round((e+0.5)/20, 3) for e in eng_list[prompt_words_len:]]
        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        speech_token_sha256 = hashlib.sha256(
            this_tts_speech_token.contiguous().numpy().tobytes()
        ).hexdigest()
        prosody_control_sha256 = hashlib.sha256(repr((
            tuple(dur_control),
            tuple(bnd_control),
            tuple(tone_control),
            tuple(f0_control),
            tuple(eng_control),
        )).encode('utf-8')).hexdigest()
        flow_started = time.perf_counter()
        this_tts_speech, stage_metrics = self.wordvoice_token2wav(token=this_tts_speech_token,
                                            start_id=start_list[0],
                                            dur_list=dur_list,
                                            bnd_list=bnd_list,
                                            tone_list=tone_list,
                                            f0_list=f0_list,
                                            eng_list=eng_list,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            token_offset=0,
                                            uuid=this_uuid,
                                            finalize=True,
                                            speed=speed)
        this_tts_speech = this_tts_speech.cpu()
        flow_seconds = time.perf_counter() - flow_started
        runtime_metrics: dict[str, Any] = {
            'llm_seconds': round(llm_seconds, 3),
            'flow_vocoder_seconds': round(flow_seconds, 3),
            'speech_token_count': int(this_tts_speech_token.shape[1]),
            'speech_token_sha256': speech_token_sha256,
            'prosody_control_sha256': prosody_control_sha256,
        }
        runtime_metrics.update(stage_metrics)
        runtime_metrics.update(self.flow_runtime_metrics)
        if native_runtime_metrics is not None:
            runtime_metrics.update(native_runtime_metrics)
        if prepared_prefix_metrics is not None:
            runtime_metrics.update(prepared_prefix_metrics)
        yield {'tts_speech': this_tts_speech,
               'dur_list': dur_control, 
               'bnd_list': bnd_control, 
               'tone_list': tone_control, 
               'f0_list': f0_control, 
               'eng_list': eng_control,
               'runtime_metrics': runtime_metrics}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.tts_word_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
