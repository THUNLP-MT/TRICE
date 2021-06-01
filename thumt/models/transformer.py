# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules
from thumt.models.modeling_mbart import MBartForConditionalGeneration
from transformers import MBartConfig


class AttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x, bias, memory=None, state=None):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y = self.attention(y, bias, memory, None)
        else:
            kv = [state["k"], state["v"]]
            y, k, v = self.attention(y, bias, memory, kv)
            state["k"], state["v"] = k, v

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class FFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(FFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 params.filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class TransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params)
            self.feed_forward = FFNSubLayer(params)

    def forward(self, x, bias):
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class TransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerDecoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params,
                                                    name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x = self.self_attention(x, attn_bias, state=state)
        x = self.encdec_attention(x, encdec_bias, memory)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(TransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_fine_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class TransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(TransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_fine_encoder_layers)])

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, attn_bias, encdec_bias, memory, state=None):
        for i, layer in enumerate(self.layers):
            if state is not None:
                x = layer(x, attn_bias, encdec_bias, memory,
                          state["decoder"]["layer_%d" % i])
            else:
                x = layer(x, attn_bias, encdec_bias, memory, None)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class Transformer(modules.Module):

    def __init__(self, params, name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.params = params

        if params.mbart_config_path and params.mbart_model_path:
            config = MBartConfig.from_json_file(params.mbart_config_path)
            state_dict = torch.load(params.mbart_model_path, map_location="cpu")
            self.mbart_model = MBartForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=None, config=config, state_dict=state_dict).model
        elif params.mbart_model_code:
            self.mbart_model = MBartForConditionalGeneration.from_pretrained(params.mbart_model_code).model
        else:
            raise ValueError("Unknown mbart loading scheme.")

        self.mbart_encoder = self.mbart_model.encoder
        self.mbart_decoder = self.mbart_model.decoder

        with utils.scope(name):
            if params.adapter_type == "FFN":
                self.adapter = FFNSubLayer(params)
            elif params.adapter_type == "Self-attn" and params.num_fine_encoder_layers > 0:
                self.adapter = TransformerEncoder(params)
            elif params.adapter_type == "Cross-attn" and params.num_fine_encoder_layers > 0:
                self.adapter = TransformerDecoder(params)
            elif params.adapter_type != "None":
                raise ValueError("Invalid adapter_type %s" % params.adapter_type)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        # self.reset_parameters()

    def position(self, channels, dtype, device):
        half_dim = channels // 2

        positions = torch.tensor([0, 1000], dtype=dtype,
                                  device=device)
        dimensions = torch.arange(half_dim, dtype=dtype,
                                  device=device)

        scale = math.log(10000.0) / float(half_dim - 1)
        dimensions.mul_(-scale).exp_()

        scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)

        if channels % 2 == 1:
            pad = torch.zeros([signal.shape[0], 1], dtype=dtype,
                              device=device)
            signal = torch.cat([signal, pad], axis=1)

        return torch.reshape(signal, [2, channels]).to(dtype=dtype, device=device)

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        if params.shared_source_target_embedding and svoc_size != tvoc_size:
            raise ValueError("Cannot share source and target embedding.")

        if not params.shared_embedding_and_softmax_weights:
            self.softmax_weights = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.softmax_weights, "softmax_weights")

        if not params.shared_source_target_embedding:
            self.source_embedding = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.target_embedding = torch.nn.Parameter(
                torch.empty([tvoc_size, params.hidden_size]))
            self.add_name(self.source_embedding, "source_embedding")
            self.add_name(self.target_embedding, "target_embedding")
        else:
            self.weights = torch.nn.Parameter(
                torch.empty([svoc_size, params.hidden_size]))
            self.add_name(self.weights, "weights")

        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.bias, "bias")

    @property
    def src_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.source_embedding

    @property
    def tgt_embedding(self):
        if self.params.shared_source_target_embedding:
            return self.weights
        else:
            return self.target_embedding

    @property
    def softmax_embedding(self):
        if not self.params.shared_embedding_and_softmax_weights:
            return self.softmax_weights
        else:
            return self.tgt_embedding

    def reset_parameters(self):
        nn.init.normal_(self.src_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.tgt_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

        if not self.params.shared_embedding_and_softmax_weights:
            nn.init.normal_(self.softmax_weights, mean=0.0,
                            std=self.params.hidden_size ** -0.5)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]

        hyp_seq = features["hypo"]
        hyp_mask = features["hypo_mask"]

        if self.params.separate_encode:
            encoder_output = self.mbart_encoder(input_ids=src_seq, attention_mask=src_mask)[0]
            encoder_output_hyp = self.mbart_encoder(input_ids=hyp_seq, attention_mask=hyp_mask)[0]
            encoder_output_all = torch.cat([encoder_output, encoder_output_hyp], axis=1)
        else:
            segment_embedding = self.position(self.params.hidden_size, device=src_seq.device, dtype=torch.float32)
            encoder_input_all = (src_seq, hyp_seq)
            attn_mask_all = torch.cat([src_mask, hyp_mask], axis=1)
            encoder_output_all = self.mbart_encoder(input_ids=encoder_input_all, attention_mask=attn_mask_all, 
                                           segment_embedding=segment_embedding if self.params.segment_embedding else None)[0]

        if self.params.adapter_type == "FFN":
            encoder_output_all = self.adapter(encoder_output_all)
        elif self.params.adapter_type == "Self-attn":
            enc_attn_bias_src = self.masking_bias(src_mask).to(encoder_output_all)
            enc_attn_bias_hyp = self.masking_bias(hyp_mask).to(encoder_output_all)
            enc_attn_bias_all = torch.cat([enc_attn_bias_src, enc_attn_bias_hyp], axis=-1)

            encoder_output_all = self.adapter(encoder_output_all, enc_attn_bias_all)
        elif self.params.adapter_type == "Cross-attn":
            enc_attn_bias_src = self.masking_bias(src_mask).to(encoder_output_all)
            enc_attn_bias_hyp = self.masking_bias(hyp_mask).to(encoder_output_all)
            encoder_output_src, encoder_output_hyp = \
                    torch.split(encoder_output_all, [features["source"].shape[1], features["hypo"].shape[1]], dim=1)
            adapter_output_src = self.adapter(encoder_output_src, enc_attn_bias_src, 
                                              enc_attn_bias_hyp, encoder_output_hyp)
            adapter_output_hyp = self.adapter(encoder_output_hyp, enc_attn_bias_hyp,
                                              enc_attn_bias_src, encoder_output_src)
            encoder_output_all = torch.cat([adapter_output_src, adapter_output_hyp], axis=1)

        state["encoder_output"] = encoder_output_all

        return state

    def decode(self, features, state, mode="infer"):
        
        def _get_causal_mask(decoder_input_ids):
            def _fill_with_neg_inf(t):
                """FP16-compatible function that fills a input_ids with -inf."""
                return t.float().fill_(float("-inf")).type_as(t)
            bsz, tgt_len = decoder_input_ids.size()
            tmp = _fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
            mask = torch.arange(tmp.size(-1))
            tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
            causal_mask = tmp.to(dtype=decoder_input_ids.dtype, device=decoder_input_ids.device)
            return causal_mask

        def _invert_mask(attention_mask):
            """Turns 1->0, 0->1, False->True, True-> False"""
            assert attention_mask.dim() == 2
            return attention_mask.eq(0)

        if mode != "infer":
            decoder_padding_mask = _invert_mask(features["target_mask"])
            causal_mask = _get_causal_mask(features["target"])
            use_cache = False
            past_key_values = None
        else:
            decoder_padding_mask, causal_mask = None, None
            use_cache = True
            past_key_values = state["past_key_values"] if "past_key_values" in state \
                                else None 

        if self.params.separate_cross_att:
            encoder_hidden_states = torch.split(state["encoder_output"], 
                                        [features["source"].shape[1], features["hypo"].shape[1]], dim=1)
            encoder_padding_mask = (features["source_mask"], features["hypo_mask"])
        else:
            encoder_hidden_states = state["encoder_output"]
            encoder_padding_mask = torch.cat([features["source_mask"], features["hypo_mask"]], axis=1)
        
        outputs = self.mbart_decoder(input_ids=features["target"],
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_padding_mask=encoder_padding_mask,
                                    decoder_padding_mask=decoder_padding_mask,
                                    decoder_causal_mask=causal_mask,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    return_dict=True,
                                    )

        if mode == "infer":
            state["past_key_values"] = outputs.past_key_values

        logits = torch.nn.functional.linear(outputs.last_hidden_state, self.mbart_model.shared.weight)

        if mode == "infer":
            logits = torch.squeeze(logits, axis=1)

        return logits, state

    def forward(self, features, labels, mode="train", level="sentence"):
        mask = features["target_mask"]

        state = {}
        state = self.encode(features, state)
        logits, _ = self.decode(features, state, mode=mode)
        loss = self.criterion(logits, labels)
        mask = mask.to(logits)

        if mode == "eval":
            if level == "sentence":
                return -torch.sum(loss * mask, 1)
            else:
                return  torch.exp(-loss) * mask - (1 - mask)

        return torch.sum(loss * mask) / torch.sum(mask)

    def empty_state(self, batch_size, device):
        state = {}

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<s>",
            eos="</s>",
            unk="<unk>",
            src_lang_tok="en_XX",
            hyp_lang_tok="de_DE",
            tgt_lang_tok="de_DE",
            hidden_size=1024,
            filter_size=2048,
            num_heads=8,
            num_fine_encoder_layers=1,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            mbart_config_path="",
            mbart_model_path="",
            mbart_model_code="facebook/mbart-large-cc25",
            separate_encode=False,
            segment_embedding=True,
            separate_cross_att=False,
            input_type="",
            adapter_type="None", # "None", "Cross-attn", "Self-attn", "FFN"
        )

        return params

    @staticmethod
    def base_params_v2():
        params = Transformer.base_params()
        params.attention_dropout = 0.1
        params.relu_dropout = 0.1
        params.learning_rate = 12e-4
        params.warmup_steps = 8000
        params.normalization = "before"
        params.adam_beta2 = 0.997

        return params

    @staticmethod
    def big_params():
        params = Transformer.base_params()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 5e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def big_params_v2():
        params = Transformer.base_params_v2()
        params.hidden_size = 1024
        params.filter_size = 4096
        params.num_heads = 16
        params.residual_dropout = 0.3
        params.learning_rate = 7e-4
        params.train_steps = 300000

        return params

    @staticmethod
    def default_params(name=None):
        if name == "base":
            return Transformer.base_params()
        elif name == "base_v2":
            return Transformer.base_params_v2()
        elif name == "big":
            return Transformer.big_params()
        elif name == "big_v2":
            return Transformer.big_params_v2()
        else:
            return Transformer.base_params()
