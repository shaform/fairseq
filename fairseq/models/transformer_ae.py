# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    BaseFairseqModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    Linear,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    VectorQuantizer
)
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer_ae")
class TransformerAutoEncodeModel(BaseFairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        source_encoder (TransformerEncoder): the source encoder
        target_encoder (TransformerEncoder): the target encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {}
        # fmt: on

    def __init__(self, args, source_encoder, target_encoder, vq_embed_tokens, decoder):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.vq_embed_tokens = vq_embed_tokens
        self.decoder = decoder
        assert isinstance(self.source_encoder, FairseqEncoder)
        assert isinstance(self.target_encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        parser.add_argument('--encoder-vq-beta', type=float, metavar='D',
                            help='scale of commitment loss')
        parser.add_argument('--encoder-vq-decay', type=float, metavar='D',
                            help='decay of vq')
        parser.add_argument('--encoder-vq-n-token', type=int, metavar='N',
                            help='')
        parser.add_argument('--encoder-vq-rho-start', type=float, metavar='D',
                            help='initial rate of discretized vectors')
        parser.add_argument('--encoder-vq-rho-warmup-updates', type=int, metavar='N',
                            help='how many steps to warmup')
        parser.add_argument('--encoder-vq-length', type=int, metavar='N',
                            help='length of encoded vectors')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary) + 1
            latent_idx = num_embeddings - 1
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb, latent_idx

        if args.encoder_embed_dim != args.decoder_embed_dim:
            raise ValueError(
                "requires --source-embed-dim to match --target-embed-dim"
            )
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            source_encoder_embed_tokens, source_latent_idx = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            target_encoder_embed_tokens = source_encoder_embed_tokens
            target_latent_idx = source_latent_idx
            decoder_embed_tokens = source_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            source_encoder_embed_tokens, source_latent_idx = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            target_encoder_embed_tokens, target_latent_idx = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
            decoder_embed_tokens, _ = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        source_encoder = cls.build_encoder(
                args, src_dict, source_encoder_embed_tokens,
                latent_idx=source_latent_idx,
                latent_len=args.encoder_vq_length)
        target_encoder = cls.build_encoder(
                args, tgt_dict, target_encoder_embed_tokens,
                latent_idx=target_latent_idx,
                latent_len=args.encoder_vq_length)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        vq_embed_tokens = VectorQuantizer(
                args.encoder_vq_n_token,
                args.encoder_embed_dim,
                decay=args.encoder_vq_decay)
        return cls(args, source_encoder, target_encoder, vq_embed_tokens, decoder)

    @classmethod
    def build_encoder(
            cls,
            args,
            src_dict,
            embed_tokens,
            latent_idx,
            latent_len
    ):
        return TransformerFixedSizeEncoder(
                args, src_dict, embed_tokens,
                latent_idx=latent_idx,
                latent_len=latent_len)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def compute_rho(self, num_updates):
        if self.args.encoder_vq_rho_warmup_updates <= 0:
            return 1.
        iter_ratio = min(1., num_updates / self.args.encoder_vq_rho_warmup_updates)
        rho = np.exp(
                np.log(self.args.encoder_vq_rho_start) * (1. - iter_ratio) +
                iter_ratio * np.log(1.))
        return rho

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        tgt_tokens,
        tgt_lengths,
        prev_output_tokens,
        num_updates=None,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if self.training:
            assert num_updates is not None
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        rho = self.compute_rho(num_updates)
        src_encoder_out = self.source_encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
            vq_embed_tokens=self.vq_embed_tokens,
            rho=rho,
        )
        src_latent_lengths = (
                torch.ones_like(src_lengths) * self.source_encoder.latent_len)
        src_decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=src_encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_latent_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        tgt_encoder_out = self.target_encoder(
            src_tokens=tgt_tokens,
            src_lengths=tgt_lengths,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
            vq_embed_tokens=self.vq_embed_tokens,
            rho=rho,

        )
        tgt_latent_lengths = (
                torch.ones_like(tgt_lengths) * self.target_encoder.latent_len)
        tgt_decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=tgt_encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=tgt_latent_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        mse = (src_encoder_out.mse + tgt_encoder_out.mse) * self.args.encoder_vq_beta
        return src_decoder_out, tgt_decoder_out, mse

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.source_encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


FixedEncoderOut = NamedTuple(
    "FixedEncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("mse", Tensor),
    ],
)


class TransformerFixedSizeEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, latent_idx, latent_len):
        super().__init__(args, dictionary, embed_tokens)
        self.latent_idx = latent_idx
        self.latent_len = latent_len

    def forward(
        self,
        src_tokens,
        src_lengths,
        vq_embed_tokens,
        rho,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        src_lengths = src_lengths + self.latent_len
        bsz = src_tokens.size()[0]
        latent_tokens = torch.ones(
                (bsz, self.latent_len),
                dtype=src_tokens.dtype,
                device=src_tokens.device) * self.latent_idx
        src_tokens = torch.cat([latent_tokens, src_tokens], dim=1)

        outputs = super().forward(
                src_tokens,
                src_lengths,
                cls_input,
                return_all_hiddens)
        encoder_out = outputs.encoder_out[:self.latent_len, :, :]
        encoder_padding_mask = outputs.encoder_padding_mask[:, :self.latent_len]
        encoder_embedding = outputs.encoder_embedding[:, :self.latent_len, :]
        if outputs.encoder_states is None:
            encoder_states = None
        else:
            encoder_states = [
                    state[:self.latent_len, :, :]
                    for state in outputs.encoder_states]

        encoder_out, mse = vq_embed_tokens(
                encoder_out, rho=rho, return_mse=True)


        return FixedEncoderOut(
            encoder_out=encoder_out,  # L x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x L
            encoder_embedding=encoder_embedding,  # B x L x C
            encoder_states=encoder_states,  # List[L x B x C]
            mse=mse.sum(),
        )


@register_model_architecture("transformer_ae", "transformer_ae")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.encoder_vq_beta = getattr(args, "encoder_vq_beta", 0.25)
    args.encoder_vq_decay = getattr(args, "encoder_vq_decay", 0.99)
    args.encoder_vq_n_token = getattr(args, "encoder_vq_n_token", 10000)
    args.encoder_vq_rho_start = getattr(args, "encoder_vq_rho_start", 0.1)
    args.encoder_vq_rho_warmup_updates = getattr(args, "encoder_vq_rho_warmup_updates", 10000)
    args.encoder_vq_length = getattr(args, "encoder_vq_length", 5)


@register_model_architecture("transformer_ae", "transformer_ae_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer_ae", "transformer_ae_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_ae", "transformer_ae_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer_ae", "transformer_ae_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer_ae", "transformer_ae_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_ae", "transformer_ae_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer_align", "transformer_ae_align")
def transformer_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)


@register_model_architecture("transformer_align", "transformer_ae_wmt_en_de_big_align")
def transformer_wmt_en_de_big_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    transformer_wmt_en_de_big(args)
