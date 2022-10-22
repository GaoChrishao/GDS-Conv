import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules.speech_to_text import Adapter, CTC
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    LegacyRelPositionalEncoding,
    RelPositionalEncoding,
    S2TTransformerEncoderLayer,
    DynamicLinearCombination,
)
from fairseq.modules.speech_to_text import (
    subsampling
)

from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_transformer")
class S2TTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # subsampling
        parser.add_argument(
            "--shrink-value",
            type=str,
            metavar="D",
            help="shrink value",
        )

        parser.add_argument(
            "--pool-thresh",
            type=float,
            metavar="D",
            help="pool thresh",
        )

        parser.add_argument(
            "--random-mask",
            default=False,
            action='store_true',
            help="use random mask while training",
        )
        parser.add_argument(
            "--thresh-dynamic",
            default=False,
            action='store_true',
            help="use dynamic thresh",
        )

        parser.add_argument(
            '--decode-debug',
            action='store_false',
            help='while decoding, print some info'
        )
        parser.add_argument(
            "--gmm-iter",
            type=int,
            metavar="N",
            help="gmm iter num",
        )
        parser.add_argument(
            "--gmm-iter-per",
            type=int,
            metavar="N",
            help="gmm iter num",
        )

        parser.add_argument(
            "--gmm-prob",
            type=int,
            metavar="N",
            help="gmm min -prob",
        )
        parser.add_argument(
            "--gmm-num",
            type=int,
            metavar="N",
            help="gmm model num",
        )

        parser.add_argument(
            "--gmm-dim",
            type=int,
            metavar="N",
            help="gmm used dimention",
        )

        parser.add_argument(
            "--subsampling-type",
            type=str,
            help="subsampling type, like conv1d and conv2d",
        )
        parser.add_argument(
            "--subsampling-layers",
            type=int,
            help="subsampling layers",
        )
        parser.add_argument(
            "--subsampling-filter",
            type=int,
            help="subsampling filter",
        )
        parser.add_argument(
            "--subsampling-kernel",
            type=int,
            help="subsampling kernel",
        )
        parser.add_argument(
            "--subsampling-stride",
            type=int,
            help="subsampling stride",
        )
        parser.add_argument(
            "--subsampling-norm",
            type=str,
            default="none",
            help="subsampling normalization type",
        )
        parser.add_argument(
            "--subsampling-activation",
            type=str,
            default="none",
            help="subsampling activation function type",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "local",
                "selfattn",
                "reduced",
                "rel_selfattn",
                "relative",
                "rel_pos_legacy",
                "rel_pos",
                "rope",
                "abs",
            ],
            help="transformer encoder self-attention layer type"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "relative",
                "local",
            ],
            help="transformer decoder self-attention layer type"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument('--share-all-embeddings',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--max-encoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--max-decoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the encoder",
        )
        parser.add_argument(
            "--decoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the decoder",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        # DLCL
        parser.add_argument(
            "--use-enc-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument(
            "--use-dec-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument('--init-value', type=str, default='avg', choices=['avg', 'one'],
                            help='how to init the learned weight matrix')
        parser.add_argument('--weight-type', type=str, default='scalar',
                            help='type of learned weight [scalar, scalar_n(n>1), vector]')
        parser.add_argument('--encoder-learnable', type=eval, default='True',
                            help='enable to learn weights for encoder')
        parser.add_argument('--decoder-learnable', type=eval, default='True',
                            help='enable to learn weights for decoder')
        parser.add_argument('--normalize-learned-weight', type=eval, default='False',
                            help='normalize learned weight by softmax')
        parser.add_argument('--normalize-embedding', type=eval, default='False',
                            help='normalize the input of embedding')
        parser.add_argument('--history-dropout', type=float, default=0.0, metavar='D',
                            help='dropout for history output')
        parser.add_argument('--history-window-size', type=int, default='-1',
                            help='how many past layers are considered. -1 means all')
        # CTC
        parser.add_argument(
            "--ctc-layer",
            default=0,
            type=int,
            help="the position of the ctc loss",
        )

        # local modeling
        parser.add_argument(
            '--hard-mask-window',
            type=float,
            metavar="D",
            default=0,
            help='window size of local mask'
        )
        parser.add_argument(
            '--gauss-mask-sigma',
            type=float,
            metavar="D",
            default=0,
            help='standard deviation of the gauss mask'
        )
        parser.add_argument(
            '--init-mask-weight',
            type=float,
            metavar="D",
            default=0.5,
            help='initialized weight for local mask'
        )

        # Conformer setting
        parser.add_argument(
            "--encoder-activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="Whether to use macaron style for positionwise layer",
        )
        # Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the upper triangular part of attention matrix.",
        )
        # Relative positional encoding
        parser.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
                 "The legacy relative positional encoding will be deprecated in the future."
                 "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # CNN module
        parser.add_argument(
            "--use-cnn-module",
            default=False,
            type=bool,
            help="Use convolution module or not",
        )
        parser.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )

        # Simultaneous speech translation
        parser.add_argument(
            "--simul",
            default=False,
            action="store_true",
            help="Simultaneous speech translation or not",
        )
        # interleaved dropout
        parser.add_argument('--interleave-dropout', type=int,
                            help='interleaved dropout probability')
        parser.add_argument('--cl-dropout',
                            action="store_true",
                            default=False,
                            help='interleaved dropout probability')
        parser.add_argument('--cl-dropout-epoch',
                            type=int,
                            default=None,
                            help='interleaved dropout probability')
        parser.add_argument('--cl-dropout-strategy',
                            type=str,
                            help='interleaved dropout probability')
        # intermedia CTC loss
        parser.add_argument(
            "--intermedia-ctc-layers",
            default=None,
            type=str,
            help="the position of the ctc loss, separated by comma ",
        )
        parser.add_argument(
            "--intermedia-adapter",
            default="none",
            type=str,
            help="type of intermedia adapter",
        )
        parser.add_argument(
            "--intermedia-distribution-cutoff",
            default=None,
            type=int,
            help="cutoff of the distribution",
        )
        parser.add_argument(
            "--intermedia-drop-prob",
            default=0,
            type=float,
            help="probability of dropping the followed layers",
        )
        parser.add_argument(
            "--intermedia-temperature",
            default=1,
            type=float,
            help="temperature of the intermedia ctc probability",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TTransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):

        if getattr(args, "simul", False):
            from examples.simultaneous_translation.models.transformer_monotonic_attention import (
                TransformerMonotonicDecoder,
            )

            decoder = TransformerMonotonicDecoder(args, task.target_dictionary, embed_tokens)
        else:
            decoder = TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=False
            )

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))
        return cls(encoder, decoder)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class S2TTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        dim = args.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = subsampling(args)
        # self.linear = nn.Linear(dim, dim)

        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")

        if self.attn_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
            self.embed_positions = LegacyRelPositionalEncoding(
                args.encoder_embed_dim, args.dropout, args.max_source_positions
            )
        elif self.attn_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.layers = nn.ModuleList(
            [S2TTransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = None

        if args.use_enc_dlcl:
            self.history = DynamicLinearCombination(args, is_encoder=True)
        else:
            self.history = None

        # self.use_ctc = "sate" in args.arch or \
        #                (getattr(args, "criterion", "") == "ctc") or \
        #                (("ctc" in getattr(args, "criterion", "")) and (getattr(args, "ctc_weight", 0) > 0))
        self.use_ctc = "sate" in args.arch or getattr(args, "ctc_weight", 0) > 0
        if self.use_ctc:
            self.ctc_layer = args.ctc_layer
            self.inter_ctc = True if self.ctc_layer != 0 and self.ctc_layer != args.encoder_layers else False
            if self.inter_ctc:
                logger.info("Intermedia CTC loss in layer %d" % self.ctc_layer)
            self.ctc = CTC(dim,
                           dictionary_size=len(task.source_dictionary),
                           dropout=args.dropout,
                           need_layernorm=True if self.inter_ctc else False)

            if task.source_dictionary == task.target_dictionary and \
                    embed_tokens is not None and dim == embed_tokens.embedding_dim:
                self.ctc.ctc_projection.weight = embed_tokens.weight

        self.interleaved_dropout = getattr(args, "interleave_dropout", None)

        self.gather_cos_sim = getattr(args, "gather_cos_sim", False)
        # self.gather_cos_sim = True
        self.dis = 2
        self.cos_sim = dict()

        self.intermedia_ctc_layers = []

        if args.intermedia_ctc_layers is not None:
            intermedia_ctc_layers = args.intermedia_ctc_layers.split(",")
            for layer_idx in intermedia_ctc_layers:
                layer_idx = int(layer_idx)
                if layer_idx <= 0:
                    layer_idx += args.encoder_layers
                self.intermedia_ctc_layers.append(layer_idx)

                logger.info("Intermedia CTC loss in layer %d" % layer_idx)

            if not self.use_ctc:
                self.ctc = CTC(dim,
                               dictionary_size=len(task.source_dictionary),
                               dropout=args.dropout)

                if task.source_dictionary == task.target_dictionary and embed_tokens is not None:
                    self.ctc.ctc_projection.weight = embed_tokens.weight

            strategy = None
            if args.intermedia_adapter == "shrink":
                strategy = getattr(args, "ctc_compress_strategy", None)
            elif args.intermedia_adapter == "league":
                strategy = getattr(args, "intermedia_distribution_cutoff", None)
            self.adapter = Adapter(dim, args.intermedia_adapter,
                                   len(task.source_dictionary), strategy=strategy)
                                   # embed_tokens=embed_tokens if embed_tokens is not None else self.ctc.ctc_projection)
            self.intermedia_drop_prob = getattr(args, "intermedia_drop_prob", 0)
            self.intermedia_temperature = getattr(args, "intermedia_temperature", 1)

    @staticmethod
    def pooling_ratio():
        return 4

    def add_to_dict(self, x, dis, idx):
        sim = 0
        seq_len = x.size(0)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        for i in range(dis, seq_len - dis):
            a = x[i, :, :]
            for j in range(-dis, dis + 1):
                if j == 0:
                    continue
                b = x[i + j, :, :]
                sim_j = cos(a, b).mean()
                sim += sim_j
        sim = sim / 2 / dis / (seq_len - 2 * dis)

        if idx not in self.cos_sim:
            self.cos_sim[idx] = []
        self.cos_sim[idx].append(float(sim))

    def forward(self, src_tokens, src_lengths,**kwargs):

        if self.history is not None:
            self.history.clean()

        # gather cosine similarity
        cos_sim_idx = -1
        dis = self.dis
        if self.gather_cos_sim:
            self.add_to_dict(src_tokens.transpose(0, 1), dis, cos_sim_idx)

        # down-sampling
        # (B, T, D) -> (T, B, D)
        x = src_tokens.transpose(0, 1)
        x, input_lengths = self.subsample(x, src_lengths)

        # embedding scaling
        x = self.embed_scale * x

        # padding and position embedding
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        if self.attn_type in ["rel_pos", "rel_pos_legacy", "rel_selfattn"]:
            positions = self.embed_positions(x)

        elif self.attn_type == "rope":
            positions = None

        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            positions = None

        # x = self.linear(x)
        x = self.dropout_module(x)

        # add emb into history
        if self.history is not None:
            self.history.push(x)

        # gather cosine similarity
        cos_sim_idx = (cos_sim_idx + 10) // 10 * 10 - 1
        if self.gather_cos_sim:
            cos_sim_idx += 1
            self.add_to_dict(x, dis, cos_sim_idx)

        layer_idx = 0
        ctc_logit = None
        intermedia_ctc_logits = []
        for layer in self.layers:
            layer_idx += 1

            if self.history is not None:
                x = self.history.pop()

            if layer_idx != len(self.layers) \
                    and self.interleaved_dropout is not None \
                    and layer_idx % self.interleaved_dropout == 0:
                x = self.dropout_module(x)

            # encoder layer
            x = layer(x, encoder_padding_mask, pos_emb=positions)

            if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                ctc_logit = self.ctc(x.clone())

            # interleave CTC
            if layer_idx in self.intermedia_ctc_layers:
                if self.intermedia_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.intermedia_drop_prob:
                        break

                norm_x = self.layer_norm(x)
                logit = self.ctc(norm_x)

                intermedia_ctc_logits.append(logit)

                logit = logit.clamp(min=-1e8 if logit.dtype == torch.float32 else -1e4,
                                    max=1e8 if logit.dtype == torch.float32 else 1e4)

                prob = utils.softmax(logit / self.intermedia_temperature, dim=-1)
                x, encoder_padding_mask = self.adapter([x, prob], encoder_padding_mask)

            # gather cosine similarity
            if self.gather_cos_sim:
                cos_sim_idx += 1
                self.add_to_dict(x, dis, cos_sim_idx)

            if self.history is not None:
                self.history.push(x)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_ctc and ctc_logit is None:
            ctc_logit = self.ctc(x)

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "intermedia_ctc_logits": intermedia_ctc_logits,  # B x T x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_ctc_logit = (
            [] if len(encoder_out["ctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["ctc_logit"] if x is not None]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "ctc_logit": new_ctc_logit,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, extra

    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)


@register_model_architecture(model_name="s2t_transformer", arch_name="s2t_transformer")
def base_architecture(args):
    # gch's Convolutional subsampler
    args.shrink_value = getattr(args, "shrink_value", "0.5")
    args.random_mask = getattr(args, "random_mask", False)
    args.thresh_dynamic = getattr(args, "thresh_dynamic", False)
    args.pool_thresh = getattr(args, "pool_thresh", 0)
    args.decode_debug = getattr(args, "decode_debug", False)
    args.gmm_iter = getattr(args, "gmm_iter", 100)
    args.gmm_iter_per = getattr(args, "gmm_iter_per", 10)
    args.gmm_prob = getattr(args, "gmm_prob", 75)
    args.gmm_num = getattr(args, "gmm_num", 2)
    args.gmm_dim = getattr(args, "gmm_dim", 80)

    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)

    # Conformer
    args.encoder_activation_fn = getattr(args, "encoder_activation_fn", "relu")
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)

    # settings for DLCL
    args.use_enc_dlcl = getattr(args, "use_enc_dlcl", False)
    args.use_dec_dlcl = getattr(args, "use_dec_dlcl", False)
    args.init_value = getattr(args, 'init_value', 'avg')
    args.weight_type = getattr(args, 'weight_type', 'scalar')
    args.encoder_learnable = getattr(args, 'encoder_learnable', True)
    args.decoder_learnable = getattr(args, 'decoder_learnable', True)
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)

    # local modeling
    args.hard_mask_window = getattr(args, 'hard_mask_window', 0)
    args.gauss_mask_sigma = getattr(args, 'gauss_mask_sigma', 0)
    args.init_mask_weight = getattr(args, 'init_mask_weight', 0)

    # interleaved dropout
    args.interleave_dropout = getattr(args, "interleave_dropout", None)
    args.cl_dropout = getattr(args, "cl_dropout", False)
    args.cl_dropout_epoch = getattr(args, "cl_dropout_epoch", None)
    args.cl_dropout_strategy = getattr(args, "cl_dropout_strategy", "linear")

    # intermedia CTC
    args.intermedia_ctc_layers = getattr(args, "intermedia_ctc_layers", None)
    args.intermedia_adapter = getattr(args, "intermedia_adapter", None)
    args.intermedia_drop_prob = getattr(args, "intermedia_drop_prob", 0)


@register_model_architecture("s2t_transformer", "s2t_transformer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_s_relative")
def s2t_transformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_xs")
def s2t_transformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_sp")
def s2t_transformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_s(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_m")
def s2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_mp")
def s2t_transformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_m(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_l")
def s2t_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_transformer", "s2t_transformer_lp")
def s2t_transformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_transformer_l(args)
