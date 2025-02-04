import cachetools
import math
import nle
import time

from typing import Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from nle import nethack
from torch import nn

import rl_baseline.torchbeast_encoder

from sample_factory.algorithms.appo.model_utils import (
    get_obs_shape, nonlinearity, create_standard_encoder, EncoderBase, register_custom_encoder
)
from sample_factory.utils.utils import log
from utils.forked_pdb import ForkedPdb


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class NLEMainEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, reward_model):
        super().__init__(cfg, timing)        

        obs_shape = get_obs_shape(obs_space)

        self.cfg = cfg
        self.reward_model = reward_model
        self.obs_shape = obs_shape

        self.obs_encoder = None
        self.add_obs_encoder = None
        self.vector_obs_head = None
        self.message_head = None
        self.encoder_out_size = 0

        if 'obs' in obs_shape: 
            # Use standard CNN for the image observation in "obs"
            # See all arguments with "-h" to change this head to e.g. ResNet
            self.obs_encoder = create_standard_encoder(cfg, obs_space, timing, encoder_subtype=cfg.encoder_subtype)
            self.encoder_out_size += self.obs_encoder.encoder_out_size

        if 'add_obs' in obs_shape:
            self.add_obs_encoder = create_standard_encoder(cfg, obs_space, timing, encoder_subtype='convnet_impala', obs_key='add_obs')
            self.encoder_out_size += self.add_obs_encoder.encoder_out_size

        if 'norm_blstats' in obs_shape and 'skipbl' not in self.cfg.experiment:
            bl_shape = obs_shape['norm_blstats'][0]
            if 'blgtenc' in self.cfg.experiment:
                bl_shape += 1

            if self.cfg.encoder_stats_model == 'legacy':
                self.vector_obs_head = nn.Sequential(
                    nn.Linear(bl_shape, 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                )
                out_size = 128
            elif self.cfg.encoder_stats_model == 'two_hid_layers':
                self.vector_obs_head = nn.Sequential(
                    nn.Linear(bl_shape, 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                )
                out_size = 128
            elif self.cfg.encoder_stats_model == 'three_hid_layers':
                self.vector_obs_head = nn.Sequential(
                    nn.Linear(bl_shape, 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                )
                out_size = 128
            
            else:
                raise NotImplementedError
            # Add vector_obs to NN output for more direct access by LSTM core
            self.encoder_out_size += out_size + bl_shape

        if 'message' in obs_shape or 'message_strs' in obs_shape:
            if not reward_model:
                if cfg.encoder_model_msg == 'legacy':
                    # _Very_ poor for text understanding.
                    self.message_head = nn.Sequential(
                        nn.Linear(obs_shape.message[0], 128),
                        nonlinearity(cfg),
                        nn.Linear(128, 128),
                        nonlinearity(cfg),
                    )
                    out_size = 128
                    self.encoder_out_size += out_size
                elif cfg.encoder_model_msg == 'torchbeast':
                    self.msg_hdim = 64
                    self.msg_edim = 32
                    self.num_chars = 256
                    self.char_lt = nn.Embedding(
                        self.num_chars, self.msg_edim, padding_idx=0
                    )
                    self.conv1 = nn.Conv1d(
                        self.msg_edim, self.msg_hdim, kernel_size=7
                    )
                    # remaining convolutions, relus, pools, and a small FC network
                    self.message_head = nn.Sequential(
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=3, stride=3),
                        # conv2
                        nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=3, stride=3),
                        # conv3
                        nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                        nn.ReLU(),
                        # conv4
                        nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                        nn.ReLU(),
                        # conv5
                        nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                        nn.ReLU(),
                        # conv6
                        nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=3, stride=3),
                        # fc receives -- [ B x h_dim x 5 ]
                        Flatten(),
                        nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                        nn.ReLU(),
                    )
                    self.encoder_out_size += 2 * self.msg_hdim
                elif cfg.encoder_model_msg == 'none':
                    self.message_head = None
            # LLM embeddings are only implemente for the reward model
            elif 'paraphrase' in cfg.encoder_model_msg and reward_model:
                self.message_head = SentenceEmbedding(
                    model_name=cfg.encoder_model_msg,
                    cache_size=50000,
                )
                out_size = self.message_head.embedding_model[1].pooling_output_dimension
                self.encoder_out_size += out_size
            elif 't5' in cfg.encoder_model_msg and reward_model:
                self.message_head = TextEmbedding(
                    model_name=cfg.encoder_model_msg,
                    cache_size=50000,
                    aggregation_mode=cfg.encoder_aggregation_mode_msg,
                )
                out_size = self.message_head.embedding_model.model.config.hidden_size
                self.encoder_out_size += out_size
            else:
                raise NotImplementedError

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        cats = []
        
        if self.obs_encoder is not None:
            # This one handles the "obs" key which contains the main image
            x = self.obs_encoder(obs_dict)
            cats.append(x)

        if self.add_obs_encoder is not None:
            # This one handles the "add_obs" key which contains the main image
            x = self.add_obs_encoder(obs_dict)
            cats.append(x)

        if self.vector_obs_head is not None:
            norm_blstats = obs_dict['norm_blstats'].float()

            if 'blgtenc' in self.cfg.experiment:
                xlvl_gt_dlvl = (obs_dict['norm_blstats'][:, 18] > obs_dict['norm_blstats'][:, 12]).unsqueeze(-1)
                norm_blstats = torch.cat([norm_blstats, xlvl_gt_dlvl], dim=1).float()

            vector_obs = self.vector_obs_head(norm_blstats)
            cats.append(vector_obs)
            cats.append(norm_blstats)

        if self.message_head is not None:
            if not self.reward_model:
                if self.cfg.encoder_model_msg == 'legacy':
                    message = self.message_head(obs_dict['message'].float() / 255.)
                elif self.cfg.encoder_model_msg == 'torchbeast':
                    char_emb = self.char_lt(obs_dict['message'].long()).transpose(1, 2)
                    message = self.message_head(self.conv1(char_emb))
            else:
                if self.cfg.encoder_model_msg == 'legacy':
                    message = self.message_head(obs_dict['message'].float() / 255.)
                else:
                    message = self.message_head(obs_dict['message_strs'])
            cats.append(message)

        x = torch.cat(cats, dim=1)

        return x

register_custom_encoder('nle_rgbcrop_encoder', NLEMainEncoder)
