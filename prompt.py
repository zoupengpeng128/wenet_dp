#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 xmly.Inc. All Rights Reserved.
# Author: pengpeng.zou@ximalaya.com (zou peng peng)
#Refer to part of the code of project https://github.com/THUDM/P-tuning-v2
from typing import Tuple, Union
import torch

class PromptEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self,
                 prompt_projection=True,
                 domain_prompt_size=64,
                 prompt_hidden_size=512,
                 domain_hidden_dropout_prob=0.1,
                 num_blocks=12,
                 attention_heads=4,
                 hidden_size=256,
                 ):
        super().__init__()
        self.pre_seq_len = domain_prompt_size
        self.num_blocks = num_blocks
        self.n_head = attention_heads
        self.n_embd = hidden_size // attention_heads
        self.prompt_projection = prompt_projection
        self.prompt_tokens = torch.arange(self.pre_seq_len).long()
        self.domain_dropout = torch.nn.Dropout(domain_hidden_dropout_prob)
        """
        self.embedding = torch.nn.Embedding(self.pre_seq_len, hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, prompt_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(prompt_hidden_size, self.num_blocks * 2 * hidden_size)
        )
        """
        if self.prompt_projection:
            # Use a two-layer MLP to encode the prompt
            self.embedding = torch.nn.Embedding(self.pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prompt_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prompt_hidden_size,  self.num_blocks * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(self.pre_seq_len, self.num_blocks * 2 * hidden_size)


    def forward(self, xs: torch.Tensor):
        batch_size = xs.size(0)
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(xs.device)

        if self.prompt_projection:
            prompt_tokens = self.embedding(prompt_tokens)
            past_key_values = self.trans(prompt_tokens)
        else:
            past_key_values = self.embedding(prompt_tokens)

        #prompt_tokens = self.embedding(prompt_tokens)
        #past_key_values = self.trans(prompt_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_blocks * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.domain_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def forward_no_split(self, xs: torch.Tensor):
        batch_size = xs.size(0)
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(xs.device)
        #"""
        if self.prompt_projection:
            prompt_tokens = self.embedding(prompt_tokens)
            past_key_values = self.trans(prompt_tokens)
        else:
            past_key_values = self.embedding(prompt_tokens)
        #"""
        #prompt_tokens = self.embedding(prompt_tokens)
        #past_key_values = self.trans(prompt_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_blocks * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.domain_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        return past_key_values
