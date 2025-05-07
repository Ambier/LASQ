#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import math
import pandas as pd
import transformers
from tqdm import tqdm
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import MT5ForConditionalGeneration, T5Tokenizer, BertModel, BertTokenizer
from torch.nn import CosineSimilarity

class XiaokaoLLM_promax(nn.Module):
    def __init__(self, cbert, hidden_size=768, num_layers=2, nhead=8):
        super(XiaokaoLLM_promax, self).__init__()

        self.cbert = cbert

        # 单独的 Transformer 编码器层用于每个特征的上下文学习
        self.individual_transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=8, #nhead, 
            dim_feedforward=hidden_size, # 前馈神经网络的隐藏层维度，通常为d_model的2-4倍
            batch_first=True
        )
        self.individual_transformer1 = nn.TransformerEncoder(self.individual_transformer_layer, num_layers=num_layers)
        self.individual_transformer2 = nn.TransformerEncoder(self.individual_transformer_layer, num_layers=num_layers)
        self.individual_transformer3 = nn.TransformerEncoder(self.individual_transformer_layer, num_layers=num_layers)
        self.individual_transformer4 = nn.TransformerEncoder(self.individual_transformer_layer, num_layers=num_layers)

        # 总体 Transformer 编码器层用于 Cross-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(hidden_size, 1)

    def extract_hidden_state(self, inputs):
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)
        token_type_ids = inputs['token_type_ids'].squeeze(1)

        embedding = self.cbert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        ).last_hidden_state

        return embedding  # 不使用 `CLS`，返回完整序列

    def forward(self, qwen_eval, question, key_pair, pair_sentence):
        # 提取 BERT 最后一层的序列输出
        key_hidden_state = self.extract_hidden_state(key_pair)
        sentence_hidden_state = self.extract_hidden_state(pair_sentence)
        question_hidden_state = self.extract_hidden_state(question)
        qwen_eval_hidden_state = self.extract_hidden_state(qwen_eval)

        # 对每个特征进行单独的上下文学习
        key_hidden_state = self.individual_transformer1(key_hidden_state)
        sentence_hidden_state = self.individual_transformer2(sentence_hidden_state)
        question_hidden_state = self.individual_transformer3(question_hidden_state)
        qwen_eval_hidden_state = self.individual_transformer4(qwen_eval_hidden_state)

        # 拼接为序列，进行跨句子交互
        combined_sequences = torch.cat(
            [key_hidden_state, sentence_hidden_state, question_hidden_state, qwen_eval_hidden_state], 
            dim=1
        )

        # Transformer 编码器建模
        transformer_output = self.transformer_encoder(combined_sequences)

        # 聚合为全局序列表示，进行最大池化或平均池化
        pooled_output = transformer_output.mean(dim=1)  # 平均池化

        # 输出层
        overall_result = self.fc(pooled_output)
        overall_result = torch.sigmoid(overall_result.squeeze(-1))
        return overall_result

