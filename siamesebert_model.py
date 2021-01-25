#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn 

class SiameseModel(nn.Module):
    
    def __init__(self, single_model):
        super().__init__()
    
        self.single_model = single_model
        self.classifier = nn.Linear(3*768, 2)
        
    def forward(self, b_input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        
        max_seq_len = 256
        
        def divide_into_two(some_tensor):
            return some_tensor[:,:max_seq_len], some_tensor[:,max_seq_len:]
        
        def process_one_input(tokens_tensors,mask_tensors):
            encoded_layers, _ = self.single_model(input_ids=tokens_tensors, attention_mask=mask_tensors)
            sentence_embedding = torch.mean(encoded_layers[11], 1) #final layer
            del encoded_layers
            return sentence_embedding
        
        b_input_ids_0, b_input_ids_1 = divide_into_two(b_input_ids) 
        token_type_ids_0,token_type_ids_1 = divide_into_two(token_type_ids)
        attention_mask_0, attention_mask_1 = divide_into_two(attention_mask)
        
        feature_0 = process_one_input(b_input_ids_0, attention_mask_0) #this is [batch_size * 768] shape
        feature_1 = process_one_input(b_input_ids_1, attention_mask_1)
        
        vectors_concat = []
        vectors_concat.append(feature_0)
        vectors_concat.append(feature_1)
        vectors_concat.append(torch.abs(feature_0 - feature_1))
        features = torch.cat(vectors_concat, 1)
        output = self.classifier(features)
        loss_fct = nn.CrossEntropyLoss()
        if next_sentence_label is not None:
            loss = loss_fct(output, next_sentence_label.view(-1))
            return loss
        else:
            return output