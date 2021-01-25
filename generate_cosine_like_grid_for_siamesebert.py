#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from pytorch_pretrained_bert import BertModel
from torch import nn 
from siamesebert_model import SiameseModel
import argparse


def load_classifier(root_dir):
    single_model = BertModel.from_pretrained('.')
    model = SiameseModel(single_model)
    
    bin_name = root_dir+"tropes_all_examples_only_siamese_proper.bin"
    
    model_state_dict = torch.load(bin_name)
    model.load_state_dict(model_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    classifier = model.classifier
    return classifier

def find_and_save_prob_similar(sentence_embedding_filename, output_cosine_like_grid_npy_filename):
    data = np.load(sentence_embedding_filename)
    all_probabilities = []
    for i in range(len(data)):
        one_probability = []
        feature_0 = data[i]
        for j in range(len(data)):
            feature_1 = data[j]
            vectors_concat = []
            vectors_concat.append(feature_0)
            vectors_concat.append(feature_1)
            vectors_concat.append(torch.abs(feature_0 - feature_1))
            features = torch.cat(vectors_concat, 1)
            output = classifier(features)
            probability = float((torch.nn.Softmax()(output))[0][0])
            one_probability.append(probability)
        all_probabilities.append(one_probability)
            
    np.save(output_cosine_like_grid_npy_filename, np.array(all_probabilities))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run baseline models')
    parser.add_argument("root_dir", default="")
    parser.add_argument("sentence_embedding_filename", default="siamese_all_examples_only.npy")
    parser.add_argument("output_cosine_like_grid_npy_filename", default="siamese_all_examples_cosine_like_grid.npy")
    args = parser.parse_args()
    classifier = load_classifier(args.root_dir)
    find_and_save_prob_similar(args.sentence_embedding_filename, args.output_cosine_like_grid_npy_filename)

