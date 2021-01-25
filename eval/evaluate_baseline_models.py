#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.feature_extraction.text import CountVectorizer
from evaluation_helper import generate_correct_pairs, generate_adj_list, report_all_scores

import argparse

def generate_eval_files(dataset_name):
    
    if dataset_name == "SiameseBERT":
        data_version = "new_reindexed" 
        
        cosine_similarity_grid = np.load(data_dir + "cosine_like_grid_siamese_proper.npy")
        
    else:
        data_version = "new"
        
        if dataset_name == "BoW":
            filename = data_dir + "test_all_examples_only_single.json"
            with open(filename, "r") as read_file:
                example_sentences = json.load(read_file)
            vectorizer = CountVectorizer()
            bow = vectorizer.fit_transform(example_sentences)
            data = bow
            
        elif dataset_name == "USE":
            data = np.load(data_dir +"new_tv_tropes_google_universal_encoder.npy")
            
        elif dataset_name == "CEM":
            data = np.load(data_dir +"new_whole_tropes_all_examples_only.npy")
        
        elif dataset_name == "BERT":
            data = np.load(data_dir +"new_unfinetuned_bert.npy")
        
        cosine_similarity_grid = cosine_similarity(data, Y=data)
    
    sorted_np_similarity_indices = np.argsort(cosine_similarity_grid, axis=1)
    correct_example_pairs_dict = generate_correct_pairs(data_version=data_version)
    adj_list = generate_adj_list(correct_example_pairs_dict) 
    
    return sorted_np_similarity_indices,correct_example_pairs_dict,adj_list
    
def main(dataset_name):
    sorted_np_similarity_indices,correct_example_pairs_dict,adj_list = generate_eval_files(dataset_name)
    all_scores = report_all_scores(dataset_name, sorted_np_similarity_indices=sorted_np_similarity_indices, 
                      correct_example_pairs_dict=correct_example_pairs_dict, adj_list=adj_list)
    
    print(' & '.join(all_scores))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run baseline models')
    parser.add_argument("encoder", default="BoW",
                        choices=['BoW', 'USE','CEM','BERT', 'SiameseBERT'],
                        help='set baseline model : \
                        BoW (Bag of Words), \
                        USE (Universal Sentence Encoder),\
                        CEM (Character Encoding Model), \
                        BERT (Bidirectional Encoder Representation from Trasnformers)\
                        SiameseBERT (BERT with similarity loss function)')
    args = parser.parse_args()
    data_dir = "new/"
    main(args.encoder)


