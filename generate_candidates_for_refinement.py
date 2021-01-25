#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from tqdm import tqdm, trange
import argparse

def get_pairs_of_100_most_similar(start_index, end_index, sorted_indices):
    
    most_similar = []
    for i in trange(len(sorted_indices)):
        own_counter = 0
        for j in range(-1,-len(sorted_indices),-1):
            #ie both are not same index since we dont want to match a movie character with itself
            if i != int(sorted_indices[i][j]):
                if own_counter >= start_index:
                    most_similar.append([i, int(sorted_indices[i][j])])
                own_counter += 1
                if own_counter >= end_index:
                    break
    return most_similar

def save_most_similar_as_adj_list(start_index, end_index, most_similar, model_name):
    
    list_of_all_top_100_pairs = {}
    
    for i in tqdm(most_similar):
        count = i[0]
        item = i[1]
        if count not in list_of_all_top_100_pairs:
            list_of_all_top_100_pairs[count] = []
        list_of_all_top_100_pairs[count].append(item)
    
    output_filename = '{}_{}_to_{}.json'.format(model_name,start_index+1, end_index)
    
    with open(output_filename, 'w+') as outfile:
        json.dump(list_of_all_top_100_pairs, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run baseline models')
    parser.add_argument("model_name", default="SiameseBERT",
                        choices=['CEM', 'SiameseBERT'],
                        help='set Select and Refine model : \
                        CEM (Character Encoding Model), \
                        SiameseBERT (BERT with similarity loss function)')
    parser.add_argument("candidate_number", default="100",
                        help='number of candidate to select for refinement \
                        1 to 500 - see paper for recommended values')
    args = parser.parse_args()
    data_dir = "new/"

    if args.model_name == "CEM":
        data = np.load(data_dir + "new_whole_tropes_all_examples_only.npy")
        cosine_similarity_grid = cosine_similarity(data, Y=data)
        
    elif args.model_name == "SiameseBERT":
        cosine_similarity_grid = np.load(data_dir + "cosine_like_grid_siamese_proper.npy")
    
    sorted_indices = np.argsort(cosine_similarity_grid, axis=1)

    start_indexes = [0,100,200,300,400]
    
    for start_index in start_indexes:
        end_index = start_index + 100
        most_similar = get_pairs_of_100_most_similar(start_index, end_index, sorted_indices)
        save_most_similar_as_adj_list(start_index, end_index, most_similar, args.model_name)








