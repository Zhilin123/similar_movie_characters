#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def get_top_100_characters_based_on_exhaustive_comparison(exhaustive_pairs_file):

    with open(exhaustive_pairs_file, "r") as read_file:
        exhaustive_pairs = json.load(read_file)
    
    sorted_exhaustive_pairs = np.argsort(exhaustive_pairs)
    
    top_n = 100
    
    most_similar_based_on_exhaustive = []
    
    for i in range(100):
        most_similar_based_on_exhaustive.append(sorted_exhaustive_pairs[i][-top_n:])
    
    return most_similar_based_on_exhaustive

def get_overlap(sorted_indices, most_similar_based_on_exhaustive):
    cosine_similar = []
    
    for i in range(100):
        cosine_similar.append(sorted_indices[i,-500:])
    
    score = []
    
    for i in range(100):
        one_score = 0
        for element in most_similar_based_on_exhaustive[i]:
            if element in cosine_similar[i]:
                one_score += 1
        score.append(one_score)
    
    print("mean = ", np.mean(score))
    print("std deve = ",np.std(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run baseline models')
    parser.add_argument("exhaustive_pairs_filename", default="hundred_characters_with_all_sentences_exhaustive_pairwise.json")
    parser.add_argument("encoded_characters_filename", default="new_whole_tropes_all_examples_only.npy")
    parser.add_argument("data_dir", default="new/")
    args = parser.parse_args()
    most_similar_based_on_exhaustive = get_top_100_characters_based_on_exhaustive_comparison(args.exhaustive_pairs_filename)
    data = np.load(args.data_dir + args.encoded_characters_filename)
    cosine_similarity_grid = cosine_similarity(data, Y=data)
    sorted_indices = np.argsort(cosine_similarity_grid, axis=1)
    get_overlap(sorted_indices, most_similar_based_on_exhaustive)

