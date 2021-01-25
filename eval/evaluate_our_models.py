#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
from tqdm import trange
import argparse
from evaluation_helper import report_all_scores,\
                            generate_correct_pairs, generate_adj_list
                            

def enter_one_top_100_into_similarity_table(top_pairs_file=None, top_pairs_similarity_file=None, limit=None, similarity_table=None):
    # load in the most simple of the 100 most similar (according to cosine method) for pairwise comparison
    with open(top_pairs_file, "r") as read_file:
        top_pairs = json.load(read_file)

    with open(top_pairs_similarity_file, "r") as read_file:
        top_pairs_similarity = json.load(read_file)

    if limit == None:
        limit = len(top_pairs[0])
    
    all_prob = []
    for i in top_pairs:
        first_number = int(i)
        for j in range(limit):
            second_number = top_pairs[i][j]
            prob = top_pairs_similarity[i][j]
            similarity_table[first_number][second_number] = prob
            similarity_table[second_number][first_number] = prob
            all_prob.append(prob)

    return similarity_table


def select_candidates_up_till(max_limit=100,top_pairs_files, 
                              top_pairs_similarity_files, 
                              data_version, dataset_name):
    
    all_results = []
    
    for limit in trange(1, max_limit+1):
        
        with open(top_pairs_files[0], "r") as read_file:
            top100_pairs = json.load(read_file)
        
        similarity_table = np.zeros((len(top100_pairs),len(top100_pairs)))
        
        if limit < 101:
            i = 0
            top_pairs_file = top_pairs_files[i]
            top_pairs_similarity_file = top_pairs_similarity_files[i]
            similarity_table = enter_one_top_100_into_similarity_table(top_pairs_file = top_pairs_file,
                                                    top_pairs_similarity_file = top_pairs_similarity_file, 
                                                    limit=limit, similarity_table=similarity_table)
        else:
            for i in range(0, limit//100):
                top_pairs_file = top_pairs_files[i]
                top_pairs_similarity_file = top_pairs_similarity_files[i]
                similarity_table = enter_one_top_100_into_similarity_table(top_pairs_file = top_pairs_file,
                                                        top_pairs_similarity_file = top_pairs_similarity_file, 
                                                        limit=100, similarity_table=similarity_table)
                
        sorted_np_similarity_indices = np.argsort(similarity_table, axis=1)
        correct_example_pairs_dict = generate_correct_pairs(data_version=data_version)
        adj_list = generate_adj_list(correct_example_pairs_dict)
        
        all_scores = report_all_scores(limit, adj_list=adj_list,
                                       sorted_np_similarity_indices=sorted_np_similarity_indices, 
                                       correct_example_pairs_dict=correct_example_pairs_dict)
        
        all_results += all_scores
    
    output_filename = "all_results_for_limit_1_{}_for_select_and_rerank_{}.json".format(max_limit, dataset_name)
    
    with open(data_dir + output_filename , "w") as write_file:
        json.dump(all_results, write_file)

def main(dataset_name, candidate_number):

    if dataset_name == "CEM":
        data_version = "new"
        top_pairs_files = ["new_all_not_only_0.1_top100pairs_by_cosine_similarity_{}_to_{}.json".format(start+1, start+100) for start in range(0,500,100)]
    
    elif dataset_name == "SiameseBERT":
        data_version = "new_reindexed"
        top_pairs_files = ["cosine_like_grid_siamese_proper_{}_to_{}.json".format(start+1, start+100) for start in range(0,500,100)]
    
    top_pairs_similarity_files = [data_dir + "output_" + i for i in top_pairs_files]
    select_candidates_up_till(max_limit=candidate_number,top_pairs_files, top_pairs_similarity_files, data_version, dataset_name)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run baseline models')
    parser.add_argument("encoder", default="SiameseBERT",
                        choices=['CEM', 'SiameseBERT'],
                        help='set Select and Refine model : \
                        CEM (Character Encoding Model), \
                        SiameseBERT (BERT with similarity loss function)')
    parser.add_argument("candidate_number", default="100",
                        help='number of candidate to select for refinement \
                        1 to 500 - see paper for recommended values')
    args = parser.parse_args()
    data_dir = "new/"
    main(args.encoder, int(args.encoder.candidate_number))
