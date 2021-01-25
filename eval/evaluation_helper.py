#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import json

def generate_correct_pairs(data_version="new"):
    # load in all the sentences

    if data_version == "new_reindexed":
        filename = "new/test_output_pairs.json"
        
        with open(filename, "r") as read_file:
            correct_example_pairs = json.load(read_file)
            
        correct_example_pairs_dict = {}
        
        for i in correct_example_pairs:
            correct_example_pairs_dict[tuple(i)] = 0
        
        return correct_example_pairs_dict

    elif data_version == "new":
        filename = "new/test_all_examples_only_single.json"

    with open(filename, "r") as read_file:
        example_sentences = json.load(read_file)

    example_sentences_dict = {}

    for i in range(len(example_sentences)):
        example_sentences_dict[example_sentences[i]] = i


    if data_version == "new":
        filename1 = "new/test_all_examples_only_pairs.json"

    with open(filename1, "r") as read_file:
        example_pairs = json.load(read_file)

    correct_example_pairs = []
    for i in example_pairs:
        if i["similar"] == 1:
            item1 = example_sentences_dict[i["title"]]
            item2 = example_sentences_dict[i["text"]]
            correct_example_pairs.append([item1,item2])

    correct_example_pairs_dict = {}
    for i in correct_example_pairs:
        correct_example_pairs_dict[tuple(i)] = 0
    return correct_example_pairs_dict

def generate_adj_list(correct_example_pairs_dict):
    adj_list = defaultdict(list)
    for i in correct_example_pairs_dict:
        adj_list[i[0]].append(i[1])
        adj_list[i[1]].append(i[0])
    return adj_list

def get_most_similar(top_n=10, sorted_np_similarity_indices=None):
    most_similar = []
    count = 0
    for i in sorted_np_similarity_indices:
        own_counter = 0
        for j in range(-1,-502,-1):
            if own_counter < top_n and count != int(i[j]):
                most_similar.append([count, int(i[j])])
                # comment out below when creating the pairwise to compare
                most_similar.append([int(i[j]), count])
                own_counter += 1
        count += 1
    return most_similar

def get_ordered_most_similar(top_n=10, sorted_np_similarity_indices=None):
    most_similar = []
    for i in range(len(sorted_np_similarity_indices)):
        one_most_similar = []
        own_counter = 0
        for j in range(-1,-502,-1):
            if sorted_np_similarity_indices[i][j] != int(i) and own_counter < top_n:
                one_most_similar.append([int(i), sorted_np_similarity_indices[i][j]])
                one_most_similar.append([sorted_np_similarity_indices[i][j], int(i)])
                own_counter += 1
        most_similar.append(one_most_similar)
    return most_similar

def get_mean_reciprocal_rank(ordered_most_similar=None, adj_list=None):
    score = []
    for i in ordered_most_similar:
        reciprocal_rank = 0
        for j in range(len(i)):
            if i[j][1] in adj_list[i[j][0]] or i[j][0] in adj_list[i[j][1]]:
                reciprocal_rank = 1/ ((j // 2) + 1)
                break
        score.append(reciprocal_rank)
    #sns.distplot(score)
    return np.mean(score)

def get_ndcg(ordered_most_similar=None, adj_list=None, correct_example_pairs_dict=None):
    score = []
    number_considered = 0
    for i in ordered_most_similar:
        one_score = 0
        for j in range(len(i)):
            pair_examined = tuple(i[j])
            position = j // 2
            if pair_examined in correct_example_pairs_dict:
                one_score += 1 / np.log2(position + 1 + 1)
        total_correct = len(adj_list[number_considered])
        total_correct = min(total_correct, len(i)//2)
        ideal_score = 0
        for i in range(total_correct):
            ideal_score += 1 / np.log2(i + 1 + 1)
        if ideal_score > 0:
            score.append(one_score/ideal_score)
        number_considered += 1
    return np.mean(score)

def get_recall(most_similar=None, correct_example_pairs_dict=None):
    #all_found_pairs to compared found pairs of various algorithms --> not returned now
    all_found_pairs = []
    score = 0
    for i in most_similar:
        #for j in range(len(i)):
        all_found_pairs.append([int(i[0]),int(i[1])])
        if tuple(i) in correct_example_pairs_dict:
            score += 1
    return score / len(correct_example_pairs_dict)

def formatted_scoring(top_n=10, metric=None, sorted_np_similarity_indices=None, 
                      correct_example_pairs_dict=None, adj_list=None):

    if metric is None:
        return "Please choose a metric - recall, mrr or ndcg"

    elif metric == "recall":
        most_similar = get_most_similar(top_n=top_n, sorted_np_similarity_indices=sorted_np_similarity_indices)
        recall_score = get_recall(most_similar = most_similar, correct_example_pairs_dict=correct_example_pairs_dict)
        score = recall_score

    elif metric == "mrr":
        #for mrr top_n should be set to 100
        top_n = 100
        ordered_most_similar = get_ordered_most_similar(top_n=top_n, sorted_np_similarity_indices=sorted_np_similarity_indices)
        mrr_score = get_mean_reciprocal_rank(ordered_most_similar=ordered_most_similar, adj_list=adj_list)
        score = mrr_score

    elif metric == "ndcg":
        ordered_most_similar = get_ordered_most_similar(top_n=top_n, sorted_np_similarity_indices=sorted_np_similarity_indices)
        nDCG_score = get_ndcg(ordered_most_similar= ordered_most_similar, adj_list=adj_list, correct_example_pairs_dict=correct_example_pairs_dict)
        score = nDCG_score

    else:
        return "Please choose a metric - recall, mrr or ndcg"

    return metric + " " + "top_"+ str(top_n)+ "_score = " + '%s' % float('%.4g' % (score*100))

def report_all_scores(dataset_name, sorted_np_similarity_indices=None, 
                      correct_example_pairs_dict=None, adj_list=None):
    
    top_n_s = [1,5,10] *2 + [100]
    metrics = ["recall"] * 3 + ["ndcg"] * 3 + ["mrr"]
    all_metrics = [dataset_name]
    
    for i in range(len(top_n_s)):
        one_metric = formatted_scoring(top_n=top_n_s[i], metric=metrics[i],
                                     sorted_np_similarity_indices=sorted_np_similarity_indices, 
                                     correct_example_pairs_dict=correct_example_pairs_dict, 
                                     adj_list=adj_list)
        all_metrics.append(one_metric)

    
    printed_format = [i.split("=")[-1] for i in all_metrics]
    
    return printed_format