#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from itertools import combinations
import random
import seaborn as sns
from tqdm import tqdm


with open("all_tropes_url_to_tropes.json", "r") as read_file:
     url_to_tropes = json.load(read_file)

clean_url_to_tropes = {}
train_url_to_tropes = {}
test_url_to_tropes = {}

counter = 0
total = 0
all_length = []
all_examples = []
all_train_examples = []
all_test_examples = []
random.seed = 42
for i in tqdm(url_to_tropes):
    examples = []
    train_examples = []
    test_examples = []
    for example in url_to_tropes[i]:
        #filter out trope examples shorter than 100 words
        if len(example['text'].split(' ')) > 100: 
            examples.append(example['text'])
            if random.random() < 0.2:
                test_examples.append(example['text'])
                all_test_examples.append(example['text'])
            else:
                train_examples.append(example['text'])
                all_train_examples.append(example['text'])
                
    total += len(examples)
    all_length.append(len(examples))
    all_examples += examples
    counter += 1
    clean_url_to_tropes[i] = examples
    train_url_to_tropes[i] = train_examples
    test_url_to_tropes[i] = test_examples
    
del url_to_tropes
del clean_url_to_tropes
del all_examples

def get_example_pairs(clean_url_to_tropes, all_examples, is_train=True):
    all_examples_only_pairs = []
    counter = 0
    count = 0
    for i in clean_url_to_tropes:
        examples = clean_url_to_tropes[i]
        all_pairs = list(combinations(examples, 2))
        for pair in all_pairs:
            one_similar = {
                    "title":pair[0],
                    "text":pair[1],
                    "similar":1
                    }
            all_examples_only_pairs.append(one_similar)
            
            if is_train:
                found = False
                while not found:
                    random_sentence = random.choice(all_examples)
                    if random_sentence not in examples:
                        found =True
                one_dissimilar = {
                        "title":pair[0],
                        "text":random_sentence,
                        "similar":0
                        }
                all_examples_only_pairs.append(one_dissimilar)
        
        counter += 1
        if counter % 100 == 0:
            print(counter)
    
    if is_train:
        with open("new/train_%d_all_examples_only_pairs.json"%(count), "w") as write_file:
            json.dump(all_examples_only_pairs, write_file)
    else:
        with open("new/test_%d_all_examples_only_pairs.json"%(count), "w") as write_file:
            json.dump(all_examples_only_pairs, write_file)
    print("no. of character pairs = ", len(all_examples_only_pairs))


def get_mean_and_std_of_examples_per_trope(url_to_tropes):
    examples_per_trope = []
    non_zero = 0
    for i in url_to_tropes:
        examples_per_trope.append(len(url_to_tropes[i]))
        if len(url_to_tropes[i]) > 0:
            non_zero += 1
    print("non_zero = ", non_zero)
    print("mean = ", np.mean(examples_per_trope))
    print("std = ", np.std(examples_per_trope))
    print("no. of tropes = ", len(examples_per_trope))
    print("no. of characters = ", sum(examples_per_trope))
    sns.distplot(examples_per_trope)


get_mean_and_std_of_examples_per_trope(train_url_to_tropes)
get_example_pairs(train_url_to_tropes, all_train_examples, is_train=True)
get_mean_and_std_of_examples_per_trope(test_url_to_tropes)
get_example_pairs(test_url_to_tropes, all_test_examples, is_train=False)




    
    



