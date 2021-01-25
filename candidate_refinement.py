#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForNextSentencePrediction
import torch
from tqdm import tqdm
import json
import argparse



def get_likelihood(post_title=None, post_text=None):
    post_title = "[CLS] " + post_title + " [SEP] "
    post_text = post_text + " [SEP]"
    max_seq_len = 512
    tokenized_text_title = tokenizer.tokenize(post_title)
    if len(tokenized_text_title) > 256: #restrict the title to 256 tokens including [CLS] and [SEP]
        tokenized_text_title = tokenized_text_title[:255] + [tokenized_text_title[-1]]
    tokenized_text_text = tokenizer.tokenize(post_text)
    if len(tokenized_text_text) > 512 - len(tokenized_text_title):
        tokenized_text_text = tokenized_text_text[:512 - len(tokenized_text_title) - 1] + [tokenized_text_text[-1]]
    tokenized_text = tokenized_text_title + tokenized_text_text
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text_title) + [1] * len(tokenized_text_text)

    padding = [0] * (max_seq_len - len(indexed_tokens))
    # removing the padding tokens actually increases the speed substantially without affecting the output vectors (by using the input mask perimeter) since the number of calculations is O(max_seq_len**2)
    padding = []
    indexed_tokens += padding
    segments_ids += padding
    input_mask = [1] * len(tokenized_text) + padding

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    mask_tensors = torch.tensor([input_mask]).to(device)

    # Predict if Next candidate
    some_input = {
            'input_ids':tokens_tensor,
            'token_type_ids': segments_tensors,
            'attention_mask': mask_tensors
    }
    predictions = model(**some_input)
    softmax_predictions = torch.nn.Softmax()(predictions)
    #print(predictions)
    prob = float(softmax_predictions[0][0])
    return prob

def get_and_save_likelihood_for_all_pairs(all_post_text, top_100_pairs, output_filename):
    resulting_dictionary = {}

    counter = 0

    for i in tqdm(top_100_pairs):
        resulting_dictionary[i] = []
        selected_index = int(i)
        post_title = all_post_text[selected_index]
        for j in top_100_pairs[i]:
            post_text = all_post_text[j]
            likelihood = get_likelihood(post_title=post_title, post_text=post_text)
            resulting_dictionary[i].append(likelihood)
        counter += 1
        if counter % 5000 == 0 and counter > 0:
            with open(str(counter) + "_"+ output_filename, "w") as write_file:
                json.dump(resulting_dictionary, write_file)

    with open(output_filename, "w") as write_file:
        json.dump(resulting_dictionary, write_file)

def load_files(candidates_filename, n_plus_1_to_n_plus_100_filename):
    with open(candidates_filename, "r") as read_file:
         all_post_text = json.load(read_file)

    with open(n_plus_1_to_n_plus_100_filename, "r") as read_file:
         top_100_pairs = json.load(read_file)

    bin_name = "tropes_all_examples_only.bin"
    model_state_dict = torch.load(bin_name)
    model = BertForNextcandidatePrediction.from_pretrained('.', state_dict=model_state_dict)
    tokenizer = BertTokenizer.from_pretrained('.', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return all_post_text, top_100_pairs, tokenizer, model, device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='refine candidates selected')

    parser.add_argument("n_plus_1_to_n_plus_100_filename", default="cosine_like_grid_siamese_proper_1_to_100.json")
    parser.add_argument("candidates_filename", default="test_all_examples_only_single.json")
    parser.add_argument("root_dir", default="")
    args = parser.parse_args()

    all_post_text, top_100_pairs, tokenizer, model, device = load_files(args.candidates_filename, args.n_plus_1_to_n_plus_100_filename)
    output_filename = "output_"+ args.n_plus_1_to_n_plus_100_filename
    get_and_save_likelihood_for_all_pairs(all_post_text, top_100_pairs, output_filename)
