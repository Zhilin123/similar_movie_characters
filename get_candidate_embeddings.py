#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel
import json
from keras.preprocessing.sequence import pad_sequences
import argparse
from siamesebert_model import SiameseModel

def get_model(model_name):

    if model_name == "CEM":
        bin_name = root_dir + 'tropes_all_examples_only.bin'
        model_state_dict = torch.load(bin_name)
        model = BertModel.from_pretrained('bert-base-uncased', state_dict=model_state_dict)

    elif model_name == "BERT":
        model = BertModel.from_pretrained('bert-base-uncased')

    elif model_name == "SiameseBERT":

        single_model = BertModel.from_pretrained(bert_location)
        big_model = SiameseModel(single_model)
        bin_name = root_dir+ "tropes_all_examples_only_siamese_proper.bin"
        big_model_state_dict = torch.load(bin_name)
        big_model.load_state_dict(big_model_state_dict)
        model = big_model.single_model

    return model


# this allows you to get the average of all tokens in the final layer but if you want something else feel free to change this line: candidate_embedding = torch.mean(encoded_layers[11], 1)
def get_embedding(MAX_SEQ_LEN = 512, all_ids=None, all_post_text=None, output_numpy_matrix_name="somefile.npy",model_name=None):
    all_bert = []
    all_cls = []
    which_batch = 0
    while which_batch < len(all_post_text) // 1:
        with torch.no_grad():
            batch_of_64 = []
            batch_of_64_masks = []
            for i in range(1):
                actual_index = which_batch * 1 + i
                if actual_index < len(all_post_text):
                    text = all_post_text[actual_index]
                    marked_text = "[CLS] " + text + " [SEP]"  #"[CLS] " + text + " [SEP]"
                    if len(tokenizer.tokenize(marked_text)) > MAX_SEQ_LEN:
                        tokenized_text = tokenizer.tokenize(marked_text)[:MAX_SEQ_LEN-1] + [tokenizer.tokenize(marked_text)[-1]]
                    else:
                        tokenized_text = tokenizer.tokenize(marked_text)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            batch_of_64.append(indexed_tokens)
            batch_of_64_masks.append([1]*len(indexed_tokens)+[0]*(MAX_SEQ_LEN-len(indexed_tokens)))
            batch_of_64 = pad_sequences(batch_of_64, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")


            mask_tensors = torch.tensor(batch_of_64_masks).to(device)
            tokens_tensors = torch.tensor(batch_of_64).type(torch.LongTensor).to(device)

            some_input = {
                    'input_ids':tokens_tensors,
                    'attention_mask': mask_tensors
                    }

            encoded_layers, _ = model(**some_input)

            mask_tensors = None
            tokens_tensors = None
            some_input = None

            del mask_tensors
            del tokens_tensors
            del some_input

            layer_number = 11 if model_name == "SiameseBERT" else 10


            candidate_embedding = torch.mean(encoded_layers[layer_number], 1)
            # this selects the 0-th element in the 1-st dimension ie the first token
            cls10 = encoded_layers[layer_number].select(1,0)

            all_bert.append(candidate_embedding)
            all_cls.append(cls10)

            del encoded_layers



        if (len(candidate_embedding)*(which_batch+1)) % 500 == 0:
            print(len(candidate_embedding)*(which_batch+1), " / ", len(all_post_text), " finished")
            print(torch.cuda.memory_allocated()/(10**9), "GB")
        which_batch += 1
        torch.cuda.empty_cache()

    print(len(all_bert))
    concated_all_bert = torch.cat(all_bert, dim=0)
    print(concated_all_bert.shape)
    concated_np = concated_all_bert.cpu().numpy()
    print(concated_np.shape)
    concated_all_cls_np = torch.cat(all_cls, dim=0).cpu().numpy()
    print(concated_all_cls_np.shape)
    np.save(root_dir + output_numpy_matrix_name , concated_np)
    np.save(root_dir + "cls_" + output_numpy_matrix_name , concated_all_cls_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='obtain encoded embeddings')
    parser.add_argument("model_name", default="CEM",
                        choices=['CEM','BERT', 'SiameseBERT'],
                        help='set baseline model : \
                        CEM (Character Encoding Model), \
                        BERT (Bidirectional Encoder Representation from Trasnformers)\
                        SiameseBERT (BERT with similarity loss function)')
    parser.add_argument("candidates_to_encode_filename")
    parser.add_argument("output_numpy_filename")
    parser.add_argument("root_dir", default="")
    args = parser.parse_args()


    root_dir = args.root_dir
    bert_location = root_dir if len(root_dir) > 0 else '.'
    tokenizer = BertTokenizer.from_pretrained(bert_location, do_lower_case=True)
    model = get_model(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    filename1 = root_dir + args.candidates_to_encode_filename
    with open(filename1, "r") as read_file:
        all_post_text = json.load(read_file)

    get_embedding(MAX_SEQ_LEN = 256, all_post_text=all_post_text , output_numpy_matrix_name=args.output_numpy_filename, model_name=args.model_name)
