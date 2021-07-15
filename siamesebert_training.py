#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForNextSentencePrediction
import torch
from tqdm import tqdm, trange
import io
import numpy as np
import json
import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertAdam
from siamesebert_model import SiameseModel

debug_mode = False
n_gpu = 4
root_dir = ''

bert_location = root_dir if len(root_dir) > 0 else '.'

single_model = BertModel.from_pretrained(bert_location)
model = SiameseModel(single_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

model.to(device)
model.train()

tokenizer = BertTokenizer.from_pretrained(bert_location, do_lower_case=True)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

def tokenize_one_text(text):
    post = "[CLS] " + text
    tokenized_text = tokenizer.tokenize(post)[:512]
    padding = [0] * (512-len(tokenized_text))
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) + padding
    segments_id = [0] * len(tokenized_text) + padding
    input_mask = [1] * len(tokenized_text) + padding
    return indexed_tokens, segments_id, input_mask



filename = "train_all_examples_only_pairs.json"


with open(root_dir + filename, "r") as read_file:
     all_posts_original = json.load(read_file)

if debug_mode:
   all_posts_original = all_posts_original[:1000]

all_post_title = []
all_post_text = []
labels = []

for i in all_posts_original:
    all_post_title.append(i["title"])
    all_post_text.append(i["text"])
    if i["similar"] == 1:
        labels.append(0)
    else:
        labels.append(1)

max_seq_len = 512

all_posts_original = None
del all_posts_original

#tokenize
input_ids = []
segment_ids = []
attention_masks = []
for i in trange(len(all_post_title)):
    indexed_tokens_0, segments_id_0, input_mask_0 = tokenize_one_text(all_post_title[i])
    indexed_tokens_1, segments_id_1, input_mask_1 = tokenize_one_text(all_post_text[i])
    input_ids.append(indexed_tokens_0 + indexed_tokens_1)
    segment_ids.append(segments_id_0 + segments_id_1)
    attention_masks.append(input_mask_0 + input_mask_1)

all_post_title = None
all_post_text = None
del all_post_title
del all_post_text

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                random_state=42, test_size=0.01)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=42, test_size=0.01)

train_segment_ids, validation_segment_ids, _, _ = train_test_split(segment_ids, input_ids,
                                                 random_state=42, test_size=0.01)

input_ids = None
segment_ids = None
attention_masks = None
del input_ids
del segment_ids
del attention_masks

#convert data to tensors

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
train_segment_ids = torch.tensor(train_segment_ids)
validation_segment_ids = torch.tensor(validation_segment_ids)


batch_size = n_gpu * 6

train_data = TensorDataset(train_inputs, train_masks,train_segment_ids, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks,validation_segment_ids, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



# Validation

print("first evaluation before training")
# Put model in evaluation mode to evaluate loss on the validation set
model.eval()

# Tracking variables
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# Evaluate data for one epoch
for batch in tqdm(validation_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_segment_id, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        logits = model(b_input_ids, token_type_ids=b_segment_id, attention_mask=b_input_mask)

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print(torch.cuda.memory_allocated()/(10**9), "GB")

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2

# trange is a tqdm wrapper around the normal python range
for epoch in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(tqdm(train_dataloader)):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_segment_id, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=b_segment_id, attention_mask=b_input_mask, next_sentence_label=b_labels)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        if step % 5000 == 0 and step != 0:
            model_fn = "{}Epoch_{}_{}_tropes_all_examples_only_siamese_proper.bin".format(root_dir,epoch, step)
            #model_fn = root_dir + "Epoch_"+ str(epoch)+ "_"+ str(step) + "_tropes_all_examples_only_siamese_proper.bin"
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_fn)
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

    print("Train loss: {}".format(tr_loss/nb_tr_steps))


    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in tqdm(validation_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_segment_id, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=b_segment_id, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    model_fn = root_dir + "first_epoch_tropes_all_examples_only_siamese_proper.bin"
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), model_fn)


#save model
model_fn = root_dir + "tropes_all_examples_only_siamese_proper.bin"
model_to_save = model.module if hasattr(model, 'module') else model
torch.save(model_to_save.state_dict(), model_fn)

#save training loss
training_loss_set = [float(i) for i in train_loss_set]

with open(root_dir + "training_losses_2_epoch_proper_siamese.json", "w") as write_file:
    json.dump(training_loss_set, write_file)
