# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example Usage: 
python run_cybert_training.py \
       --training-data /opt/nvidia/clx_streamz/data/apache_raw_sample_1k.csv \
       --model-dir /opt/nvidia/clx_streamz/ml/cybert/models \
       --output-dir trained_models \
       --vocab-filepath /rapids/clx/analytics/resources/bert-base-cased-hash.txt
"""
import cudf
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.dlpack import from_dlpack
import torch.nn.functional as F
from seqeval.metrics import classification_report,accuracy_score,f1_score
from transformers import BertForTokenClassification
from tqdm import tqdm,trange
from collections import defaultdict
import random
import argparse
import pandas as pd
import numpy as np
import cupy


def data_preprocessing(training_data):
    #loading csv with header
    logs_df = cudf.read_csv(training_data)
    logs_df['raw_preprocess'] = logs_df.raw.str.replace('"','')
    
    # column names to use as lables
    cols = logs_df.columns.values.tolist()
    
    # do not use raw columns as labels
    cols.remove('raw')
    cols.remove('raw_preprocess')
    
    # using for loop for labeling funcition until string UDF capability in rapids- it is currently slow
    labels = []
    texts = []
    for indx in range(len(logs_df)):
        row_labels, row_raw = raw_labeler(logs_df, indx, cols)
        labels.append(row_labels)
        texts.append(row_raw)
    subword_labels = subword_labeler(texts, labels)
    del logs_df
    
    # create set of labels
    label_values = list(set(x for l in labels for x in l))
    label_values[:0] = ['[PAD]']  
    label_values.append('X')
    
    # create a dict for mapping id to label name
    label2idx = {t: i for i, t in enumerate(label_values)}
    idx2label = {v: k for k, v in label2idx.items()}
    padded_labels = [pad(x[:256], '[PAD]', 256) for x in subword_labels]
    int_labels = [[label2idx.get(l) for l in lab] for lab in padded_labels]
    label_tensor = torch.tensor(int_labels).to('cuda')
    del subword_labels    
    
    input_ids, attention_masks = bert_cased_tokenizer(cudf.Series(texts))   
    # create dataset
    
    dataset = TensorDataset(input_ids, attention_masks, label_tensor)
    
    # use pytorch random_split to create training and validation data subsets
    dataset_size = len(input_ids)
    training_dataset, validation_dataset = random_split(dataset, (int(dataset_size*.8), int(dataset_size*.2)))
    
    # create dataloader
    train_dataloader = DataLoader(dataset=training_dataset, shuffle=True, batch_size=32)
    val_dataloader = DataLoader(dataset=validation_dataset, shuffle=False, batch_size=1)
    print("Data Preprocessing Finished")         
    return train_dataloader, val_dataloader, idx2label
    
def train_model(model_dir, train_dataloader):
    model = BertForTokenClassification.from_pretrained(model_dir)
    # model to gpu
    model.cuda();
    FULL_FINETUNING = True
    if FULL_FINETUNING:
    #fine tune all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
    ]
    else:
        # only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    
    # using 3 epochs to avoid overfitting
    epochs = 3
    max_grad_norm = 1.0

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss, scores = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps)) 
    return model

def save_model(model, idx2label, output_dir):
    model.config.id2label.update(idx2label)
    model.config.label2id = ({v: k for k, v in model.config.id2label.items()})
    model.save_pretrained(output_dir)
    

def model_eval(model, val_dataloader, idx2label):

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    for step, batch in enumerate(val_dataloader):
        input_ids, input_mask, label_ids = batch
        
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None,
            attention_mask=input_mask,)
        
            # For eval mode, the first result of outputs is logits
            logits = outputs[0] 
        
        # Get NER predicted result
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
    
        # Get NER true result
        label_ids = label_ids.detach().cpu().numpy()
    
        # Only predict the groud truth, mask=0, will not calculate
        input_mask = input_mask.detach().cpu().numpy()
    
        # Compare the valuable predict result
        for i,mask in enumerate(input_mask):
            # ground truth 
            temp_1 = []
            # Prediction
            temp_2 = []
        
            for j, m in enumerate(mask):
                # Mask=0 is PAD, do not compare
                if m: # Exclude the X label
                    if idx2label[label_ids[i][j]] != "X" and idx2label[label_ids[i][j]] != "[PAD]": 
                        temp_1.append(idx2label[label_ids[i][j]])
                        temp_2.append(idx2label[logits[i][j]])
                else:
                    break      
            y_true.append(temp_1)
            y_pred.append(temp_2)

    print("f1 score: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

    # Get acc , recall, F1 result report
    print(classification_report(y_true, y_pred,digits=4))
          
def raw_labeler(df, index_no, cols):
    """
    sample log and label the sequences with the column name from the parsed log
    """
    raw_split = df.raw_preprocess[index_no].split()
    # if log is less than 100 whitespace words keep whole thing, otherwise random sample
    len_raw = len(raw_split)
    if len_raw > 100:
        random_start = random.randint(0, len_raw-20)
        sample_split = raw_split[random_start:random_start+200]
        sampled_raw = (" ").join(sample_split)
    else: 
        sampled_raw = df.raw_preprocess[index_no]
        sample_split = raw_split

    # words in raw but not in parsed logs labeled as 'other'
    label_list = ['other'] * len(sample_split) 
    
    # for each parsed column find the location of the sequence of words (sublist) in the raw log
    for col in cols:
        if str(df[col][index_no]) not in {'','-','None','NaN'}:
            sublist = str(df[col][index_no]).split()
            sublist_len=len(sublist)
            match_count = 0
            for ind in (i for i,el in enumerate(sample_split) if el==sublist[0]):
                # words in raw log not present in the parsed log will be labeled with 'other'
                if ((match_count < 1) and (sample_split[ind:ind+sublist_len]==sublist) and 
                    (label_list[ind:ind+sublist_len] == ['other'] * sublist_len)):
                    label_list[ind:ind+sublist_len] = [col] * sublist_len
                    match_count = 1
    return label_list, sampled_raw
    
def subword_labeler(log_list, label_list):
    """
    label all subword pieces in tokenized log with an 'X'
    """
    subword_labels = []
    for log, tags in zip(log_list,label_list):
        temp_tags = []
        words = cudf.Series(log.split())
        words_size = len(words)
        subword_counts = words.str.subword_tokenize(args.vocab_filepath, 10000, 10000,
                                                    max_num_strings=words_size,max_num_chars=10000,
                                                    max_rows_tensor=words_size,
                                                    do_lower=False, do_truncate=False)[2].reshape(words_size, 3)[:,2]
        for i, tag in enumerate(tags):
            temp_tags.append(tag)
            temp_tags.extend('X'* subword_counts[i].item())
        subword_labels.append(temp_tags)
    return subword_labels

def bert_cased_tokenizer(strings):
    """
    converts cudf.Seires of strings to two torch tensors- token ids and attention mask with padding
    """    
    num_strings = len(strings)
    num_bytes = strings.str.byte_count().sum()
    token_ids, mask = strings.str.subword_tokenize(args.vocab_filepath, 256, 256,
                                                            max_num_strings=num_strings,
                                                            max_num_chars=num_bytes,
                                                            max_rows_tensor=num_strings,
                                                            do_lower=False, do_truncate=True)[:2]
    # convert from cupy to torch tensor using dlpack
    input_ids = from_dlpack(token_ids.reshape(num_strings,256).astype(cupy.float).toDlpack())
    attention_mask = from_dlpack(mask.reshape(num_strings,256).astype(cupy.float).toDlpack())
    return input_ids.type(torch.long), attention_mask.type(torch.long)

def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l


def main():
    print("Data Labeling and Preprocessing...")
    train_dataloader, val_dataloader, idx2label = data_preprocessing(args.training_data)
    print("Model Training...")
    model = train_model(args.model_dir, train_dataloader)
    save_model(model, idx2label, args.output_dir)
    print("Model Evaluation...")
    model_eval(model, val_dataloader, idx2label)
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--training-data", required=True,
                      help="CSV with raw and parsed fields")
  parser.add_argument("--model-dir", required=True,
                      help="Directory with base model files")
  parser.add_argument("--output-dir", required=True,
                      help="output directory to save new model files")
  parser.add_argument("--vocab-filepath", required=True,
                      help="vocab file path")
  args = parser.parse_args()

  main()