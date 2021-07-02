import os
import csv
import time
import json
import torch
import random
import argparse
import logging
import traceback
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import pytrec_eval

logger = logging.getLogger()


# ------------------------------------------------------------
# ------------------------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    



## ------------------------------------------------------------
## ------------------------------------------------------------
## Save files

def create_folder_fct(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def save_list2jsonl(data_list, save_filename):
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        for data in data_list:
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()
        
def save_dict2jsonl(data_dict, save_filename, id_name, text_key):
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        for key in data_dict:
            data = {id_name:key, text_key:data_dict[key]}
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()
        

def save_list2txt(data_list, save_filename):
    with open(save_filename, mode="w", encoding="utf-8") as fw:
        for data in data_list:
            fw.write(data["_id"] + "\t" + data["text"] + "\n")
        fw.close()


def save_bm25(data_dict, save_dir):
    save_dir = os.path.join(save_dir, "corpus")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, "corpus.jsonl"), mode="w", encoding="utf-8") as fw:
        for _, data in data_dict.items():
            fw.write("{}\n".format(json.dumps({"id": data["_id"], "contents": data["title"] + " " + data["text"]})))
        fw.close()


def load_json2list(file_path):
    """used in load_dataset."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for line in tqdm(fi):
            data = json.loads(line)
            data_list.append(data)
    return data_list


def load_json2dict(file_path):
    """used in load_dataset."""
    data_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for line in tqdm(fi):
            data = json.loads(line)
            data_dict[data["_id"]] = data
    return data_dict


def load_trec(input_file, topk):
    """
    Convert base retrieval scores to qid2docids & qid2docid2scores.
    """
    results = {}
    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip('\n').split(' ')
            if len(line) == 6:
                qid, _, docid, rank, score, _ = line
                if qid not in results:
                    results[qid] = list()
                results[qid].append(docid)
    for qid in list(results.keys()):
        results[qid] = results[qid][:topk]
    return results
