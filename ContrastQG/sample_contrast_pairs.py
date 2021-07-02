import os
import sys
import json
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from utils import *

def add_default_args(parser):
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help="Input path of orignal dataset path."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Input path of orignal dataset path."
    )
    parser.add_argument(
        '--topk', 
        type=int,
        default=100,
        help="Number of retrieved depth."
    )
    parser.add_argument(
        '--sample_n', 
        type=int,
        default=5,
        help="Number of doc pairs per query."
    )
    parser.add_argument(
        '--max_qg',
        type=int,
        default=100000,
        help="Number generated queries."
    )
    


def sample_contast_pairs(results, sample_n, max_qg):
    all_pairs = set()
    pairs = list()
    results = list(results.items())
    for qid, doc_list in results:
        doc_pairs = list()
        for i in range(len(doc_list)):
            for j in range(len(doc_list)):
                if i != j:
                    doc_pairs.append((doc_list[i], doc_list[j]))
        np.random.shuffle(doc_pairs)
        counter = 0
        for pair in doc_pairs:
            if counter >= sample_n:
                break
            if pair not in all_pairs:
                all_pairs.add(pair)
                counter += 1
    all_pairs = list(set(all_pairs))
    np.random.shuffle(all_pairs)
    all_pairs = all_pairs[:max_qg]
    for pair in all_pairs:
        pairs.append({"qid": qid, "pos_doc_id": pair[0], "neg_doc_id": pair[1]})
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'sample_contrast_pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_default_args(parser)
    args = parser.parse_args()
    
    # load retrieval files
    bm25_filename = os.path.join(args.input_path, "bm25.trec")
    results = load_trec(bm25_filename, topk=args.topk)
    print (len(results))
    # sample pairs
    sample_pairs = sample_contast_pairs(results, sample_n=args.sample_n, max_qg=args.max_qg)
    
    # save pairs
    create_folder_fct(args.output_path)
    save_list2jsonl(sample_pairs, os.path.join(args.output_path, "contrast_pairs.jsonl"))