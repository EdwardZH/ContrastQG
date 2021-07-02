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
        '--task',
        type=str,
        required=True,
        help="The task."
    )
    parser.add_argument(
        '--max_qg',
        type=int,
        default=100000,
        help="Number generated queries."
    )


def sample_docs(results, args):
    doc_ids = list(results.keys())
    np.random.shuffle(doc_ids)
    doc_ids = doc_ids[:args.max_qg]
    pairs = [{"pos_doc_id": doc_id} for doc_id in doc_ids]
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'sample_pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_default_args(parser)
    args = parser.parse_args()

    # load retrieval files
    corpus_filename = os.path.join(args.input_path, args.task, "corpus.jsonl")
    results = load_json2dict(corpus_filename)
    print (len(results))
    # sample pairs
    sample_pairs = sample_docs(results, args)

    # save pairs
    create_folder_fct(os.path.join(args.output_path, "qg_t5-base", args.task))
    save_list2jsonl(sample_pairs, os.path.join(args.output_path, "qg_t5-base", args.task, "docs.jsonl"))