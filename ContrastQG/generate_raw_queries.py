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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'sample_contrast_pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_default_args(parser)
    args = parser.parse_args()

    # load retrieval files
    query_path = os.path.join(args.input_path, "queries.jsonl")
    out_path = os.path.join(args.output_path, "queries.txt")
    queries = load_json2list(query_path)
    print (len(queries))

    # save pairs
    create_folder_fct(args.output_path)
    save_list2txt(queries, os.path.join(args.output_path, "queries.txt"))