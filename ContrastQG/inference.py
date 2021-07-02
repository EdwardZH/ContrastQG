import os
import sys
import time
import tqdm
import json
import torch
import random
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

sys.path.append("..")
import utils
import config
from data_loader import QGDataLoader
from model import QGenerator
from t5_utils import T5_Tokenizer
torch.backends.cudnn.benchmark=True


logger = logging.getLogger()


def do_inference(args, generate_loader, generator):
    gen_examples = []
    counter = 0
    for batch in tqdm(generate_loader):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "max_length": args.max_gen_len,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_return_sequences": args.num_return_sequences,
            "do_sample": args.do_sample
        }

        outputs = generator.predict(inputs)
        pos_doc_ids = batch["pos_doc_ids"]
        neg_doc_ids = batch["neg_doc_ids"]
        assert len(pos_doc_ids) == len(neg_doc_ids)
        assert len(outputs) == args.num_return_sequences * len(pos_doc_ids)
        for step, (pos_doc_id, neg_doc_id) in enumerate(zip(pos_doc_ids, neg_doc_ids)):
            for query in outputs[step * args.num_return_sequences: (step + 1) * args.num_return_sequences]:
                counter += 1
                qid = "%s_%s_%d"%(
                    args.generator_mode,
                    args.pretrain_generator_type,
                    counter
                )
                if args.generator_mode == "cqg":
                    gen_examples.append({
                        "_id":qid,
                        "text": query,
                        "pos_doc_id":pos_doc_id,
                        "neg_doc_id":neg_doc_id,
                    })
                else:
                    gen_examples.append({
                        "_id":qid,
                        "text": query,
                        "pos_doc_id": pos_doc_id,
                    })
    return gen_examples


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # setting args
    parser = argparse.ArgumentParser(
        'ContrastQG', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    config.add_default_args(parser)
    args = parser.parse_args()
    config.init_args_config(args)

    save_folder = os.path.join(args.output_dir, "%s_%s"%(args.generator_mode, args.pretrain_generator_type), args.task)
    utils.create_folder_fct(save_folder)
    args.output_dir = save_folder
    args.input_dir = os.path.join(args.input_dir, args.task)
    # random seed
    utils.set_seed(args)

    ## **********************************************
    # load tokenizer
    tokenizer = T5_Tokenizer(args)
    generate_dataset = QGDataLoader(
        args=args,
        tokenizer=tokenizer,
    )
    logger.info("generation batch size = {}".format(args.batch_size))

    gen_data_loader = torch.utils.data.DataLoader(
        generate_dataset,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=args.data_workers,
        collate_fn=generate_dataset.collate_fn
    )
    generator = QGenerator(args, tokenizer)
    generator = generator.cuda()
    ## ***********************
    # [4] Generator Inference
    gen_examples = do_inference(**{
        "args":args, 
        "generate_loader":gen_data_loader,
        "generator":generator})
    
    ## ***********************
    # [5] Save files


    utils.save_list2jsonl(
        data_list=gen_examples, 
        save_filename=os.path.join(args.output_dir, "queries.jsonl")
    )

    if args.save_txt:
        utils.save_list2txt(data_list=gen_examples,
        save_filename=os.path.join(args.output_dir, "queries.txt")
        )
        utils.save_bm25(data_dict=generate_dataset.corpus,
        save_dir=args.output_dir
        )
