import os
import logging
import torch
from torch.utils.data import Dataset
import utils
import numpy as np
import random
from t5_utils import t5_converter
logger = logging.getLogger()



class QGDataLoader(Dataset):
    def __init__(
        self, 
        args,
        tokenizer, 
    ):
        """
        :param intput_dir: examples.jsonl ("pos_docid"/"neg_docid"); docid2doc.jsonl
        :param tokenizer: T5Tokenizer or None
        """
        self.corpus = utils.load_json2dict(os.path.join(args.input_dir, "corpus.jsonl"))
        self.pos_doc_ids = None
        self.neg_doc_ids = None
        if args.generator_mode == "qg":
            #doc_ids = list(self.corpus.keys())
            #np.random.shuffle(doc_ids)
            #self.pos_doc_ids = doc_ids[:args.max_qg_doc]
            examples = utils.load_json2list(os.path.join(args.output_dir, "docs.jsonl"))
            self.pos_doc_ids = list()
            for example in examples:
                self.pos_doc_ids.append(example["pos_doc_id"])
        elif args.generator_mode == "cqg":
            examples = utils.load_json2list(os.path.join(args.output_dir, "contrast_pairs.jsonl"))
            np.random.shuffle(examples)
            examples = examples[:args.max_qg_doc]
            self.pos_doc_ids = list()
            self.neg_doc_ids = list()
            for example in examples:
                self.pos_doc_ids.append(example["pos_doc_id"])
                self.neg_doc_ids.append(example["neg_doc_id"])

        else:
            raise ("Not implement!")
        logger.info('[%s] needs generate %d examples'%(args.generator_mode, len(self.pos_doc_ids)))

        self.args = args
        self.tokenizer = tokenizer
                
    def __len__(self):
        return len(self.pos_doc_ids)


    def __getitem__(self, index):
        neg_doc = None
        pos_doc_id = self.pos_doc_ids[index]
        pos_doc = self.corpus[pos_doc_id]["title"] + " " + self.corpus[pos_doc_id]["text"]
        neg_doc_id = None
        if self.neg_doc_ids:
            neg_doc_id = self.neg_doc_ids[index]
            neg_doc = self.corpus[neg_doc_id]["title"] + " " + self.corpus[neg_doc_id]["text"]
        input_ids = t5_converter(self.args, pos_doc, self.tokenizer, neg_doc=neg_doc)
        return {"input_ids": input_ids, "neg_doc_id": neg_doc_id, "pos_doc_id": pos_doc_id}

    def collate_fn(self, batch_data):
        input_ids = [data["input_ids"] for data in batch_data]
        input_ids = [input_id + [self.tokenizer.pad_token_id] * (self.args.max_input_len - len(input_id))
                     for input_id in input_ids]
        neg_doc_ids = [data["neg_doc_id"] for data in batch_data]
        pos_doc_ids = [data["pos_doc_id"] for data in batch_data]
        input_ids = torch.LongTensor(input_ids)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids.cuda(), "attention_mask": attention_mask.cuda(), "pos_doc_ids": pos_doc_ids, "neg_doc_ids": neg_doc_ids}
