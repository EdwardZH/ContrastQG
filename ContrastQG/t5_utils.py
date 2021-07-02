import os
import re
import torch
import json
import logging
from tqdm import tqdm
from contrastqg import T5Tokenizer
        
logger = logging.getLogger()

T5_MAX_LEN = 512


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class T5_Tokenizer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_dir)
        self.special_tokens = ['<|NEG|>', '<|POS|>']
        [self.neg_token_id, self.pos_token_id] = self.tokenizer.convert_tokens_to_ids(self.special_tokens)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def convert_doc_to_ids(self, pos_doc, max_length, neg_doc=None):
        pos_doc_ids = self.tokenizer.encode(pos_doc, max_length=max_length, truncation=True)
        if neg_doc:
            neg_doc_ids = self.tokenizer.encode(neg_doc, max_length=max_length, truncation=True)
            input_ids = [self.pos_token_id] + pos_doc_ids + [self.neg_token_id] + neg_doc_ids + [self.eos_token_id]
        else:
            pos_doc_ids = self.tokenizer.encode(pos_doc, max_length=max_length, truncation=True)
            input_ids = [self.pos_token_id] + pos_doc_ids + [self.eos_token_id]

        return input_ids
        
    
    def convert_outputs_to_tokens(self, outputs):
        batch_text = self.tokenizer.batch_decode(outputs)
        return batch_text
    
    def __len__(self):
        return len(self.tokenizer)


## ----------------------------------------------------------------------
## ----------------------------------------------------------------------


def t5_converter(
    args,
    pos_doc,
    tokenizer,
    neg_doc=None
):
    if neg_doc:
        max_doc_len = (args.max_input_len - 3) // 2
        input_ids = tokenizer.convert_doc_to_ids(
            pos_doc=pos_doc,
            neg_doc=neg_doc,
            max_length=max_doc_len
        )
    else:
        max_doc_len = (args.max_input_len - 2)
        input_ids = tokenizer.convert_doc_to_ids(
            pos_doc=pos_doc,
            max_length=max_doc_len
        )
    return input_ids





