import os
import math
import torch
from torch import nn, optim
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from contrastqg import (T5ForConditionalGeneration)

logger = logging.getLogger()

class QGenerator(nn.Module):
    def __init__(self, args, tokenizer):
        super(QGenerator, self).__init__()
        self.network = T5ForConditionalGeneration.from_pretrained(args.pretrain_generator_type)
        self.network.resize_token_embeddings(len(tokenizer))
        self.network.load_state_dict(torch.load(args.model_dir + '/pytorch_model.bin'))
        logger.info("Load checkpoint from {} !".format(args.model_dir))
        self.tokenizer = tokenizer
        self.args = args


    def predict(self, inputs):
        self.network.cuda().eval()
        outputs = self.network.generate(**inputs)
        pred_tokens = self.tokenizer.convert_outputs_to_tokens(outputs)
        return pred_tokens
