from transformers import BigBirdModel, BertModel
from transformers import BigBirdTokenizer, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import time
import numpy as np
import torch

bigbird = 'google/bigbird-roberta-base'
bert = 'roberta-base'

bbtokenizer = BigBirdTokenizer.from_pretrained(bigbird)
bbmodel = BigBirdModel.from_pretrained(bigbird)

bttokenizer = RobertaTokenizer.from_pretrained(bert)
btmodel = RobertaModel.from_pretrained(bert)

use_bigbird = True

if use_bigbird:
    tokenizer = bbtokenizer
    model = bbmodel
else:
    tokenizer = bttokenizer
    model = btmodel

def get_latency(model, inputs):
    start = time.time()
    for _ in tqdm(range(100)):
        output = model(**inputs)
        #output = bbmodel(**encoded_input)
    end = time.time()
    #print(f'latency: {(end - start)/100}')
    return (end - start) / 100


def make_inputs(tokenizer, length):
    input_ids = np.random.randint(0, len(tokenizer), (1, length))
    attention_mask = np.ones_like(input_ids)
    input_ids = torch.from_numpy(input_ids)
    attention_mask = torch.from_numpy(attention_mask)

    encoded_input = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    return encoded_input


for i in range(5, 10):
    inputs = make_inputs(tokenizer, 2 ** i)
    latency = get_latency(model, inputs)
    print(f'latency for token_length={2 ** i}: {latency:.4f}')





