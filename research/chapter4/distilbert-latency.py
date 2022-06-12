import torch
from tqdm import tqdm
import time
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import BertModel, BertTokenizer

dbtokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbmodel = DistilBertModel.from_pretrained("distilbert-base-uncased", num_labels = 2).cuda()
btokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
bmodel = BertModel.from_pretrained("distilbert-base-uncased", num_labels = 2).cuda()

input_ids = np.random.randint(0, len(btokenizer), (1, 512))
attention_mask = np.ones_like(input_ids)
input_ids = torch.from_numpy(input_ids)
attention_mask = torch.from_numpy(attention_mask)
input_ids = input_ids.cuda()
attention_mask = attention_mask.cuda()

inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
}

def get_latency(model, inputs):
    start = time.time()
    for _ in tqdm(range(100)):
        output = model(**inputs)
        #output = bbmodel(**encoded_input)
    end = time.time()
    #print(f'latency: {(end - start)/100}')
    return (end - start) / 100

latency = get_latency(bmodel, inputs)
print(f'BERT latency={latency:.4f}')
latency = get_latency(dbmodel, inputs)
print(f'DistilBERT latency={latency:.4f}')
