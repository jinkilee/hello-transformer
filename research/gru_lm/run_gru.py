import torch
import numpy as np
import nltk
from torch import nn, optim
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


text = '''
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
    she sells sea shells by the sea shore. the shells she sells are sea shells, I'm sure. for if she sells sea shells by the sea shore then i'm sure she sells sea shore shells.
'''

def preprocess(sentences, add_special_tokens=True):
    '''Split list of sentences into words and make a list of words

    Args:
        sentences (list of str): a list of sentences
    Returns:
        A list of tokens which were tokenized from each sentence
    '''

    BOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'

    # STEP1: 소문자 치환하기
    sentences = list(map(str.lower, sentences))

    # STEP2: BOS, EOS 추가하기
    if add_special_tokens:
        sentences = [' '.join([BOS, s, EOS]) for s in sentences]

    # STEP3: 토큰화하기
    sentences = list(map(lambda s: s.split(), sentences))
    return sentences

class GRULanguageModelDataset(Dataset):
    def __init__(self):
        sentence_list = nltk.tokenize.sent_tokenize(text)
        tokenized_sentences = preprocess(sentence_list)
        tokens = list(reduce(lambda a, b: a+b, tokenized_sentences))
        self.vocab = self.make_vocab(tokens)
        self.i2v = {v:k for k, v in self.vocab.items()}
        self.indice = list(map(lambda s: self.convert_tokens_to_indice(s), tokenized_sentences))

    def convert_tokens_to_indice(self, sentence):
        indice = []
        for s in sentence:
            try:
                indice.append(self.vocab[s])
            except KeyError:
                indice.append(self.vocab['<unk>'])
        return torch.tensor(indice)

    def make_vocab(self, tokens):
        vocab = {}

        vocab['<pad>'] = 0
        vocab['<s>'] = 1
        vocab['</s>'] = 2
        vocab['<unk>'] = 3
        index = 4
        for t in tokens:
            try:
                vocab[t]
                continue
            except KeyError:
                vocab[t] = index
                index += 1
        return vocab

    def __len__(self):
        return len(self.indice)

    def __getitem__(self, idx):
        return self.indice[idx]

def collate_fn(batch):
    batch = pad_sequence(batch, batch_first=True)
    return batch

dataset = GRULanguageModelDataset()
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=16)

indice = np.array([
    [1, 5, 8, 3, 3, 1, 5, 2, 0, 0],
    [1, 7, 4, 5, 8, 4, 7, 3, 7, 2],
    [1, 1, 5, 8, 0, 4, 2, 0, 0, 0],
    [1, 4, 9, 7, 8, 5, 9, 8, 2, 0]
])
indice = indice[:,1:]
inputs = indice[:,:-1]
labels = indice[:,1:]
inputs = torch.tensor(inputs)   # 5, 8, 3, 3, 1, 5, 2, 0
labels = torch.tensor(labels)   # 8, 3, 3, 1, 5, 2, 0, 0

class GRULanguageModel(nn.Module):
    def __init__(self, hidden_size=30, output_size=10):
        super(GRULanguageModel, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        '''
        Input Parameters
        - inputs: (B,1)
        - hidden: (1,B,H)

        Output returns
        - output: (B,1,O)
        - hidden: (1,B,H)

		Logging outputs
        ** inputs: torch.Size([4, 1])
        ** hidden: torch.Size([1, 4, 30])
        ** embedded: torch.Size([4, 1, 30])
        ** output: torch.Size([4, 1, 10])
        ** hidden: torch.Size([1, 4, 30])
        '''
        #print('** inputs: {}'.format(inputs.shape))
        #print('** hidden: {}'.format(hidden.shape))

        embedded = self.embedding(inputs)
        #print('** embedded: {}'.format(embedded.shape))

        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))
        #print('** output: {}'.format(output.shape))
        #print('** hidden: {}'.format(hidden.shape))

        return output, hidden


hidden_size = 30
output_size = len(dataset.vocab)
model = GRULanguageModel(hidden_size=hidden_size, output_size=output_size)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

def train(inputs, labels, model, criterion, optimizer, max_grad_norm=None):
    batch_size = inputs.size()[0]
    hidden = torch.zeros((1, batch_size, hidden_size))
    input_length = inputs.size()[1]

    loss = 0

    teacher_forcing = True if np.random.random() < 0.5 else False
    lm_inputs = inputs[:,0].unsqueeze(-1)
    for i in range(input_length):
        output, hidden = model(lm_inputs, hidden)
        output = output.squeeze(1)
        loss += criterion(output, labels[:,i])

        #print('** {} vs {}'.format(lm_inputs[0,0], labels[0,i]))
        if teacher_forcing:
            lm_inputs = labels[:,i].unsqueeze(-1)
        else:
            topv, topi = output.topk(1)
            lm_inputs = topi

    loss_before = loss
    loss.backward()
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss

def generate_sentence(text):
    tokenized_sentences = preprocess([text], add_special_tokens=False)
    indice= list(map(lambda s: dataset.convert_tokens_to_indice(s), tokenized_sentences))

    hidden = torch.zeros((1, 1, hidden_size))
    lm_inputs = torch.tensor(indice).unsqueeze(-1)

    cnt = 0
    generated_sequence = [lm_inputs[0].data.item()]
    while True:
        if cnt == 30:
            break
        output, hidden = model(lm_inputs, hidden)
        output = output.squeeze(1)
        topv, topi = output.topk(1)
        lm_inputs = topi

        if topi.data.item() == dataset.vocab['</s>']:
            print('</s> was generated. therefore finished at {}th iteration'.format(cnt))
            tokens = list(map(lambda w: dataset.i2v[w], generated_sequence))
            return ' ' .join(tokens)

        generated_sequence.append(topi.data.item())
        cnt += 1

    print('max iteration reached. therefore finishing forcefully')
    tokens = list(map(lambda w: dataset.i2v[w], generated_sequence))
    
    return ' '.join(tokens)

# Train
for i in range(500):
    for batch in dataloader:
        batch = batch[:,1:]
        inputs = batch[:,:-1]
        labels = batch[:,1:]

        loss = train(inputs, labels, model, criterion, optimizer, max_grad_norm=5.0)
        print('{} - loss: {:.4f}'.format(i, loss))


print('-----------------------------------------')

# Generate a text
generated_text = generate_sentence('she')
print(len(generated_text), generated_text)
sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
print('sum of weight: {:.4f}'.format(sum_of_weight))
