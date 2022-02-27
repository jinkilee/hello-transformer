import torch
import numpy as np
import nltk
import pickle
import argparse
from torch import nn, optim
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import GRULanguageModel
from dataset import GRULanguageModelDataset

nltk.download('punkt')

def collate_fn(batch):
    batch = pad_sequence(batch, batch_first=True)
    return batch


def train(inputs, labels, model, criterion, optimizer, max_grad_norm=None):
    '''
        Input Parameters
        - inputs: (B,M)
        - labels: (B,M)

        Output returns
        - loss: calculated loss for one batch tensor
        Example
            >>> from torch import nn, optim
            >>> from dataset import GRULanguageModelDataset
            >>> from run_gru import GRULanguageModel, text
            >>> hidden_size = 30
            >>> dataset = GRULanguageModelDataset(text)
            >>> output_size = len(dataset.vocab)
            >>> model = GRULanguageModel(hidden_size=hidden_size, output_size=output_size)
            >>> criterion = nn.NLLLoss()
            >>> optimizer = optim.SGD(model.parameters(), lr=0.005)
            >>> inputs = dataset[0][:-1].unsqueeze(0)
            >>> labels = dataset[0][1:].unsqueeze(0)
            >>> loss = train(inputs, labels, model, criterion, optimizer, max_grad_norm=5.0)
            >>> loss
            tensor(27.3188, grad_fn=<AddBackward0>)
    '''
    hidden_size = model.hidden_size
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

    loss.backward()
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, help='hidden size of GRULanguageModel')
    args = parser.parse_args()

    with open('input_data.txt', 'r') as f:
        text = f.readline().rstrip()

    # define dataset and dataloader
    dataset = GRULanguageModelDataset(text)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=16)

    # save vocab
    pickle.dump(dataset.vocab, open('vocab.pickle', 'wb'))

    # define model, criterion, optimizer
    hidden_size = args.hidden_size
    output_size = len(dataset.vocab)

    model = GRULanguageModel(hidden_size=hidden_size, output_size=output_size)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # Train
    for i in range(1000):
        for batch in dataloader:
            inputs = batch[:,:-1]
            labels = batch[:,1:]

            loss = train(inputs, labels, model, criterion, optimizer, max_grad_norm=5.0)
            print('{}th iteration -> loss={:.4f}'.format(i, loss))

    print('-----------------------------------------')

    # save model
    torch.save(model.state_dict(), 'gru_model.bin')


if __name__ == '__main__':
    main()

