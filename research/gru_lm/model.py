from torch import nn

class GRULanguageModel(nn.Module):
    def __init__(self, hidden_size=30, output_size=10):
        super(GRULanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
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

        Example
            >>> import torch
            >>> from dataset import GRULanguageModelDataset
            >>> from run_gru import GRULanguageModel, text
            >>> dataset = GRULanguageModelDataset(text)
            >>> hidden_size = 30
            >>> output_size = len(dataset.vocab)
            >>> hidden = torch.zeros((1, 1, hidden_size))
            >>> inputs = dataset[0].unsqueeze(0)
            >>> model = GRULanguageModel(hidden_size=hidden_size, output_size=output_size)
            >>> out = model(inputs, hidden)
            >>> hidden = torch.zeros((1, 1, hidden_size))
            >>> inputs = dataset[0].unsqueeze(0)
            >>> out = model(inputs, hidden)
            >>> out[0].shape, out[1].shape
            (torch.Size([1, 10, 21]), torch.Size([1, 1, 30]))
        '''

        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output))

        return output, hidden


