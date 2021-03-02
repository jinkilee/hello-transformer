import torch
import nltk
from functools import reduce
from torch.utils.data import Dataset

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
    '''
    GRU 언어 모델을 위한 Dataset 클래스.

    Args:
        text: 전체 말뭉치 데이터셋
    Returns:
        토큰화된 text를 텐서 객체로 변환하는 Dataset 클래스
    
    Example:
        >>> from dataset import GRULanguageModelDataset
        >>> text = 'she sells sea shells by the sea shore'
        >>> dataset = GRULanguageModelDataset(text)
        >>> for d in dataset:
        ...    print(d)
        ...    break
        ...
        tensor([ 1,  4,  5,  6,  7,  8,  9,  6, 10,  2])
        >>> dataset.vocab
        {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, 'she': 4, 'sells': 5, 'sea': 6, 'shells': 7, 'by': 8, 'the': 9, 'shore': 10}
    '''
    def __init__(self, text):
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

