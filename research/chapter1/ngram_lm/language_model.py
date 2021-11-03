import nltk
from collections import Counter
from functools import reduce

text = '''she sells sea-shells by the sea-shore. the shells she sells are sea-shells, I'm sure. for if she sells sea-shells by the sea-shore then i'm sure she sells sea-shore shells.'''

def preprocess(sentences, n):
    '''Split list of sentences into words and make a list of words

    Args:
        sentences (list of str): a list of sentences
        n (int): variable N for N-gram model. How many words you want to see before a target word.
    Returns:
        A list of tokens which were tokenized from each sentence
    '''

    BOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'

    # STEP1: 소문자 치환하기
    sentences = list(map(str.lower, sentences))

    # STEP2: BOS, EOS 추가하기
    BOSs = ' '.join([BOS]*(n-1) if n > 1 else [BOS])
    sentences = [' '.join([BOSs, s, EOS]) for s in sentences]

    # STEP3: 토큰화하기
    sentences = list(map(lambda s: s.split(), sentences))
    tokens = list(reduce(lambda a, b: a+b, sentences))

    # STEP4: 한번 출현한 단어 UNK으로 치환하기
    freq = nltk.FreqDist(tokens)
    tokens = [t if freq[t] > 1 else UNK for t in tokens]

    return tokens


class SimpleNgramLanguageModel():
    def __init__(self, train_data, n):
        sentences = nltk.tokenize.sent_tokenize(train_data)
        sentences = train_data.split('.')

        tokens = preprocess(sentences, n)
        self.vocab = self.build_model(tokens, n)

    def build_model(self, tokens, n):
        ngrams = nltk.ngrams(tokens, n)
        nvocab = nltk.FreqDist(ngrams)

        if n == 1:
            vocab = nltk.FreqDist(tokens)
            vocab_size = len(nvocab)
            return {v: c/vocab_size for v, c in vocab.items()}
        else:
            mgrams = nltk.ngrams(tokens, n-1)
            mvocab = nltk.FreqDist(mgrams)
            def ngram_prob(ngram, ncount):
                mgram = ngram[:-1]
                mcount = mvocab[mgram]
                return ncount / mcount
            return {v: ngram_prob(v, c) for v, c in nvocab.items()}

    def smoothing(smt_type='laplace'):
        if smt_type == 'laplace':
            return None
        elif smt_type == 'add_one':
            return None

    def generate_sentence(self):
        return 'hello'

    def build_vocab(self, data):
        vocab = {}
        for d in data:
            for k, v in Counter(d).items():
                try:
                    vocab[k] += v
                except KeyError:
                    vocab[k] = v

        return vocab


lm = SimpleNgramLanguageModel(text, n=3)
vocab = lm.vocab
for k, v in vocab.items():
    given = k[:-1]
    word = k[-1]
    print(f'P({word}|{given}): {v}')
