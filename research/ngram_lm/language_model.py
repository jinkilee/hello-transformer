from collections import Counter

text = '''After four years of refusing to hold Donald Trump accountable for his lies, conspiracy theories and hateful rhetoric, Republicans passed up another chance to purge those forces from their ranks Thursday when they overwhelmingly opposed Democrats' efforts to rebuke Georgia Rep. Marjorie Taylor Greene. The GOP is complaining that Democratic leaders are not only overreaching but also setting a dangerous precedent in both punishing Greene and pursuing a doomed-to-fail impeachment trial for an ex-President in the Senate next week. But genuinely unprecedented events have forced Democrats to take action. Despite national outrage about Trump's undemocratic actions, only 10 House Republicans voted to impeach him last month. And most Republicans balked Thursday at punishing Greene for espousing the dangerous lies and violent rhetoric that threaten the future of their party, with only 11 House Republicans joining Democrats in voting to kick Greene off her committees. Before being elected to Congress from Georgia, Greene compiled a long list of unhinged comments and social media posts, including endorsement of violence against and assassinations of top Democrats, 9/11 trutherism and denials of school shootings. For weeks, she was unrepentant -- at least until her fate was sealed Thursday, when she showed shades of contrition on the House floor -- though her combative speech adopted many of the outlandish tropes of Trumpism.'''

def preprocess(text):
    text = text.lower()
    text = text.split(' ')
    return text

class SimpleNgramLanguageModel():
    def __init__(self, train_data):
        #sentences = nltk.tokenize.sent_tokenize(train_data)
        sentences = train_data.split('.')

        self.vocab = self.build_vocab(list(map(preprocess, sentences)))

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


SimpleNgramLanguageModel(text)
