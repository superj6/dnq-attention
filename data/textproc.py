from io import open
import unicodedata
import re
import os

SPLIT_TYPE = 'word'
match SPLIT_TYPE:
    case 'char':
        MAX_LENGTH = 32
    case 'word':
        MAX_LENGTH = 8

def max_len2():
    return 1 << (MAX_LENGTH - 1).bit_length()

def split_tokens(sentence):
    match SPLIT_TYPE:
        case 'char':
            return list(sentence)
        case 'word':
            return sentence.split(' ')

def join_tokens(toks):
    match SPLIT_TYPE:
        case 'char':
            return ''.join(toks)
        case 'word':
            return ' '.join(toks)

SOS_token = 0
EOS_token = 1

class TokenLang:
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {SOS_token: '<SOS>', EOS_token: '<EOS>'}
        self.n_tokens = 2

    def add_token(self, tok):
        if tok not in self.token2index:
            self.token2index[tok] = self.n_tokens
            self.token2count[tok] = 1
            self.index2token[self.n_tokens] = tok
            self.n_tokens += 1
        else:
            self.token2count[tok] += 1

    def add_sentence(self, sentence):
        for tok in split_tokens(sentence):
            self.add_token(tok)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    raw_path = os.path.join(os.path.dirname(__file__), f'raw/{lang1}-{lang2}.txt')
    lines = open(raw_path, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = TokenLang(lang2)
        output_lang = TokenLang(lang1)
    else:
        input_lang = TokenLang(lang1)
        output_lang = TokenLang(lang2)

    return input_lang, output_lang, pairs

def filter_pair(p):
    return len(split_tokens(p[0])) < MAX_LENGTH and \
        len(split_tokens(p[1])) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1, lang2, reverse = False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    xpairs = pairs
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting tokens...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted tokens:")
    print(input_lang.name, input_lang.n_tokens)
    print(output_lang.name, output_lang.n_tokens)
    return input_lang, output_lang, xpairs
