import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

from . import textproc

def indexes_from_sentence(lang, sentence):
    return [lang.token2index[tok] for tok in textproc.split_tokens(sentence)]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(textproc.EOS_token)
    
    while len(indexes) & (len(indexes) - 1):
        indexes.append(0) 
    
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

def sentence_from_tensor(lang, tensor):
    _, topi = tensor.topk(1)
    token_ids = topi.squeeze()

    tokens = []
    for idx in token_ids:
        if idx.item() == textproc.EOS_token:
            break
        tokens.append(lang.index2token[idx.item()])

    sentence = textproc.join_tokens(tokens)
    return sentence

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(lang1, lang2, batch_size):
    input_lang, output_lang, pairs = textproc.prepare_data(lang1, lang2)

    n = len(pairs)
    input_ids = np.zeros((n, textproc.max_len2()), dtype=np.int32)
    target_ids = np.zeros((n, textproc.max_len2()), dtype=np.int32)
    
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexes_from_sentence(input_lang, inp)
        tgt_ids = indexes_from_sentence(output_lang, tgt)
        inp_ids.append(textproc.EOS_token)
        tgt_ids.append(textproc.EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader
