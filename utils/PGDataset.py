import jieba
import json

import torch
from torch.utils.data import Dataset


def truncate_pad(tokenized_result, length, padding_token):
    if len(tokenized_result) > length:
        tokenized_result = tokenized_result[:length]
        valid_len = len(tokenized_result)

        return tokenized_result, valid_len

    valid_len = len(tokenized_result)
    tokenized_result = tokenized_result + [padding_token] * (length - len(tokenized_result))

    return tokenized_result, valid_len


def origin_sent2id(sent, tokenizer, length=512):
    ids = []

    tokens = list(jieba.cut(sent))

    for i in tokens:
        if i in tokenizer.word2idx:
            ids.append(tokenizer.word2idx[i])
        else:
            ids.append(tokenizer.word2idx['<OOV>'])

    ids.append(tokenizer.word2idx['</s>'])

    ids, true_len = truncate_pad(ids, length=length, padding_token=tokenizer.word2idx['<PAD>'])

    # ids: tokenized result -> we don't need true_len here
    return ids


def source2id(sent, tokenizer, length=512, vocab_size=50000):
    """tokenize source sentence without <OOV> token"""

    ids = []
    oovs = []  # oovs能记录所有不在词表的tokens

    tokens = list(jieba.cut(sent))  # 使用 jieba 进行分词

    for i in tokens:
        if i in tokenizer.word2idx:
            ids.append(tokenizer.word2idx[i])
        else:
            if i not in oovs:
                oovs.append(i)

            oov_index = oovs.index(i)
            ids.append(vocab_size + oov_index)

    ids.append(tokenizer.word2idx['</s>'])

    ids, true_len = truncate_pad(ids, length=length, padding_token=tokenizer.word2idx['<PAD>'])

    # ids: tokenized result
    # oovs: all the words (words, not tokens) NOT in vocab -> [string_1, string_2, ..., string_n]
    # true_len: valid length
    return ids, oovs, true_len


def target2id(sent, oovs, tokenizer, length=512, vocab_size=50000):
    """
        tokenize target sentence with oov list from source sentence
        so that the model can outputs words in original sentence
    """

    ids = []

    tokens = list(jieba.cut(sent))

    for i in tokens:
        if i in tokenizer.word2idx:
            ids.append(tokenizer.word2idx[i])
        else:
            if i in oovs:
                # Which means, the current word is not in vocab, but in the source sentence
                ids.append(vocab_size + oovs.index(i))
            else:
                # the current word is neither in vocab, nor in the source sentence -> we have to mark it as <OOV>
                ids.append(tokenizer.word2idx['<OOV>'])

    ids.append(tokenizer.word2idx['</s>'])

    ids, true_len = truncate_pad(ids, length=length, padding_token=tokenizer.word2idx['<PAD>'])

    return ids, true_len


class SumDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_source_length=512, max_target_length=512,
                 src_name='source', tgt_name='target'):

        vocab_size = len(tokenizer.word2idx)
        print(f'vocab size is: {vocab_size}')

        self.source_sent = []
        self.target_sent = []

        self.source_sent_ext = []
        self.target_sent_ext = []

        self.max_oov_num = []

        self.source_length = []
        self.target_length = []

        with open(json_file, 'r', encoding='utf-8') as load_f:
            temp = load_f.readlines()

            for line in temp:
                dic = json.loads(line)

                ids, oovs, source_len = source2id(dic[src_name], tokenizer,
                                                  length=max_source_length, vocab_size=vocab_size)

                self.source_length.append(source_len)
                self.source_sent_ext.append(ids)
                self.max_oov_num.append(len(oovs))

                ids, target_len = target2id(dic[tgt_name], oovs, tokenizer,
                                            length=max_target_length, vocab_size=vocab_size)

                self.target_length.append(target_len)
                self.target_sent_ext.append(ids)

                self.source_sent.append(origin_sent2id(dic[src_name], tokenizer, length=max_source_length))
                self.target_sent.append(origin_sent2id(dic[tgt_name], tokenizer, length=max_target_length))

    def __len__(self):
        return len(self.source_sent)

    def __getitem__(self, idx):
        return torch.tensor(self.source_sent[idx]), torch.tensor(self.target_sent[idx]), torch.tensor(
            self.source_sent_ext[idx]), torch.tensor(self.target_sent_ext[idx]), torch.tensor(
            self.max_oov_num[idx]), torch.tensor(self.source_length[idx]), torch.tensor(self.target_length[idx])
