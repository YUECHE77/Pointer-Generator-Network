import os
import json
import jieba
import unicodedata
from collections import Counter


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True

    cat = unicodedata.category(char)
    if cat == "Zs":
        return True

    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char == "\t" or char == "\n" or char == "\r":
        return False

    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True

    return False


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []

    for char in text:
        cp = ord(char)

        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue

        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)

    return "".join(output)


class BasicTokenizer(object):
    """Runs basic tokenization using jieba for Chinese text."""

    def __init__(self, do_lower_case=True):
        """
        Constructs a BasicTokenizer

        do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text using jieba."""
        text = _clean_text(text)

        if self.do_lower_case:
            text = text.lower()

        # Use jieba to tokenize the text
        tokens = list(jieba.cut(text))

        return tokens


class Tokenizer:
    def __init__(self, max_wordn, divide, lines, min_freq, existed_txt_path):
        self.max_wordn = max_wordn  # max length of tokenizer
        self.divide = divide  # basicTokenizer
        self.min_freq = min_freq  # min-frequency

        self.existed_txt_path = existed_txt_path  # path to dict.txt (if you already have dict.txt)

        self.word2idx = {}
        self.idx2word = {}

        self.build_dict(lines)

    def build_dict(self, sents):
        if self.existed_txt_path is not None:
            if os.path.exists(self.existed_txt_path):
                print("Using existing dict")

                with open(self.existed_txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                    for index, line in enumerate(lines):
                        word = line.strip()
                        self.word2idx[word] = index
                        self.idx2word[index] = word

                print("Dict len: ", len(self.word2idx))

            else:
                raise FileNotFoundError(f'The dictionary path {self.existed_txt_path} does not exist')

        else:
            all_vocab = []
            reserved_tokens = ['<PAD>', '<OOV>', '<s>', '</s>']

            for sent in sents:
                tokens = self.divide.tokenize(sent)  # tokens is a list
                all_vocab.extend(tokens)  # use extend, instead of append

            counter = Counter(all_vocab)

            # Filter words below the min_freq threshold
            count_pairs = [(word, freq) for word, freq in counter.items() if freq >= self.min_freq]
            count_pairs = sorted(count_pairs, key=lambda x: -x[1])  # sort by frequency (descending)
            count_pairs = count_pairs[:self.max_wordn - len(reserved_tokens)]  # the length must smaller than max_length

            # count_pairs: [(tuple_1), (tuple_2), ..., (tuple_n)] -> (tuple_i) = (word_i, freq_i) -> zip(*count_pairs)
            # -> get 'words' in each tuple
            words, _ = zip(*count_pairs)

            words = reserved_tokens + list(words)  # words: (word_1, word_2, ..., word_n) -> convert to list

            for pos, word in enumerate(words):
                self.word2idx[word] = pos
                self.idx2word[pos] = word

            print("Dict len: ", len(self.word2idx))

            with open('dict.txt', 'w', encoding='utf-8') as f:
                for i in range(len(self.word2idx)):
                    f.write(self.idx2word[i] + '\n')


def build_tokenizer(json_file, src_name='source', n_src_vocab=50000, min_freq=2, existed_txt_path=None):
    corpus = []

    with open(json_file, 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()

        for line in temp:
            dic = json.loads(line)
            corpus.append(dic[src_name])

    divide = BasicTokenizer()

    tokenizer = Tokenizer(n_src_vocab, divide, lines=corpus, min_freq=min_freq, existed_txt_path=existed_txt_path)

    print('\n==== Tokenizer built successfully ====\n')

    return tokenizer
