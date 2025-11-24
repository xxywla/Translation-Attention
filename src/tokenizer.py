from abc import abstractmethod, ABC

import nltk
from nltk import TreebankWordDetokenizer
from tqdm import tqdm


class BaseTokenizer(ABC):
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)

        self.word2index = {word: i for i, word in enumerate(vocab_list)}
        self.index2word = {i: word for i, word in enumerate(vocab_list)}

        self.unk_index = self.word2index[self.unk_token]
        self.pad_index = self.word2index[self.pad_token]
        self.sos_index = self.word2index[self.sos_token]
        self.eos_index = self.word2index[self.eos_token]

    @staticmethod
    @abstractmethod
    def tokenize(text):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass

    def encode(self, text, seq_len=None, add_sos_eos=False):
        word_list = self.tokenize(text)
        if seq_len is None:
            if add_sos_eos:
                word_list = [self.sos_token] + word_list + [self.eos_token]
            return [self.word2index.get(word, self.unk_index) for word in word_list]

        if add_sos_eos:
            seq_len -= 2
            if len(word_list) == seq_len:
                word_list = [self.sos_token] + word_list + [self.eos_token]
            elif len(word_list) < seq_len:
                word_list = [self.sos_token] + word_list + [self.eos_token] + [self.pad_token] * (
                        seq_len - len(word_list))
            else:
                word_list = [self.sos_token] + word_list[:seq_len] + [self.eos_token]
        else:
            if len(word_list) < seq_len:
                word_list = word_list + [self.pad_token] * (seq_len - len(word_list))
            elif len(word_list) > seq_len:
                word_list = word_list[:seq_len]

        return [self.word2index.get(word, self.unk_index) for word in word_list]

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, "r", encoding='utf-8') as f:
            vocab_list = [line[:-1] for line in f.readlines()]
        return cls(vocab_list)

    @classmethod
    def build_vocab(cls, vocab_path, sentences):
        print('开始构建词表...')
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            for word in cls.tokenize(sentence):
                if word.strip() == '':
                    continue
                vocab_set.add(word)
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word in vocab_list:
                f.write(word + "\n")
        print(f'词表保存完成，大小为{len(vocab_list)}')


class ChineseTokenizer(BaseTokenizer):

    def decode(self, tokens):
        return ''.join([self.index2word[index] for index in tokens])

    @staticmethod
    def tokenize(text):
        return list(text)


class EnglishTokenizer(BaseTokenizer):

    def decode(self, tokens):
        word_list = [self.index2word[index] for index in tokens]
        return TreebankWordDetokenizer().detokenize(word_list)

    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text)


if __name__ == '__main__':
    print(ChineseTokenizer.tokenize("我喜欢你"))
    print(EnglishTokenizer.tokenize("I'm in China."))
