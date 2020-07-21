import re
from itertools import chain

from konlpy.tag import Mecab, Okt
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from nlp_commons.functions import *


class BagOfWords:
    def __init__(self, list_of_texts=None):
        lemm = WordNetLemmatizer()
        self.lemmatizer = lemm.lemmatize
        self.kor_tokenizer = KoreanTokenizer('mecab')

        if list_of_texts:
            self.make_bow(list_of_texts)

    def make_tokenized_matrix_english(self, texts: List[str], lemmatize=True):
        if lemmatize:
            self.tokenized_matrix = [[self.lemmatizer(word) for word in word_tokenize(text)] for text in texts]
        else:
            self.tokenized_matrix = [word_tokenize(text) for text in texts]

    def make_tokenized_matrix_korean(self, texts: List[str]):
        pass

    def make_filtered_tokenized_matrix_korean(self, token_matrix):
        self.tokenized_matrix = [[t[0] for t in corp['tokens'] if t[1] in self.kor_tokenizer.include_poses] for corp in
                                 token_matrix]

    def make_bow(self):
        assert self.tokenized_matrix
        self.unique_tokens = get_uniques_from_nested_lists(self.tokenized_matrix)
        self.token2idx, self.idx2token = get_item2idx(self.unique_tokens, unique=True)
        self.vocab_size = len(self.token2idx)

    def get_window_pairs(self, tokens: List[str], win_size=4, as_index=True) -> List[Tuple]:
        window_pairs = []
        for idx, token in enumerate(tokens):
            start = max(0, idx - win_size)
            end = min(len(tokens), idx + win_size + 1)
            for win_idx in range(start, end):
                if not idx == win_idx:
                    pair = (token, tokens[win_idx])
                    pair = pair if not as_index else tuple(self.token2idx[t] for t in pair)
                    window_pairs.append(pair)
        return window_pairs

    def make_pairs_matrix(self, win_size, as_index=True):
        self.pairs_matrix = [self.get_window_pairs(sent, win_size, as_index) for sent in self.tokenized_matrix]
        self.pairs_flat = list(chain.from_iterable(self.pairs_matrix))


class RegexWrapper:
    """
    정규표현식 토크나이저 wrapper class
    """

    def __init__(self):
        self.regex = re.compile('[가-힇|a-z|A-Z]+')

    def pos(self, text):
        """
        konlpy.tag.pos의 리턴형식과 형식을 맞춘 결과 반환
        :param str text: 인풋 텍스트
        :return list
        """
        return [(t, 'REGEX') for t in self.regex.findall(text)]

    def morphs(self, text):
        """
        토큰 리스트 반환
        :param str text: 인풋 텍스트
        :return list
        """
        return self.regex.findall(text)


class KoreanTokenizer:
    """
    konlpy와 정규표현식 토크나이즈의 wrapper class
    """
    tokenizer_list = {
        'mecab', 'okt', 'twitter', 'regex'
    }

    __stopwords = ['및', '후', '사업', '산업', '대부분', '광범위', '습니다', '있습니다', '으로', 'dart', 'fss', 'or', 'kr',
                   'Page']

    __toks = {
        'mecab': Mecab,
        'okt': Okt,
        'twitter': Okt,
        'regex': RegexWrapper
    }

    __exclude_poses_dic = {
        'okt': ['Punctuation', 'Number', 'Josa', 'Determiner', 'Suffix', 'Modifier'],
        'twitter': ['Punctuation', 'Number', 'Josa', 'Determiner', 'Suffix', 'Modifier'],
    }
    __include_poses_dic = {
        'okt': ['Noun', 'Alpha'],
        'twitter': ['Noun', 'Alpha'],
        'mecab': ['NNG', 'NNP'],
        'regex': ['REGEX']
    }

    def __init__(self, tok_name):
        tok = self.__toks.get(tok_name.lower())
        if not tok:
            raise
        else:
            self.tokenizer = tok()
            self.tokenizer_name = tok_name.lower()
            self.include_poses = self.__include_poses_dic.get(self.tokenizer_name)
            self.exclude_poses = self.__exclude_poses_dic.get(self.tokenizer_name)
            self.stopwords = self.__stopwords[:]
            self.stopwords_hash = {k: 1 for k in self.stopwords}

    def pos(self, text):
        return self.tokenizer.pos(text)

    def nouns(self, text):
        return self.tokenizer.nouns(text)

    def morphs(self, text):
        return self.tokenizer.morphs(text)
