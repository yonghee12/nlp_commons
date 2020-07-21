from typing import *
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def get_sequences_from_tokens(token_list: List[str], token2idx: Dict) -> List:
    token_indices = [token2idx[token] for token in token_list]
    sequences = [token_indices[:i + 1] for i in range(1, len(token_indices))]
    return sequences


def pad_sequence_list(sequence: Iterable, max_len: int, method: str, truncating: str, value=0):
    assert isinstance(sequence, Iterable)
    sequence = list(sequence)
    if len(sequence) > max_len:
        if not truncating:
            raise Exception("The Length of a sequence is longer than max_len")
        if method == 'pre':
            return sequence[len(sequence) - max_len:]
        elif method == 'post':
            return sequence[:max_len - len(sequence)]
    else:
        if method == 'pre':
            return [value for _ in range(max_len - len(sequence))] + sequence
        elif method == 'post':
            return sequence + [value for _ in range(max_len - len(sequence))]


def pad_sequence_nested_lists(nested_sequence, max_len, method='pre', truncating='pre'):
    return [pad_sequence_list(seq, max_len, method, truncating) for seq in nested_sequence]


def to_categorical_one(index, length) -> np.ndarray:
    onehot = np.zeros(shape=(length,))
    onehot[index] = 1


def to_categorical_iterable(classes: Iterable, num_classes: int):
    assert isinstance(classes, Iterable)
    nrows, ncols = len(classes), num_classes
    onehot = np.zeros(shape=(nrows, ncols))
    onehot[range(nrows), classes] = 1
    return onehot


def get_uniques_from_nested_lists(nested_lists: List[List]) -> List:
    uniques = {}
    for one_line in nested_lists:
        for item in one_line:
            if not uniques.get(item):
                uniques[item] = 1
    return list(uniques.keys())


def get_item2idx(items, unique=False, from_one=False) -> Tuple[Dict, Dict]:
    item2idx, idx2item = dict(), dict()
    items_unique = items if unique else set(items)
    for idx, item in enumerate(items_unique):
        i = idx + 1 if from_one else idx
        item2idx[item] = i
        idx2item[i] = item
    return item2idx, idx2item


def tsne_plot(labels, vectors, filename, perplexity=10, figsize=(8, 8), cmap='nipy_spectral', dpi=300):
    tsne_model = TSNE(perplexity=perplexity, n_components=2,
                      metric='cosine',
                      init='pca', n_iter=5000, random_state=22)
    new_values = tsne_model.fit_transform(vectors)

    x, y = [], []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.rcParams["font.family"] = 'D2Coding'
    plt.clf()
    plt.figure(figsize=figsize)
    plt.title(filename)
    plt.scatter(x, y, cmap=cmap, alpha=0.5)

    for i in range(len(x)):
        #         plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    timestamp = dt.today().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("results/{}_perp{}.png".format(filename, perplexity), dpi=dpi)
