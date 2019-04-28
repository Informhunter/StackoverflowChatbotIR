import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize

# Preprocessing

FILTER_RE = re.compile(r'(<[^>]+>)|(&[a-z]{2,5};)')
TOKEN_RE = re.compile(r'([a-zA-Z\.\-\$#][0-9a-zA-Z\.\-\$#]+)')
TAGS_RE = re.compile(r'<([^<>]+)')
SPLIT_RE = re.compile(r'[\.\?\!]\s|[\r\n]+')
ENG_STOP = stopwords.words('english')


def create_tokenizer(token_regex=TOKEN_RE, filter_regex=FILTER_RE, stopwords=ENG_STOP):
    def _tokenizer(text):
        text = text.lower()
        if not filter_regex is None:
            text = filter_regex.sub(' ', text)
        if not stopwords is None:
            return [x for x in token_regex.findall(text) if not x in stopwords]
        else:
            return [x for x in token_regex.findall(text)]
    return _tokenizer


default_tokenizer = create_tokenizer()
space_tokenizer = create_tokenizer(re.compile(r'[^\s]*(.*)[^\s]*'))

def bigram_tokenizer(text, unigram_tokenizer=default_tokenizer):
    tokens = unigram_tokenizer(text)
    return [x[0] + '_' + x[1] for x in nltk.bigrams(tokens)]



# Embeddings

def load_starspace(path):
    result = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            word, *emb = line.rstrip().split('\t')
            result[word] = np.array([float(x) for x in emb], dtype='float32')
    return result


def load_word_embeddings(path, embedding_type='starspace'):
    if embedding_type == 'starspace':
        return load_starspace(path)


def text_to_embedding(text, word_embeddings, tokenizer=default_tokenizer, token_weights=None, n_tokens=100):
    emb_shape = len(next(iter(word_embeddings.values())))
    tokens = tokenizer(text)
    usable_tokens = [token for token in tokens if token in word_embeddings]
    train_tokens = usable_tokens[:n_tokens]
    embeddings = np.array([word_embeddings[token] for token in train_tokens])
    if embeddings.shape[0] == 0:
        embeddings = np.zeros((1, emb_shape))

    result = normalize(np.mean(embeddings, axis=0).reshape(1, -1))[0]
    return result

# Hash functions

def get_hash(hash_function, emb):
    return (np.dot(emb, hash_function) > 0).astype('int8')

def get_buckets(hash_functions, emb):
    hashes = [get_hash(hash_function, emb) for hash_function in hash_functions]
    buckets = [h.dot(1 << np.arange(h.shape[-1] - 1, -1, -1)) for h in hashes]
    return buckets


# Validation

def hits_metric(ranked_lists, expected_ids, k=5):
    return np.mean([1 if e_id in r_list[:k] else 0
                    for e_id, r_list in zip(expected_ids, ranked_lists)])

def dcg_metric(ranked_lists, expected_ids, k=5):
    return np.mean([1 / np.log2(2 + r_list.index(e_id)) if e_id in r_list[:k] else 0
                    for e_id, r_list in zip(expected_ids, ranked_lists)])