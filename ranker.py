import re
import sqlite3
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from itertools import chain
from collections import defaultdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from util import default_tokenizer, text_to_embedding, load_word_embeddings


def get_hash(hash_function, emb):
    return (np.dot(emb, hash_function)>0).astype('int8')

def get_buckets(hash_functions, emb):
    hashes = [get_hash(hash_function, emb) for hash_function in hash_functions]
    buckets = [h.dot(1 << np.arange(h.shape[-1] - 1, -1, -1)) for h in hashes]
    return buckets

def query_to_hash(query, hash_function, emb):
    query_emb = text_to_embedding(query, emb).reshape(1, -1)
    return get_hash(hash_function, query_emb)

def query_to_buckets(query, hash_functions, emb):
    query_emb = text_to_embedding(query, emb).reshape(1, -1)
    buckets = get_buckets(hash_functions, query_emb)
    return [x[0] for x in buckets]
    

def search_full(query, post_embeddings, emb, k, n_tokens=100):
    query_emb = text_to_embedding(query, emb, n_tokens=n_tokens).reshape(1, -1)
    topk = []
    for i in range(0, post_embeddings.shape[0], 100000):
        sims = cosine_similarity(post_embeddings[i:i + 100000], query_emb)
        idxs = np.argpartition(sims, -k, axis=None)[-k:]
        topk += list(zip(idxs + i, sims[idxs]))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)[:k]
    return topk


def search_buckets_sqlite(query, hash_functions, post_embeddings, con, emb, k, n_tokens=100):
    query_emb = text_to_embedding(query, emb, n_tokens=n_tokens).reshape(1, -1)
    query_buckets = query_to_buckets(query, hash_functions, emb)
    cur = con.cursor()
    topk = []
    for bucket_id in query_buckets:
        potential_ids = cur.execute('select PostIndex from BucketRegistry where BucketId = ?;', (int(bucket_id),))
        potential_ids = set([x[0] for x in potential_ids])
        potential_ids = list(potential_ids - set([x[0] for x in topk]))
        sims = cosine_similarity(post_embeddings[potential_ids], query_emb)
        top = max(-k, -len(potential_ids))
        idxs = np.argpartition(sims, top, axis=None)[-top:]
        idxs_ = [potential_ids[x] for x in idxs]
        topk += list(zip(idxs_, sims[idxs]))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)[:k]
    cur.close()
    return topk


def search_buckets_memory(query, hash_functions, post_embeddings, bucket_registry, emb, k, n_tokens=100):
    query_emb = text_to_embedding(query, emb, n_tokens=n_tokens).reshape(1, -1)
    query_buckets = query_to_buckets(query, hash_functions, emb)
    topk = []
    for bucket_id in query_buckets:
        potential_ids = bucket_registry[bucket_id]
        potential_ids = list(potential_ids - set([x[0] for x in topk]))
        sims = cosine_similarity(post_embeddings[potential_ids], query_emb)
        top = max(-k, -len(potential_ids))
        idxs = np.argpartition(sims, top, axis=None)[-top:]
        idxs_ = [potential_ids[x] for x in idxs]
        topk += list(zip(idxs_, sims[idxs]))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)[:k]
    return topk


class Ranker:
    def __init__(self,
                 word_embeddings=None,
                 post_embeddings=None,
                 hash_functions=None,
                 db_path=None,
                 id_registry=None,
                 bucket_registry=None):
        self.word_embeddings = word_embeddings
        self.post_embeddings = post_embeddings
        self.hash_functions = hash_functions
        self.db_path = db_path
        self.bucket_registry = bucket_registry
        self.id_registry = id_registry


    def load(self, we_path=None,
             pe_path=None, hf_path=None,
             db_path=None, br_path=None, idr_path=None):

        if not we_path is None:
            self.word_embeddings = load_word_embeddings(we_path)

        if not pe_path is None:
            with open(pe_path, 'rb') as f:
                self.post_embeddings = pickle.load(f)
        
        if not hf_path is None:
            with open(hf_path, 'rb') as f:
                self.hash_functions = pickle.load(f)

        if not db_path is None:
            self.db_path = db_path

        if not br_path is None:
            with open(br_path, 'rb') as f:
                self.bucket_registry = pickle.load(f)
    
        if not idr_path is None:
            with open(idr_path, 'rb') as f:
                self.id_registry = pickle.load(f)

    def search(self, query, topk, mode='full', original_ids=True, n_tokens=100):
        if self.word_embeddings is None:
            raise RuntimeError('No word embeddings provided')
        if self.post_embeddings is None:
            raise RuntimeError('No post embeddings provided')

        if mode == 'full':
            result = search_full(query, self.post_embeddings, self.word_embeddings, topk, n_tokens=n_tokens)
        elif mode == 'lsh_mem':
            if self.hash_functions is None:
                raise RuntimeError('No hash functions provided')
            if self.bucket_registry is None:
                raise RuntimeError('No buckets registry provided')
            result = search_buckets_memory(query,
                                         self.hash_functions,
                                         self.post_embeddings,
                                         self.bucket_registry,
                                         self.word_embeddings,
                                         topk,
                                         n_tokens=n_tokens)
        elif mode == 'lsh_sqlite':
            if self.hash_functions is None:
                raise RuntimeError('No hash functions provided')
            if self.db_path is None:
                raise RuntimeError('No sqlite db path provided')
            con = sqlite3.connect(self.db_path)
            result = search_buckets_sqlite(query,
                                         self.hash_functions,
                                         self.post_embeddings,
                                         con,
                                         self.word_embeddings,
                                         topk,
                                         n_tokens=n_tokens)
        else:
            raise RuntimeError('Unknown search mode')
        if original_ids:
            return [self.id_registry[x[0]] for x in result]
        else:
            return [x[0] for x in result]

