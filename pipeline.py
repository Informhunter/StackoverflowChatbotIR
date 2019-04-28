import re
import numpy as np
import os.path
import sqlite3
import pickle
import xmltodict
from zipfile import ZipFile
import subprocess
from sklearn.preprocessing import normalize
from util import default_tokenizer, text_to_embedding,\
                 load_word_embeddings, hits_metric, dcg_metric, TAGS_RE, SPLIT_RE,\
                 get_buckets, get_hash
from itertools import chain
import tqdm

def load_posts_to_db(zfile_path, db_path, use_tqdm=False):
    with ZipFile(zfile_path) as z,\
         z.open('Posts.xml') as f,\
         sqlite3.connect(db_path) as con:
        
        cur = con.cursor()

        for _ in range(3):
            f.readline()

        cur.execute('create table Posts('\
            'Id int primary key,'\
            'Score int,'\
            'Title text,'\
            'Body text,'\
            'Tags text'\
            ');')
        
        buffer = []

        if use_tqdm:
            f = tqdm.tqdm(f)
        
        for line in f:
            try:
                r = xmltodict.parse(line)
            except:
                cur.executemany('insert or ignore into Posts values(?, ?, ?, ?, ?)', buffer)
                con.commit()
                buffer = []
                break
            
            if r['row']['@PostTypeId'] != '1' or not '@Tags' in r['row']:
                continue
            
            tags = set(TAGS_RE.findall(r['row']['@Tags']))
                
            score = int(r['row']['@Score'])
            post_id = int(r['row']['@Id'])
            title = r['row']['@Title']
            body = r['row']['@Body']
            

            buffer.append((post_id, score, title, body, ' '.join(tags)))
            
            if(len(buffer) >= 50000):
                cur.executemany('insert or ignore into Posts values(?, ?, ?, ?, ?);', buffer)
                con.commit()
                buffer = []


def create_embedding_training_file(db_path, outfile_path, tokenizer=default_tokenizer, use_tqdm=False):
    with open(outfile_path, 'w', encoding='utf-8') as f,\
         sqlite3.connect(db_path) as con:
        
        cur = con.cursor()
        buffer = []
        result = cur.execute('select Title || ". " || Body from Posts;')
        if use_tqdm:
            result = tqdm.tqdm(result)
        for post_text in result:
            sentences = []
            for sentence in SPLIT_RE.split(post_text[0]):
                sentence = ' '.join(tokenizer(sentence))
                if len(sentence) > 0:
                    sentences.append()
            buffer.append('\t'.join(sentences) + '\n')
            if len(buffer) >= 50000:
                f.writelines(buffer)
                f.flush()
                buffer = []

        f.writelines(buffer)
        f.flush()


def train_embeddings(starspace_exec, train_path, model_path, arguments=None):
    default_arguments = {
        '-trainFile': train_path,
        '-model': model_path,
        '-trainMode': '3',
        '-adagrad': 'true',
        '-ngrams': '1',
        '-lr': '0.05',
        '-epoch': '5',
        '-dim': '200',
        '-negSearchLimit': '10',
        '-fileFormat': 'labelDoc',
        '-similarity': 'cosine',
        '-minCount': '100',
        '-verbose': 'true',
        '-trainWord': '1',
    }

    if not arguments is None:
        default_arguments.update(arguments)
    
    subprocess.run([starspace_exec, 'train'] + 
                   list(chain(*[(k, v) for k, v in default_arguments.items()])))

def create_validation(post_db_path, results_dir, n_tokens=100,
                      tokenizer=default_tokenizer, use_tqdm=False):
    with sqlite3.connect(post_db_path) as con:
        cur = con.cursor()
        validation_set = []
        result = cur.execute('select Id, Title, Body from Posts where Score > 3;')
        if use_tqdm:
            result = tqdm.tqdm(result)
        
        for post_id, title, body in result:
            tokens = tokenizer(title + '. ' + body)[n_tokens:]
            if len(tokens) > 0:
                validation_set.append((tokens, post_id))

    validation_path = os.path.join(results_dir, 'validation.tsv')
    with open(validation_path, 'w', encoding='utf-8') as f:
        for validation_tokens, post_id in validation_set:
            f.write(' '.join(validation_tokens))
            f.write('\t')
            f.write(str(post_id) + '\n')


def create_post_embeddings(post_db_path, we_path, results_dir, n_tokens=100,
                           tokenizer=default_tokenizer, use_tqdm=False):
    word_embeddings = load_word_embeddings(we_path)
    with sqlite3.connect(post_db_path) as con:

        cur = con.cursor()
        post_embeddings = []
        post_ids = []
        result = cur.execute('select Id, Title, Body from Posts where Score > 3;')
        if use_tqdm:
            result = tqdm.tqdm(result)
        
        for post_id, title, body in result:
            post_ids.append(post_id)
            embedding = text_to_embedding(title + '. ' + body,
                                          word_embeddings,
                                          n_tokens=n_tokens,
                                          tokenizer=tokenizer)
            post_embeddings.append(embedding)

    post_embeddings = np.array(post_embeddings)

    embeddings_path = os.path.join(results_dir, 'post_embeddings.pkl4')
    post_ids_path = os.path.join(results_dir, 'post_ids.pkl4')

    with open(embeddings_path, 'wb') as f:
        pickle.dump(post_embeddings, f, protocol=4)

    with open(post_ids_path, 'wb') as f:
        pickle.dump(post_ids, f, protocol=4)


def create_lsh(pe_path, results_path, embeddings_shape=200, hash_size=20, n_hashes=100, use_tqdm=False):
    hash_functions = [np.random.rand(embeddings_shape, hash_size) - 0.5 for _ in range(n_hashes)]
    hf_path = os.path.join(results_path, 'hash_functions.pkl')
    with open(hf_path, 'wb') as f:
        pickle.dump(hash_functions, f)

    with open(pe_path, 'rb') as f:
        post_embeddings = pickle.load(f)

    db_path = os.path.join(results_path, 'bucket_registry.db')
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute('create table if not exists BucketRegistry('\
                 'BucketId int,'\
                 'PostIndex int);')
        cur.execute('create index if not exists bucket_bucket_id on BucketRegistry(BucketId);')

        hash_functions = enumerate(hash_functions)
        if use_tqdm:
            hash_functions = tqdm.tqdm(hash_functions)

        for hash_index, hash_function in hash_functions:
            buckets = get_buckets([hash_function], post_embeddings)[0]
            values = []
            for post_index, post_bucket in enumerate(buckets):
                values.append((int(post_bucket), int(post_index)))
            cur.executemany('insert into BucketRegistry values(?, ?);', values)
            con.commit()



EVALUATION_REPORT = '''Hits@1\t{0:.3f}
Hits@5\t{0:.3f}
Hits@10\t{0:.3f}
Hits@100\t{0:.3f}
DCG@5\t{0:.3f}
DCG@10\t{0:.3f}
DCG@100\t{0:.3f}'''

def evaluate_ranker(ranker, validation_path, results_path, use_tqdm=False):
    queries = []
    expected_ids = []
    with open(validation_path, encoding='utf-8') as f:
        for line in f:
            query, expected_id = line.strip().split('\t')
            queries.append(query)
            expected_ids.append(expected_id)
    expected_ids = expected_ids[::10000]
    queries = queries[::10000]
    if use_tqdm:
        queries = tqdm.tqdm(queries)
    
    retreival_results = [ranker.search(query, 100, n_tokens=10) for query in queries]
    hits1 = hits_metric(retreival_results, expected_ids, 1)
    hits5 = hits_metric(retreival_results, expected_ids, 5)
    hits10 = hits_metric(retreival_results, expected_ids, 10)
    hits100 = hits_metric(retreival_results, expected_ids, 100)

    dcg5 = dcg_metric(retreival_results, expected_ids, 5)
    dcg10 = dcg_metric(retreival_results, expected_ids, 10)
    dcg100 = dcg_metric(retreival_results, expected_ids, 100)

    with open(results_path, 'w') as f:
        f.write(EVALUATION_REPORT.format(hits1, hits5, hits10, hits100, dcg5, dcg10, dcg100))