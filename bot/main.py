import sys
import os.path
from pipeline import *
from ranker import Ranker
from bot import start_bot


def usage():
    print('''
Usage:
1. Extract data from zip archive to sqlite database
2. Create embedding training file from sqlite database
3. Train word embeddings
{0} extract_data <zip file path> <db path>
{0} create_etf   <db path> <embedding training file path>
{0} train_we     <starspace executable path> <embedding training file path> <word embeddings path>
{0} create_v     <db path > <results path>
{0} create_pe    <db path > <word embeddings path> <results path>
{0} evaluate     <dir with all embeddings> <validation path> <evaluation results path>
'''.format(sys.argv[0]))


def main():
    if len(sys.argv) == 1:
        usage()
        return
    if sys.argv[1] == 'extract_data':
        print('Loading data from {} to DB {}'.format(sys.argv[2], sys.argv[3]))
        load_posts_to_db(sys.argv[2], sys.argv[3], use_tqdm=True)
    elif sys.argv[1] == 'create_etf':
        print('Creating training file for word embeddings')
        create_embedding_training_file(sys.argv[2], sys.argv[3], use_tqdm=True)
    elif sys.argv[1] == 'train_we':
        train_embeddings(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'create_v':
        print('Creating validation set')
        create_validation(sys.argv[2], sys.argv[3], use_tqdm=True)
    elif sys.argv[1] == 'create_pe':
        print('Creating post embeddings')
        create_post_embeddings(sys.argv[2], sys.argv[3], sys.argv[4], use_tqdm=True)
    elif sys.argv[1] == 'create_lsh':
        create_lsh(sys.argv[2], sys.argv[3], use_tqdm=True)
    elif sys.argv[1] == 'evaluate':
        print('Evaluating ranker')
        r = Ranker()
        we_path = os.path.join(sys.argv[2], 'embeddings.emb.tsv')
        pe_path = os.path.join(sys.argv[2], 'post_embeddings.pkl4')
        idr_path = os.path.join(sys.argv[2], 'post_ids.pkl4')

        r.load(we_path=we_path, pe_path=pe_path, idr_path=idr_path)
        
        evaluate_ranker(r, sys.argv[3], sys.argv[4], use_tqdm=True)
    elif sys.argv[1] == 'start_bot':
        start_bot(sys.argv[2])
    else:
        pass

if __name__ == "__main__":
    main()
