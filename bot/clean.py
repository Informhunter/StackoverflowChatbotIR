import re
import tqdm

SPLIT_RE = re.compile(r'[\t]+')


with open('D:\\SNLP data\\embeddings_training.tsv', encoding='utf-8') as infile,\
     open('D:\\SNLP data\\embeddings_training_clean.tsv', 'w', encoding='utf-8') as outfile:
    buffer = []
    for line in tqdm.tqdm(infile):
        line = line.strip()
        line = '\t'.join(SPLIT_RE.split(line)) + '\n'
        buffer.append(line)
        if len(buffer) >= 1000000:
            outfile.writelines(buffer)
            buffer = []

    if len(buffer) >= 0:
        outfile.writelines(buffer)
        buffer = []

