from allennlp.predictors.predictor import Predictor
from nltk import Tree
from tqdm import tqdm
import numpy as np
import os, pdb

import mmap

def get_num_lines(file_path):
    fp = open(file_path, "rb+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

parser_path = "/home/rizwan/.allennlp/cache/elmo-allennlp_constituency_parser"
parser = Predictor.from_path(parser_path)


TRAINING_DIR ='/local/harold/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'
HELDOUT_DIR = TRAINING_DIR.replace('training', 'heldout')
OUTPUT_DIR_NO_LEAF = '../SYNTAX_CORPUS_NO_LEAF/'
OUTPUT_DIR_WITH_LEAF = '../SYNTAX_CORPUS_WITH_LEAF/'

begin = 21
end = 35

#so_far_begin_end(4,20,21,35)

from multiprocessing import Pool
import mmap
from  tqdm import tqdm
import time
import lzma, os
from smart_open import open, register_compressor


def _handle_xz(file_obj, mode):
    return lzma.LZMAFile(filename=file_obj, mode=mode, format=lzma.FORMAT_XZ)
register_compressor('.xz', _handle_xz)


NCORES=32
COMMON_FILE_SZ = 306116
lprocessed = 0


def routine(lines, file_name):
    global lprocessed
    lprocessed+=1
    try:
        with open(OUTPUT_DIR_NO_LEAF + file_name, 'a') as sntx_file_no_leaf, open(OUTPUT_DIR_WITH_LEAF + file_name, 'a') as sntx_file_with_leaf:
            try:
                rnd_number = np.random.randint(low=1, high=3)
                if rnd_number%2==0: parse_trees = [ tree["trees"] for tree in parser.predict_batch_json(lines) ]
                else: parse_trees = [ tree["trees"] for tree in parser.predict_batch_json(lines) ]
                for parse_tree in parse_trees:
                    lparse = ' '.join(str(Tree.fromstring(parse_tree, read_leaf=lambda x: '')).replace('(','').replace(')', '').split())
                    sntx_file_no_leaf.write(lparse + '\n')
                    sntx_file_no_leaf.flush()

                    lparse = ' '.join(
                        str(Tree.fromstring(parse_tree)).replace('(', '').replace(')', '').split())
                    sntx_file_with_leaf.write(lparse + '\n')
                    sntx_file_with_leaf.flush()
                    print(f'{lprocessed}/{COMMON_FILE_SZ}', flush=True)
            except:
                pass
    except:
        with open(OUTPUT_DIR_NO_LEAF + file_name, 'w') as sntx_file_no_leaf, open(OUTPUT_DIR_WITH_LEAF + file_name, 'w') as sntx_file_with_leaf:
            try:
                rnd_number = np.random.randint(low=1, high=3)
                if rnd_number % 2 == 0:
                    parse_trees = [tree["trees"] for tree in parser.predict_batch_json(lines)]
                else:
                    parse_trees = [tree["trees"] for tree in parser.predict_batch_json(lines)]

                for parse_tree in parse_trees:
                    lparse = ' '.join(
                        str(Tree.fromstring(parse_tree, read_leaf=lambda x: '')).replace('(', '').replace(')',
                                                                                                          '').split())
                    sntx_file_no_leaf.write(lparse + '\n')
                    sntx_file_no_leaf.flush()

                    lparse = ' '.join(
                        str(Tree.fromstring(parse_tree)).replace('(', '').replace(')', '').split())
                    sntx_file_with_leaf.write(lparse + '\n')
                    sntx_file_with_leaf.flush()
                    print(f'{lprocessed}/{COMMON_FILE_SZ}', flush=True)
            except:
                pass



def process_wrapper(chunkStart, chunkSize, file_path, file_name):
    with open(file_path) as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        batch = []
        for line in lines:
            batch.append({"sentence": line})
            if len(batch)==8:
                routine(batch, file_name)
                batch=[]

def chunkify(fname,size=1024*1024):
    fileEnd = os.path.getsize(fname)
    with open(fname,'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size,1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break

#init objects
pool = Pool(NCORES)
jobs = []

counter = 0
#create jobs
for dir in [TRAINING_DIR]:#[TRAINING_DIR, HELDOUT_DIR]:
    for file_name in tqdm(os.listdir(dir), total=len(os.listdir(dir))):
        counter+=1
        if counter<begin:
            print(file_name, ' is not processed', flush=True)
            continue
        if counter>end:
            print(file_name, ' is not processed', flush=True)
            continue
        file_path = os.path.join(dir, file_name)
        print(f'Processing: {dir}/{file_name}', flush=True)
        for chunkStart,chunkSize in chunkify(file_path):
            jobs.append( pool.apply_async(process_wrapper,(chunkStart, chunkSize,  file_path, file_name)) )

#wait for all jobs to finish
for job in jobs:
    job.get()

#clean up
pool.close()


#CUDA_VISIBLE_DEVICES=1 python preprocess_syntax_corpus.py










