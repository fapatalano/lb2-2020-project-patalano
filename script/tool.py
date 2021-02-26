#!/bin/python3
from typing import List

import pandas as pd
import warnings
import os
from tqdm import tqdm
from collections.abc import Iterable
from sklearn.metrics import matthews_corrcoef, accuracy_score
from statistics import mean
from Bio.Blast.Applications import NcbipsiblastCommandline, NcbimakeblastdbCommandline
from tempfile import TemporaryDirectory, NamedTemporaryFile
from joblib import dump
from pathlib import Path

residue = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
ss=['H', 'E', 'C', 'R']

def save_tsv(model, dir, out):
    model.to_csv(path_or_buf=dir + out, sep='\t')

def convert_to_df(data_structure,w=17):
    d = w // 2
    window = range(-d, d + 1)
    model_df: DataFrame = pd.DataFrame(data_structure,
                            pd.MultiIndex.from_product([ss, window], names=['Secondary_Structure', 'PW']),
                            residue)
    return model_df

def convert_df_to_dict(df):
    '''
    DOC to do
    '''
    dict = df.groupby(level=0).apply(lambda df: df.xs(df.name).to_numpy()).to_dict()
    return dict


def extract_id(path,ext):
    id_list: List[str]=[]
    for id in sorted(os.listdir(path)):
        id_list.append(id.split(ext)[0])
    return id_list

def check_len(file,min_len,max_len):
    for seq_record in SeqIO.parse(file, "fasta"):
        sequence = str(seq_record.seq)
        if len(sequence) < self.min_len or len(sequence) > self.max_len:    return seq_record.id

def check_upper_case(file):
    for seq_record in SeqIO.parse(file, "fasta"):
        sequence = str(seq_record.seq)
        if sequence.isupper() == False:    return sequence

def file_with_X_remover(file):
    for record in SeqIO.parse(file, "fasta"):
        if record.seq.count('X') == 0: return record.format("fasta")


def random_extractor(directory):
    with open(directory) as file:
        seqs = SeqIO.parse(f, "fasta")
        samples = ((seq.name, seq.seq) for seq in sample(list(seqs), 2))
        for sample in samples:
            print(">{}\n{}".format(*sample))

