#!/bin/python3

from Bio import SeqIO
import os
import pandas as pd
import numpy as np
from random import sample

def __init__ (self,w=17):
    self.w=w
    self.ss=['H', 'E', 'C','R']
    self.residue=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']



    def save_tsv(model,dir='../test'):
        # model.to_csv(path_or_buf=dir+"information_matrix_df.tsv", sep='\t')
        #model.to_csv(path_or_buf=dir+"gor_df.tsv", sep='\t')
        np.savetxt(dir+"x_train.tsv", model, delimiter="\t")
        
    def to_df(self,structure_to_convert):

        d=self.w//2
        window = range(-d,d+1)
        model_df = pd.DataFrame(structure_to_convert,pd.MultiIndex.from_product([self.ss,window],names=['Secondary_Structure', 'PW']),self.residue)
        return model_df

class Dictionary:

        def to_dict(df):
            dict=df.groupby(level=0).apply(lambda df: df.xs(df.name).to_numpy()).to_dict()
            return dict

class Parse:

    def __init__(self,path):

        self.path=path
        self.extensions= { 'profile':'.prf',
                           'dssp':'.dssp',
                           'fasta':'.fasta'
                        }

        self.max_len=300
        self.min_len=50


    def extract_id(self,ext):
        id_list=[]
        for id in sorted(os.listdir(self.path)):
            id_list.append(id.split(self.extensions[ext])[0])
        return id_list

    def check_len(self,file):
        for seq_record in SeqIO.parse(file, "fasta"):
            sequence=str(seq_record.seq)
            if len(sequence)<self.min_len or len(sequence)>self.max_len :    print(seq_record.id)

    def check_upper_case (file):
        for seq_record in SeqIO.parse(file, "fasta"):
            sequence=str(seq_record.seq)
            if sequence.isupper()==False:    print(sequence)

    def X_remover(file):
        for record in SeqIO.parse(file, "fasta"):
            if record.seq.count('X') == 0: print(record.format("fasta"))

    def random_extractor(directory):

        '''
        this script needs to randomly extract some sequences
        '''

        with open(directory) as file:
            seqs = SeqIO.parse(f, "fasta")
            samples = ((seq.name, seq.seq) for seq in  sample(list(seqs),2))
            for sample in samples:
                print(">{}\n{}".format(*sample))

    # def a function:
    #     wanted = [line.strip() for line in open(sys.argv[2])]
    #     seqiter = SeqIO.parse(open(sys.argv[1]), 'fasta')
    #     for seq in seqiter:
    #         if seq.id[:6] in wanted:
    #             print('>'+seq.id[:6])
    #             print(seq.seq)

class Save:
    def __init__(self,output_path):
        self.output_path=output_path


    def save_tsv(model,dir):
        model.to_csv(path_or_buf=dir+"information_matrix_df.tsv", sep='\t')
        #model.to_csv(path_or_buf=dir+"gor_df.tsv", sep='\t')
        #np.savetxt(dir+"gor_model.tsv", model, delimiter="\t")

        def save_model(self,name):
            filename = os.path.join(self.output_path, name+'.joblib')
            joblib.dump(self.model, filename)
