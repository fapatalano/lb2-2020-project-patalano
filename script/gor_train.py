#!/bin/python3

## the output should be a matrix that has 20 col and w*4 rows . the script should be run as follow
# -- input training.txt --data profile_data/ --output gor_model.txt
# the order of aa in the profile is: {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}

# TO DO:
# change in order to pass as argument an id_list <- important for the cv, you may also do a function to extract the id
# find a way to obtain ss and to not have the values as constant variables
# add class and try to improve your script (OOP)

# convert in a class:
    # extract id --> tool

def extract_id(path):

    id_list=[]
    for id in sorted(os.listdir(path)):
        id_list.append(id.split('.p')[0])
    return id_list

def to_df(model,w):

    d=w//2
    gor_columns=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    ss = ['H', 'E', 'C','R']
    window = range(-d,d+1)
    gor_df = pd.DataFrame(model,pd.MultiIndex.from_product([ss,window],names=['R_SS', 'PW']),gor_columns)

    return gor_df

def print_model(model,dir):
    model.to_csv(path_or_buf=dir+"information_matrix_df.tsv", sep='\t')
    #model.to_csv(path_or_buf=dir+"gor_df.tsv", sep='\t')
    #np.savetxt(dir+"gor_model.tsv", model, delimiter="\t")

def build_gor_model(dssp_path,profile_path,w):

    '''
    write a comment for the function
    '''

    gor_dict = {
    'H' : np.zeros((w,20)),
    'E' : np.zeros((w,20)),
    'C' : np.zeros((w,20)),
    'R' : np.zeros((w,20))
    }

    ss_dict= {
    'H': 0,
    'E': 0,
    'C': 0
    }

    id_list=extract_id(profile_path)

    for id in id_list:
        with open(profile_path+id+'.prf') as file:
            profile=pd.read_csv(file,sep='\s+',header=None).iloc[:,1:].values
            padding=np.zeros((w//2,len(profile[0])))
            profile=np.vstack((padding,profile,padding))
            dssp=open(dssp_path+id+'.dssp').readlines()[1].strip()
            for i in range(len(dssp)):
                ss=dssp[i]
                if ss== '-': ss='C'
                gor_dict[ss]+=profile[i:i+w]
                gor_dict['R']+=profile[i:i+w]
                ss_dict[ss]+=1
    for row in range(len(gor_dict ['R'])):
        tot=sum(gor_dict ['R'][row])
        gor_dict ['H'][row]/=tot
        gor_dict ['E'][row]/=tot
        gor_dict ['C'][row]/=tot
        gor_dict ['R'][row]/=tot
    gor_model=np.vstack(list(gor_dict.values()))
    gor_model_df= to_df(gor_model,w)
    total=sum(ss_dict.values())
    ss_dict={k: v / total for k, v in ss_dict.items()}
    return gor_model_df,ss_dict

def build_information_matrix(gor_model,ss_vector):

    '''
    p(SS,R) is the frequency of the type R observed in position k in the window whose central position is in configuration SS
    p(SS) is the frequency of each SS in the training set
    p(Rk) is the frequency of the residue R observed in the position k
    '''

    information_matrix= gor_model.copy()
    for index in information_matrix.index:
        for column in information_matrix:
            residue=column
            windows_position=index[1]
            s_structure=index[0]
            joint_probability_r_ss=gor_model.loc[index,column]
            marginal_residue=gor_model.loc[('R',windows_position),residue]
            if s_structure != 'R':
                marginal_ss=ss_vector[s_structure]
                information_matrix.loc[(s_structure,windows_position),residue] = math.log2(joint_probability_r_ss/(marginal_ss*marginal_residue))
#    print_model(information_matrix,'/home/fabiana/Desktop/lab2/project/test/')
    return information_matrix.drop(index='R',level=0)

    def to_dict(df):
    dict=df.groupby(level=0).apply(lambda df: df.xs(df.name).to_numpy()).to_dict()
    return dict

if __name__ == '__main__':

    import argparse
    import os
    import pandas as pd
    import numpy as np
    import math

    parser= argparse.ArgumentParser()
    parser.add_argument("--profile",help="path to the profile directory")
    parser.add_argument("--dssp",help="path to the dssp directory")
    parser.add_argument("--w",help="window size", type=int)
    args=parser.parse_args()
    gor_model,ss_vector=build_gor_model(args.dssp,args.profile,args.w)
    #print_gor_model(gor_model,'/home/fabiana/Desktop/lab2/project/test/')
    information_matrix_df=build_information_matrix(gor_model,ss_vector)
    information_matrix_dict=to_dict(information_matrix_df)
