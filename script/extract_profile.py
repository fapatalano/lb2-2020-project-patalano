#!/bin/python3

import sys

def normalize_values(list):
    normalized_list=[]
    for value in list:
        normalized_list.append(str(int(value)/100))
    return normalized_list

def parse_pssm(pssm_file):
    with open(pssm_file) as file:
        next(file)
        next(file)
        next(file)
        for line in file:
            line=line.strip().split()
            if len(line)!=0 and line[0].isnumeric() :
                print('\t'.join(line[:2])+'\t'+'\t'.join(normalize_values(line[22:-2]))+'\n')



if __name__ == '__main__':
    pssm_file= sys.argv[1]
    parse_pssm(pssm_file)
