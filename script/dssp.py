#!/usr/bin/python3
import sys

structure = {'T':'C', 'S':'C','':'C',
'H':'H','G':'H','I':'H',
'E':'E','B':'E'}

def get_ss(dsspfile_oligo):
    file=open(dsspfile_oligo)
    i=0
    seq=''
    s_structure=''
    dssp=[]
    chains=[]
    for line in file:
        if line.find('  #  RESIDUE')>-1:            #-1 means not found
            i=1
            continue
        if i==0 or len(line)<115:continue
        aa=line[13]
        ss=line[16]
        if aa!='!':
            seq+=aa
            if ss=='T' or ss=='S' or ss==' ': s_structure+='C'
            elif ss=='H' or ss=='G' or ss=='I' : s_structure+='H'
            elif ss=='B' or ss=='E': s_structure+='E'
            else : print('error', ss, 'not found')
        elif aa=='!':
            s_structure+='C'
            seq+='X'
            # dssp.append(seq)
            # dssp.append(s_structure)
            # seq=''
            # s_structure=''
    dssp.append(seq)
    dssp.append(s_structure)
    print('>'+dsspfile_oligo[12:-5])
    print(seq)
    # print(s_structure)
    return dssp


# if you apply this program to protein data bank data sets containing oligomers,
# solvent exposure is for the entire assembly, not for the monomer

def get_ACC(file):
    i=0
    ASA=[]
    z=[]
    for line in file:
        if line.find('  #  RESIDUE')>-1:            #-1 means not found
            i=1
            continue
        if i==0 or len(line)<115:continue
        aa=line[13]
        acc=float(line[35:38])
        if aa!='!':   z.append(acc)
        elif aa=='!':
            ASA.append(z)
            z=[]
    ASA.append(z)
    return ASA

def get_ASA(dsspfile_oligo,dsspfile_mono_1,dsspfile_mono_2):          #accessible surface area
    oligo=open(dsspfile_oligo)
    mono_1=open(dsspfile_mono_1)
    mono_2=open(dsspfile_mono_2)
    mono_Acc_1=get_ACC(mono_1)
    mono_Acc_2=get_ACC(mono_2)
    oligo_Acc=get_ACC(oligo)
    a=0
    b=0
    # tot_mon=sum(mono_Acc_1[0])+sum(mono_Acc_2[0])
    # tot_oligo=sum(oligo_Acc[0])+sum(oligo_Acc[1])
    for i in range (len(mono_Acc_1[0])):
        if mono_Acc_1[0][i]!= oligo_Acc[0][i]:
            a+=1
            print('Chain A')
            print(i,mono_Acc_1[0][i],oligo_Acc[0][i])
        if mono_Acc_2[0][i]!= oligo_Acc[1][i]:
            print('Chain B')
            b+=1
            print(i,mono_Acc_2[0][i],oligo_Acc[1][i])

if __name__=='__main__':
    dsspfile_oligo=sys.argv[1]
    # dsspfile_mono_1=sys.argv[2]
    # dsspfile_mono_2=sys.argv[3]
    dssp=get_ss(dsspfile_oligo)
    # for i in range(0,len(dssp),2):
    #     print('Sequence:','\n',dssp[i],'\n','Secondary structure:','\n',dssp[i+1])
    # asa=get_ASA(dsspfile_oligo,dsspfile_mono_1,dsspfile_mono_2)
