#!/bin/python3

import argparse
import os
import math
from statistics import mean
from collections import Counter
from scipy.stats import sem
import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit,cross_validate
from sklearn.metrics import classification_report,matthews_corrcoef,accuracy_score,multilabel_confusion_matrix


class Gor:

    def __init__ (self,training_path,dssp_path,test_path,cv):
        self.ss=['H', 'E', 'C','R']
        self.residue=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        self.w=17
        self.gor_dict = {
        'H' : np.zeros((self.w,20)),
        'E' : np.zeros((self.w,20)),
        'C' : np.zeros((self.w,20)),
        'R' : np.zeros((self.w,20))
        }

        self.ss_dict= {
        'H': 0,
        'E': 0,
        'C': 0
        }

        self.dssp_path=dssp_path
        self.training_path = training_path
        self.training_id=Gor.extract_id(training_path)
        if cv:
            self.test_path=training_path
            k_fold=Gor.get_k_fold_ids(cv,self.training_id)
            mcc=[]
            acc={'H':0,'E':0,'C':0}
            for k in range(len(k_fold.keys())):
                self.training_id=[]
                [self.training_id.extend(values) for key,values in k_fold.items() if key!=k]
                self.test_id=k_fold[k]
                self.gor_model= Gor.train(self)
                self.information_matrix=Gor.build_information_matrix(self)
                self.information_dict=Gor.to_dict(self.information_matrix)
                self.y_pred,self.y_true=Gor.prediction(self)
                prediction=Gor.performance(self)
                acc= dict(Counter(acc)+Counter(prediction[0]))
                mcc.append(prediction[1])
                classification_report=prediction[2]
                # output = (f"\nSplit{k}:\n"
                          # f"\nH:\tMCC:{round(prediction[1]['H'],6)}\taccuracy:{round(prediction[0]['H'],6)}\tprecision:{round(classification_report['H']['precision'],6)}\trecall:{round(classification_report['H']['recall'],6)}"
                          # f"\nE:\tMCC:{round(prediction[1]['E'],6)}\taccuracy:{round(prediction[0]['E'],6)}\tprecision:{round(classification_report['E']['precision'],6)}\trecall:{round(classification_report['E']['recall'],6)}"
                          # f"\nC:\tMCC:{round(prediction[1]['C'],6)}\taccuracy:{round(prediction[0]['C'],6)}\tprecision:{round(classification_report['C']['precision'],6)}\trecall:{round(classification_report['C']['recall'],6)}"
                          # )
            se=sem(mcc)
            print(mcc)
            print('mcc:',mean(mcc),'se:',se)
            #     # print(output)
            # mcc_str={k: v / 5 for k, v in mcc_str.items()}
            # acc={k: v / 5 for k, v in acc.items()}
            # mean_output=(
            #  f"\nH:\tMCC:{round(mcc_str['H'],6)}\taccuracy:{round(acc['H'],6)}"
            #  f"\nE:\tMCC:{round(mcc_str['E'],6)}\taccuracy:{round(acc['E'],6)}"
            #  f"\nC:\tMCC:{round(mcc_str['C'],6)}\taccuracy:{round(acc['C'],6)}"
            # )
            # print(mean_output)
        else:
            self.test_path=test_path
            self.gor_model= Gor.train(self)
            self.information_matrix=Gor.build_information_matrix(self)
            self.information_dict=Gor.to_dict(self.information_matrix)
            self.y_pred,self.y_true=Gor.prediction(self)
            prediction=Gor.performance(self,True)
            mcc= prediction[1]
            acc= prediction[0]
            classification_report=prediction[2]
            multiclass_conf_mat = multilabel_confusion_matrix(self.y_true,self.y_pred, labels=['H','E','C'])
            # total = np.sum(multiclass_conf_mat[0])
            # true_predicted = 0
            # for index in range(3):
            #     true_predicted += multiclass_conf_mat[index][1][1]
            # q3_score = true_predicted/total
            output = (f"Test set:"
            f"\nQ3:{round(acc,6) }"
            f"\nH:\tMCC:{round(mcc['H'],6)}\tprecision:{round(classification_report['H']['precision'],6)}\trecall:{round(classification_report['H']['recall'],6)}"
            f"\nE:\tMCC:{round(mcc['E'],6)}\tprecision:{round(classification_report['E']['precision'],6)}\trecall:{round(classification_report['E']['recall'],6)}"
            f"\nC:\tMCC:{round(mcc['C'],6)}\tprecision:{round(classification_report['C']['precision'],6)}\trecall:{round(classification_report['C']['recall'],6)}")
            print(multiclass_conf_mat)
            print(output)



    #########################################
    ##           utils function            ##
    #########################################

    @staticmethod
    def extract_id(path):
        id_list=[]
        for id in sorted(os.listdir(path)):
            id_list.append(id.split('.prf')[0].strip())
        return id_list

    @staticmethod
    def to_dict(file):
        dict=file.groupby(level=0).apply(lambda df: df.xs(df.name).to_numpy()).to_dict()
        return dict

    @staticmethod
    def get_acc(y_true,y_pred):
        ss=['H','E','C']
        ACC_vs_all={'H':0,'E':0,'C':0}
        for structure in ss:
            if structure== '-': structure='C'
            y_pred_tmp=[i if i==structure else 0 for i in y_pred]
            y_true_tmp=[i if i==structure else 0 for i in y_true]
            acc= accuracy_score(y_true_tmp,y_pred_tmp)
            ACC_vs_all[structure]+=acc
        return ACC_vs_all

    @staticmethod
    def get_mcc(y_pred,y_true,f=False):
        ss=['H','E','C']
        if f==True: MCC_vs_all={'H':0,'E':0,'C':0}
        else:MCC_vs_all=[]
        for structure in ss:
            y_pred_tmp=[i if i==structure else 0 for i in y_pred]
            y_true_tmp=[i if i==structure else 0 for i in y_true]
            mcc= matthews_corrcoef(y_true_tmp,y_pred_tmp)
            if f==True: MCC_vs_all[structure]+=mcc
            else: MCC_vs_all.append(mcc)
        if f==False: MCC_vs_all= mean(MCC_vs_all)
        return MCC_vs_all

    @staticmethod
    def get_y_true(path,ids):
        y_true=[]
        for id in ids:
            dssp=open(path+id+'.dssp').readlines()[1].strip()
            [y_true.append('C') if x=='-' else y_true.append(x) for x in dssp ]
        return y_true

    @staticmethod
    def get_k_fold_ids(path,ids):
        k_fold={}
        k=0
        for cv_set in sorted(os.listdir(path)):
            tmp=[]
            with open (path+cv_set) as file:
                [tmp.append(i.strip())for i in file.readlines() if i.strip() in ids]
                k_fold[k]=tmp
                k+=1
        return k_fold

#convert to a static method
    def to_df(self,gor_model):
        d=self.w//2
        window = range(-d,d+1)
        model_df = pd.DataFrame(gor_model,pd.MultiIndex.from_product([self.ss,window],names=['SS', 'PW']),self.residue)
        return model_df


    #########################################
    ##        training                     ##
    #########################################

    def train(self):

        '''
        write a comment for the function
        '''

        for id in self.training_id:
            with open(self.training_path+id+'.prf') as file:
                profile=pd.read_csv(file,sep='\s+',header=None).iloc[:,1:].values
                padding=np.zeros((self.w//2,len(profile[0])))
                profile=np.vstack((padding,profile,padding))
                dssp=open(self.dssp_path+id+'.dssp').readlines()[1].strip()
                for i in range(len(dssp)):
                    ss=dssp[i]
                    if ss== '-': ss='C'
                    self.gor_dict[ss]+=profile[i:i+self.w]
                    self.gor_dict['R']+=profile[i:i+self.w]
                    self.ss_dict[ss]+=1
        for row in range(len(self.gor_dict ['R'])):
            tot=sum(self.gor_dict ['R'][row])
            self.gor_dict ['H'][row]/=tot
            self.gor_dict ['E'][row]/=tot
            self.gor_dict ['C'][row]/=tot
            self.gor_dict ['R'][row]/=tot
        gor_model=np.vstack(list(self.gor_dict.values()))
        gor_model_df=Gor.to_df(self,gor_model)
        total=sum(self.ss_dict.values())
        self.ss_dict={k: v / total for k, v in self.ss_dict.items()}
        return gor_model_df

    #########################################
    ##        INFORMATION                  ##
    ##        MATRIX                       ##
    #########################################

    def build_information_matrix(self):

        '''
        p(SS,R) is the frequency of the type R observed in position k in the window whose central position is in configuration SS
        p(SS) is the frequency of each SS in the training set
        p(Rk) is the frequency of the residue R observed in the position k
        '''

        information_matrix= self.gor_model.copy()
        for index in information_matrix.index:
            for column in information_matrix:
                residue=column
                windows_position=index[1]
                s_structure=index[0]
                joint_probability_r_ss=self.gor_model.loc[index,column]
                marginal_residue=self.gor_model.loc[('R',windows_position),residue]
                if s_structure != 'R':
                    marginal_ss=self.ss_dict[s_structure]
                    information_matrix.loc[(s_structure,windows_position),residue] = math.log2(joint_probability_r_ss/(marginal_ss*marginal_residue))
    #    print_model(information_matrix,'/home/fabiana/Desktop/lab2/project/test/')
        return information_matrix.drop(index='R',level=0)

    #########################################
    ##        prediction                   ##
    #########################################

    def prediction(self):
        ids=Gor.extract_id(self.test_path)
        y_true=Gor.get_y_true(self.dssp_path,ids)
        predicted_sequence = []
        for id in ids:
            with open(self.test_path+id+'.prf') as file:
                profile=pd.read_csv(file,sep='\s+',header=None).iloc[:,1:].values
                padding=np.zeros((self.w//2,len(profile[0])))
                profile=np.vstack((padding,profile,padding))
                for i in range (0,(len(profile)-self.w+1)):
                    information_content={'H':0,'E':0,'C':0}
                    if np.sum(profile[i:i+self.w])==0:
                        predicted_sequence += 'C'
                    else:
                        information_content['H'] = np.sum(np.multiply(self.information_dict['H'],profile[i:i+self.w]))
                        information_content['E'] = np.sum(np.multiply(self.information_dict['E'],profile[i:i+self.w]))
                        information_content['C'] = np.sum(np.multiply(self.information_dict['C'],profile[i:i+self.w]))
                        predicted_ss = max(information_content,key=information_content.get)
                        predicted_sequence += predicted_ss
        return predicted_sequence,y_true

    def performance(self,f=False):
        acc=Gor.get_acc(self.y_true,self.y_pred) #one structure vs all
        q3=accuracy_score(self.y_true,self.y_pred)
        if f==True:  mcc=Gor.get_mcc(self.y_true,self.y_pred,True)  #one structure vs all
        else: mcc=Gor.get_mcc(self.y_true,self.y_pred)
        class_report=classification_report(self.y_true,self.y_pred,labels=['H','E','C'],output_dict=True)
        return q3,mcc,class_report


if __name__ == '__main__':

    parser= argparse.ArgumentParser()
    parser.add_argument("training",action='store',help="path to the training test set directory")
    parser.add_argument("testing",action='store',help="path to the testing directory")
    parser.add_argument("dssp",action='store',help="path to the dssp directory")
    parser.add_argument("--cv",action='store',default=None,help="Turn on the cross validation")
    args=parser.parse_args()
    print(args)
    gor= Gor(args.training,args.dssp,args.testing,args.cv)
