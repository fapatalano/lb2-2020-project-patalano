

'''
Svm() is a python3 class written to encode data. An object SVM is instantiated by giving in input a list of IDs.
Once a dataset, in form of dictionary either pickled or not, is uploaded via SVM.load() method, the dataset can be:
-   either encoded in the right format via SVM.encode() method and can be saved into a ??
    by using the SVM.save() method and y providing a path where save the file.
-   or with the use of SVM.decoding() method, by providing an outcome file of libSVM's predictions, the 1D secondary structure
    sequence is saved in dictionary['svm_pred']
'''

def extract_id(directory_path):
    id_list=[]
    for id in sorted(os.listdir(directory_path)):
        id_list.append(id.split('.p')[0])
    return id_list

def encode(dssp_path,profile_path,w):
    '''
    The encoding required by libSVM program is in the format:
    [0,1,2] 1:0. 2:0. ... 340:0.
    where 0,1,2 are the secondary structure states derived by the process of 8- to 3-state reduction;
    the next numbers come as consequence of the data transformation from a profile Lx20 window (matrix) where L is set
    by default as 17, to a vector of Lx20 dimensions.
    '''
    class_code = {'H': '0', 'E': '1', 'C': '2','-':'2'}
    id_list= extract_id(profile_path)
    Y=[]
    for id in id_list:
        with open(profile_path+id+'.prf') as file:
            profile=pd.read_csv(file,sep='\s+',header=None).iloc[:,1:].values
            padding= np.zeros((w//2,20))
            profile=np.vstack((padding,profile,padding))
            dssp=open(dssp_path+id+'.dssp').readlines()[1].strip()
            X=profile[0:w].flatten()
            for i in range(1,len(dssp)):
                Y+=class_code[dssp[i]]
                row=profile[i:i+w].flatten()
                X=np.vstack((X,row))
    return X,Y


#TO DO :
# grid search () --> {2,0.5}
# model.fit()
# Loading the dataset

def datasets_parse(profile_id,file):
    set=[]
    [set.append(i.strip()) for i in file.readlines() if i.strip() in profile_id]
    return set

def svc(profile_path,cv_path):
    profile_id=extract_id(profile_path)
    set={}
    i=0
    for cv_set in sorted(os.listdir(cv_path)):
        print(cv_set)
        with open (cv_path+cv_set) as file:
            set[i]=datasets_parse(profile_id,file)
            print(len(set[i]))
            i+=1


# #
# X= encode(dssp_path,profile_path,w)[0]
# y= encode(dssp_path,profile_path,w)[1]
#
# # Set the hyperparameters
# parameters = [{'kernel': ['rbf'], 'gamma': [0.5],
#                      'C': [2]}]
#

if __name__ == '__main__':

    import argparse
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV,PredefinedSplit
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC

    parser= argparse.ArgumentParser()
    parser.add_argument("--profile",help="path to the profile directory")
    parser.add_argument("--dssp",help="path to the dssp directory")
#    parser.add_argument("--w",help="window size", type=int)
    parser.add_argument("--cv",help="path to the cv set directory")
    args=parser.parse_args()
    svc= svc(args.profile,args.cv)

#    encode(args.dssp,args.profile,args.w)
