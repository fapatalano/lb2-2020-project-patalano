#!/bin/python3


if __name__ == '__main__':

    import argparse
    from sklearn.metrics import multilabel_confusion_matrix
    import pandas as pd
    import numpy as np
    import argparse


    x,y = compute_similarity(args.dssp, args.SSpredictions)
    multiclass_conf_mat = multilabel_confusion_matrix(x,y, labels=['H','E','C'])
    total = np.sum(multiclass_conf_mat[0])
    print('\nHelix one-vs-all:\n', multiclass_conf_mat[0], '\n\nStrand one-vs-all:\n', multiclass_conf_mat[1], '\n\nCoil one-vs-all:\n', multiclass_conf_mat[2] )
    ss = ['H', 'E', 'C']
    true_predicted = 0
    dictionary = {'H': None, 'E': None, 'C': None}
    for index in range(3):
        true_predicted += multiclass_conf_mat[index][1][1]
        MCC, ACC, TPR, PPV, FPR, NPV = print_performance(multiclass_conf_mat[index])
        dictionary[ss[index]] = [MCC, ACC, TPR, PPV, FPR, NPV]
    d = pd.DataFrame(data=dictionary, index=['MCC', 'ACC', 'TPR', 'PPV', 'FPR', 'NPV'], dtype=float)

    q3_score = true_predicted/total
    print(d)
    print('\nThis is the Q3 score:\t%f\n' %q3_score)




        def encode(self):
            '''
            The encoding required by scikitlearn module is in the format:
            '''
            X=[]
            Y_train=[]
            for id in self.training_id:
                if self.cv_path: set=[key for key, value in self.cv_set.items() if id in value]
                with open(self.training_path+id+'.prf') as file:
                    profile=pd.read_csv(file,sep='\s+',header=None).iloc[:,1:].values
                    padding= np.zeros((self.w//2,20))
                    profile=np.vstack((padding,profile,padding))
                    dssp=open(self.dssp_path+id+'.dssp').readlines()[1].strip()
                    for i in range(0,len(dssp)):
                        if self.cv_path: self.test_fold.append(set[0])
                        row=profile[i:i+self.w].flatten().tolist()
                        Y_train.append(self.class_code[dssp[i]])
                        X.append(row)
            X_train=np.array(X)
            print('finished encode ')
            return X_train,Y_train
