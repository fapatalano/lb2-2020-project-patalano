#!/bin/python3


import argparse
import os
from statistics import mean
import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, matthews_corrcoef, make_scorer, accuracy_score, multilabel_confusion_matrix
from thundersvm import SVC
import joblib


class svc():
    def __init__(self, training_path, test_path, dssp_path, cv_path, output_path):
        self.w = 17
        self.class_code = {'H': 1, 'E': 2, 'C': 3, '-': 3}
        self.training_path = training_path
        self.dssp_path = dssp_path
        self.training_id = svc.extract_id(training_path)
        self.cv_path = cv_path
        self.output_path = output_path
        self.test_fold = []
        self.cv_set = svc.get_k_fold_ids(cv_path, self.training_id)
        self.X_train, self.y_train = svc.encode(
            self, self.training_id, self.training_path, self.cv_path)
        performance = svc.grid_search(self)
        classification_report = prediction[0]
        mcc = prediction[1]
        q3_score = prediction[2]
        print(classification_report, mcc, q3_score)

    @staticmethod
    def extract_id(path):
        id_list = []
        for id in sorted(os.listdir(path)):
            # for id in sorted(open(path)):
            id_list.append(id.split('.prf')[0].strip())
        return id_list

    @staticmethod
    def get_k_fold_ids(cv_path, ids):
        k_fold = {}
        k = 0
        for cv_set in sorted(os.listdir(cv_path)):
            tmp = []
            with open(cv_path + cv_set) as file:
                [tmp.append(i.strip())
                 for i in file.readlines() if i.strip() in ids]
                k_fold[k] = tmp
                k += 1
        return k_fold

    @staticmethod
    def get_y_true(path, ids):
        class_code = {'H': 1, 'E': 2, 'C': 3, '-': 3}
        y_true = []
        for id in ids:
            dssp = open(path + id + '.dssp').readlines()[1].strip()
            [y_true.append(class_code[x]) for x in dssp]
        return y_true

    @staticmethod
    def get_mcc(y_pred, y_true, f=False):
        ss = [1, 2, 3]
        if f == True:
            MCC_vs_all = {1: 0, 2: 0, 3: 0}
        else:
            MCC_vs_all = []
        for structure in ss:
            y_pred_tmp = [i if i == structure else 0 for i in y_pred]
            y_true_tmp = [i if i == structure else 0 for i in y_true]
            mcc = matthews_corrcoef(y_true_tmp, y_pred_tmp)
            if f == True:
                MCC_vs_all[structure] += mcc
            else:
                MCC_vs_all.append(mcc)
        if f == False:
            MCC_vs_all = mean(MCC_vs_all)
        return MCC_vs_all

    @staticmethod
    def get_acc(y_pred, y_true):
        ss = [1, 2, 3]
        ACC_vs_all = {1: 0, 2: 0, 3: 0}
        for structure in ss:
            y_pred_tmp = [i if i == structure else 0 for i in y_pred]
            y_true_tmp = [i if i == structure else 0 for i in y_true]
            acc = accuracy_score(y_true_tmp, y_pred_tmp)
            ACC_vs_all[structure] += acc
        # ACC_vs_all= mean(ACC_vs_all)
        return ACC_vs_all

    def encode(self, ids, path, cv_path=None):
        '''
        The encoding required by scikitlearn module is in the format:
        '''
        X = []
        Y = []
        for id in ids:
            if cv_path:
                set = [key for key, value in self.cv_set.items() if id in value]
            with open(path + id + '.prf') as file:
                profile = pd.read_csv(
                    file, sep='\s+', header=None).iloc[:, 1:].values
                padding = np.zeros((self.w // 2, 20))
                profile = np.vstack((padding, profile, padding))
                dssp = open(self.dssp_path + id +
                            '.dssp').readlines()[1].strip()
                for i in range(0, len(dssp)):
                    if cv_path:
                        self.test_fold.append(set[0])
                    row = profile[i:i + self.w].flatten().tolist()
                    Y.append(self.class_code[dssp[i]])
                    X.append(row)
        X_array = np.array(X)
        print('finished encode ')
        return X_array, Y

    def grid_search(self):
        print("# Tuning hyper-parameters")
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.5, 2],
                             'C': [2, 4]}]
        model = SVC(random_state=42, gpu_id=0, cache_size=5000, verbose=True)
        mcc = make_scorer(svc.get_mcc)
        ps = PredefinedSplit(self.test_fold)
        clf = GridSearchCV(model,
                           tuned_parameters,
                           scoring=mcc,
                           cv=ps,
                           verbose=10,
                           n_jobs=1,
                           return_train_score=True,
                           ).fit(self.X_train, self.y_train)
        if self.output_path:
            joblib.dumb(clf.cv_results_, os.path.join(self.output_path, 'cv_results.joblib')
            joblib.dumb(clf.best_estimator_, os.path.join(self.output_path, 'best_estimator.joblib')
            joblib.dumb(clf.best_score_, os.path.join(self.output_path, 'best_score.joblib')

        best=clf.best_estimator_
        test_ids=svc.extract_id(test_path)
        X_test, y_test=svc.encode(self, test_ids, test_path)
        y_true=svc.get_y_true(self.dssp_path, test_ids)
        y_pred=best.predict(X_test)
        class_report=classification_report(y_true, y_pred,
                                             target_names=['H', 'E', 'C'],
                                             output_dict=True)
        mcc=svc.get_mcc(y_pred, y_true, True)
        # multiclass_conf_mat = multilabel_confusion_matrix(y_true,y_pred, labels=[1,2,3])
        q3_score=accuracy_score(y_true, y_pred)
        return class_report, mcc, q3_score


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument("training", action='store',
                        help="path to the training profile directory")
    parser.add_argument("dssp", action='store',
                        help="path to the dssp directory")
    parser.add_argument("testing", action='store',
                        help="path to the testing profile directory")
    parser.add_argument("cv", action='store',
                        help="path to the cv set directory")
    parser.add_argument("--o", action='store', default=None,
                        help="path to the output path")
    args=parser.parse_args()
    print(args)
    SVM=svc(args.training, args.testing, args.dssp, args.cv, args.o)
