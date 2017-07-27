  # Copyright 2017 Kandice McLean
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
from mla_load_data import datasets
from sklearn.metrics import accuracy_score, precision_score, f1_score
import time
import sys

if len(sys.argv) != 2:
    print "Please provide a data file"
    sys.exit() 

datasets_array = []

datasets_array.append(datasets(30,sys.argv[1]))
datasets_array.append(datasets(20,sys.argv[1]))
datasets_array.append(datasets(10,sys.argv[1]))
datasets_array.append(datasets(5,sys.argv[1]))

f = open('svm.csv', 'w+')
f.write('svm\nfolds,num_features,gamma,C,accuracy,precision_micro,precision_macro,f1_micro,f1_macro,train_t,test_t,total_t\n')

for data in datasets_array:
    X = data.get_X()
    y = data.get_y()

    j = data.get_num_features()

    while j > 0:
        X = SelectKBest(chi2, k=j).fit_transform(X,y)

        #Split data for training and testing
        k_fold = data.get_kFold()

        for k in range(-10, 1) :
            k1 = k-10
            svm_classifier = svm.SVC(kernel = 'rbf', gamma= 2**k1 , C= 2**k, cache_size=500,  decision_function_shape='ovr', class_weight = 'balanced')

            for train, test in k_fold.split(X,y):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                before_time = time.time()
                svm_classifier.fit(X_test,y_test)
                after_time = time.time()
                train_t = (after_time - before_time)*1000 #convert to milliseconds   
                before_time = time.time()
                results = svm_classifier.predict(X_train)
                after_time = time.time()
                test_t = (after_time - before_time)*1000 #convert to milliseconds  
                f.write('{0},{1},{2},{3},{4},{5:.3f},{6:.3f},{7:.3f},{8:.3f},{9:.3f},{10:.3f},{11:.3f}\n'.format(k_fold.get_n_splits(),j,2**k1,2**k,accuracy_score(y_train, results),precision_score(y_train, results,  average='micro'),precision_score(y_train, results,  average='macro'),f1_score(y_train,results, average='micro'),f1_score(y_train,results, average='macro'),train_t,test_t,train_t+test_t))
         
     
        j = j-3
    
f.close()





    
