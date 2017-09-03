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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, precision_score, f1_score
import random
import time
import sys
from mla_load_data import datasets

if len(sys.argv) != 2:
    print "Please provide a data file"
    sys.exit()    

data = datasets(10,sys.argv[1])

f = open('naive_bayes.csv', 'w+')
f.write('test_num,num_features,scaled,accuracy,train_t,test_t\n')

x = 1

while x <= 5: #run tests 5 times
    X = data.get_X()
    y = data.get_y()
    j = data.get_num_features()

    while j > 0:

        X = SelectKBest(chi2, k=j).fit_transform(X,y)

        #Split data for training and testing
        k_fold = data.get_kFold()

        gnb_classifier = GaussianNB()
        all_acc = np.zeros(10)
        all_train = np.zeros(10)
        all_test = np.zeros(10)
        z = 0

        for train, test in k_fold.split(X,y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            before_time = time.time()
            gnb_classifier.fit(X_train,y_train)
            after_time = time.time()
            all_train[z] = (after_time - before_time)*1000 #convert to milliseconds    
            before_time = time.time()
            results = gnb_classifier.predict(X_test)
            after_time = time.time()
            all_test[z] = (after_time - before_time)*1000 #convert to milliseconds  
            all_acc[z] = accuracy_score(y_test, results)           

        f.write('{0},{1},{2},{3:.3f},{4:.3f},{5:.3f}\n'.format(x,j,'false',np.average(all_acc),np.average(all_train),np.average(all_test)))

        all_acc = np.zeros(10)
        all_train = np.zeros(10)
        all_test = np.zeros(10)
        z = 0

        for train, test in k_fold.split(X,y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            scaler = preprocessing.StandardScaler().fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)

            before_time = time.time()
            gnb_classifier.fit(X_train,y_train)
            after_time = time.time()
            all_train[z] = (after_time - before_time)*1000 #convert to milliseconds    
            before_time = time.time()
            results = gnb_classifier.predict(X_test)
            after_time = time.time()
            all_test[z] = (after_time - before_time)*1000 #convert to milliseconds  
            all_acc[z] = accuracy_score(y_test, results)           


        f.write('{0},{1},{2},{3:.3f},{4:.3f},{5:.3f}\n'.format(x,j,'true',np.average(all_acc),np.average(all_train),np.average(all_test)))

        j = j-3 #feature reduction

    x = x + 1 #increment test number
    
f.close()
    






    
