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
import random
from mla_load_data import datasets

datasets_array = []

datasets_array.append(datasets(30))
datasets_array.append(datasets(20))
datasets_array.append(datasets(10))
datasets_array.append(datasets(5))

for data in datasets_array:
    print "##########################################"
    X = data.get_X()
    y = data.get_y()

    j = data.get_num_features()

    while j > 0:
        print "******************************************"
        print "num features = ", j
        X = SelectKBest(chi2, k=j).fit_transform(X,y)

        #Split data for training and testing
        k_fold = data.get_kFold()

        gnb_classifier = GaussianNB()


        for train, test in k_fold.split(X,y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            gnb_classifier.fit(X_train,y_train)
    
            results = gnb_classifier.predict(X_test)
            mislabeled = (y_test != results).sum()
            correct = (y_test == results).sum()
            print correct,len(y_test), int(float(correct)/len(y_test)*100),"%"

        j = j-3
    

    






    
