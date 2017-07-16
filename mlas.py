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
from sklearn.model_selection import KFold, cross_val_score

#Flows within flows.txt are in the following format:
#First line: first 3 values are total number of flows(x) and number of features recorded(n) and number of applications(m)
#Second Line: the next m values are strings denoting the different application labels.
#Subsequant Lines: The next line is the identifier of the flow (ip_addr, port, ip_addr, port, IP_proto)
#Subsequant Lines: For each flow, the first n numbers are features. After n doubles, a string denotes the application labels.
#EXAMPLE:
#1 5 2
#skype facebook
#1.0 3.4 5.0 2.3 12.0 facebook

#For SVM, X is array of size [num_samples, num_features]. Y is array of their labels size [num_samples]

#Read the data into arrays

f = open('flow_output.txt','r')

i = 0

for line in f:
    j = 0
    for s in line.split(' '):
        if i == 0:
            if j == 0:
                num_samples = int(s)
                j = j + 1
            elif j == 1:
                num_features = int(s)
                X = np.zeros((num_samples, num_features))
                y = []
                j = j + 1
            else:
                num_lables = int(s)
        elif i == 1 or i == 3:
            continue
        elif i % 2 == 1:
            if j < num_features:
                X[(i-4)/2][j] = float(s)    
                j = j + 1
            else:
                y.append(s.rstrip())
        else:
            continue
    i = i + 1

#Split data for training and testing
k_fold = KFold(n_splits=3)
all_results = []


svm_classifier = svm.SVC(gamma=0.001, C=100)
svm_classifier.fit(X,y)


for train, test in k_fold.split(X):
    
     train_array = X.array(map(X,test)).astype(np_float)
     train_label_array = y.array(map(y, test)).astype(np_float)
     self._classifier.fit(train_array, train_label_array)














