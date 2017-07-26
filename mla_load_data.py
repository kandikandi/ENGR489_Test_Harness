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
import random
from sklearn.model_selection import StratifiedKFold

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

class datasets:

    X = None
    y = None
    k_fold =None
    num_features = None
    num_samples = None
    num_classes = None
    classes = None

    def __init__(self,folds):
        """Initialise"""
        f = open('flow_output.txt','r')

        i = 0

        for line in f:
            j = 0
            for s in line.split(' '):
                if i == 0:
                    if j == 0:
                        self.num_samples = int(s)
                        j = j + 1
                    elif j == 1:
                        self.num_features = int(s)
                        self.X = np.zeros((self.num_samples, self.num_features))
                        self.y = np.empty([self.num_samples], dtype = "S16")
                        j = j + 1
                    else:
                        self.num_classes = int(s)
                        self.classes = np.empty([self.num_classes], dtype = "S16")
                elif i == 1:
                    if j < self.num_classes:
                        self.classes[j] = s.rstrip()
                        j = j + 1                
                elif i == 3:
                    continue
                elif i % 2 == 1:
                    if j < self.num_features:
                        self.X[(i-4)/2][j] = float(s)    
                        j = j + 1
                    else:
                        self.y[(i-4)/2] = s.rstrip()              
                else:
                    continue
            i = i + 1

        self.k_fold = StratifiedKFold(n_splits=folds)
        
    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_kFold(self):
        return self.k_fold  

    def get_num_features(self):
        return self.num_features

    def get_num_samples(self):
        return self.num_samples

    def get_num_classes(self):
        return self.num_classes


class results:
    
    folds = None
    params = None
    results = None
    labels = None
    dataset = None
    train_t = None
    test_t = None
    total_t = None
        
    
    def __init__(self,folds,params,results,labels,dataset,test_t,train_t):
        self.folds = folds
        self.params = params
        self.results = results
        self.labels = labels
        self.dataset = dataset
        self.train_t = train_t
        self.test_t = test_t
        self.total_t = train_t + test_t

        print_all_class()
        for i in range(0,dataset.get_num_classes()):
            print i

        
    
    def print_all_class(self):
        print "this is hard"


    
    def print_one_class(self,class_idx) :
        print "just one class"









