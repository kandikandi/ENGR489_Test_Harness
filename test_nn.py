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
from sklearn.neural_network import MLPClassifier
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

data = datasets(10,sys.argv[1])
np.set_printoptions(threshold=np.inf)

#f = open('neural_net.csv', 'w+')
#f.write('test_num,num_features,num_hidden_layers,num_neurons_per_layer,scaled,accuracy,train_t,test_t\n')

X = data.get_X()
y = data.get_y()
j = data.get_num_features()
k=20 #number of hidden layers

j=4

while j > 0:
    X = SelectKBest(chi2, k=j).fit_transform(X,y)        
    #Split data for training and testing
    k_fold = data.get_kFold()               



    hls = k*10 #number of neurons in hidden layer, beta is a scaling value
    layers = []
    l=0

    while l < k:
        layers.append(hls)
        l = l + 1
       
    nn_classifier = MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
  
    for train, test in k_fold.split(X,y):
        print("*\n")
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        nn_classifier.fit(X_train,y_train)
        print("**\n")
        results = nn_classifier.predict(X_test)
        print(y_test)
        print(results)        
       
                        
    j = j-3
    
f.close()



