import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sympy import false

train_file = pd.read_csv("train.txt", header=None)
test_file = pd.read_csv("test.txt", header=None)

X_train_domain = train_file.values[0:40000, 0]
label_dga = train_file.values[0:40000, -1]
Y = np.zeros(len(label_dga))
X_test_domain = test_file.values[0:-1, 0]

sel = ['y', 'p', 'b', 'v', 'k', 'x', 'j', 'q', 'z']

X_train_features = ['length', 'vowel/consonants', 'numbers/letters', 'entropy', 'seldomseen']
for i in range(0, len(X_train_domain)):
    if label_dga[i] == "dga":
        Y[i] = 1
    j = 0
    templist = list()
    vowel = 0
    consonant = 0
    num = 0
    selnum = 0
    ch = X_train_domain[i][0]
    templist.append(len(X_train_domain[i]))
    while ch != '.' and j < len(X_train_domain[i]):
        if ch in ['a','e','i','o','u']:
            vowel = vowel + 1
        elif ch in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            num = num + 1
        else:
            consonant = consonant + 1
        if ch in sel:
            selnum = selnum + 1
        j = j + 1
        ch = X_train_domain[i][j]
    if ch == '.':
        t = vowel + consonant + num + 0.0000001
        templist.append(vowel / (consonant + 0.0000001))
        templist.append(num / (vowel + consonant + 0.0000001))
        if t:
            en = - (vowel + 0.0000001) / t * math.log((vowel + 0.0000001) / t, 2) - (consonant + 0.0000001) / t * math.log((consonant + 0.0000001) / t, 2) - (num + 0.0000001) / t * math.log((num + 0.0000001) / t, 2)
            templist.append(en)
        else:
            templist.append(0)
        templist.append(selnum / len(X_train_domain[i]))
        X_train_features = np.vstack((X_train_features, templist))
    elif j == len(X_train_domain[i]):
        X_train_features = np.vstack((X_train_features, [0, 0, 0, 0, 0]))

X_test_features = ['length', 'vowel/consonants', 'numbers/letters', 'entropy', 'seldomseen']
for i in range(0, len(X_test_domain)):
    j = 0
    templist = list()
    vowel = 0
    consonant = 0
    num = 0
    selnum = 0
    ch = X_test_domain[i][0]
    templist.append(len(X_test_domain[i]))
    while ch != '.' and j < len(X_test_domain[i]):
        if ch in ['a','e','i','o','u']:
            vowel = vowel + 1
        elif ch in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            num = num + 1
        else:
            consonant = consonant + 1
        if ch in sel:
            selnum = selnum + 1
        j = j + 1
        ch = X_test_domain[i][j]
    if ch == '.':
        t = vowel + consonant + num + 0.0000001
        templist.append(vowel / (consonant + 0.0000001))
        templist.append(num / (vowel + consonant + 0.0000001))
        if t:
            en = - (vowel + 0.0000001) / t * math.log((vowel + 0.0000001) / t, 2) - (consonant + 0.0000001) / t * math.log((consonant + 0.0000001) / t, 2) - (num + 0.0000001) / t * math.log((num + 0.0000001) / t, 2)
            templist.append(en)
        else:
            templist.append(0)
        templist.append(selnum / len(X_test_domain[i]))
        X_test_features = np.vstack((X_test_features, templist))
    elif j == len(X_test_domain[i]):
        X_test_features = np.vstack((X_test_features, [0, 0, 0, 0, 0]))

kNN_classifier = KNeighborsClassifier(weights="distance", n_neighbors=6)
kNN_classifier.fit(X_train_features[1:40001, :], Y)
res_KNN = kNN_classifier.predict(X_test_features[1:])

# print(res_KNN)
res = []

for i in range(0, len(res_KNN)):
    if res_KNN[i] == 1:
        res.append('dga')
    else:
        res.append('notdga')

output = np.vstack((X_test_domain, res))
df = pd.DataFrame(output.T)
df.to_csv('output.txt', sep=',', header=0, index=false, encoding='utf-8')