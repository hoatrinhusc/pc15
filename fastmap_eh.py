import fastmapsvm as fm
import h5py
import numpy as np
import sklearn.metrics
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from rdkit import Chem
import tmap as tm
from map4 import MAP4Calculator
from tmap import Minhash
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from functools import lru_cache


input_dir = './segments/'


labels= []
indices = []
objects = dict()

X_train = []
y_train = []
X_test = []
y_test = []

with open('pc15_w11.txt', 'r') as f:
    for idx, line in enumerate(f.readlines()):
        fields = line.strip('\n').split(',')
        objects[idx] = fields[3]
        X_train.append(idx)
        y_train.append(fields[4])

idx0 = idx + 1


with open('cb513_w11.txt', 'r') as f:
    for idx, line in enumerate(f.readlines()):
        fields = line.strip('\n').split(',')
        objects[idx0 + idx] = fields[3]
        X_test.append(idx0 + idx)
        y_test.append(fields[4])


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


#X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

eh_train_idx = np.where((y_train == 'H') | (y_train == 'C'))
#eh_test_idx = np.where((y_test == '-') | (y_test == 'H'))

X_eh_train = X_train[eh_train_idx]
y_eh_train = y_train[eh_train_idx]
print('y_eh_train', y_eh_train[:20])
#X_eh_test = X_test[eh_test_idx]
#y_eh_test = y_test[eh_test_idx]

le1 = preprocessing.LabelEncoder()
le1.fit(y_eh_train)
y_eh_train = le1.transform(y_eh_train)
print('y_eh_train', y_eh_train[:20])

#le2 = preprocessing.LabelEncoder()
#le2.fit(y_eh_test)

#y_eh_test = le2.transform(y_eh_test)

print('# test', len(y_test), '# train', len(y_eh_train))

'''

with open('ml_data_w11.txt', 'r') as f:
    for idx, line in enumerate(f.readlines()):
        fields = line.strip('\n').split(',')
        #print(fields)
        
        if fields[4] == 'E':
            labels.append(0)
            indices.append(idx)
            objects[idx] = fields[3]

        if fields[4] == 'H':
            labels.append(1)
            indices.append(idx)
            objects[idx] = fields[3]

print(len(labels))

X_train, X_test, y_train, y_test = train_test_split(np.array(indices), np.array(labels), test_size=0.5, random_state=42)
'''
hf = h5py.File('w11.hdf5', 'w')
g1 = hf.create_group('X')
g1.create_dataset('test', data=X_test)
g1.create_dataset('train', data=X_eh_train)

g2 = hf.create_group('y')
#g2.create_dataset('test', data=y_test)
g2.create_dataset('train', data=y_eh_train)

# IMPORTANT: need to close file, otherwise cannot read with hdfview
hf.close()


#@lru_cache(maxsize=None)

def correlation_distance(idx1, idx2): # mol_1 and mol_2 are converted from FASTA file
    #print('objects', smiles_1, smiles_2)
    dim = 1024
    MAP4 = MAP4Calculator(dimensions=dim)
    ENC = Minhash(dim)
    #ENC = tm.Minhash(dim)
    mol_1 = Chem.MolFromSmiles(objects[idx1])
    #map4_1 = MAP4.calculate(mol_1)
    mol_2 = Chem.MolFromSmiles(objects[idx2])
    #map4_2 = MAP4.calculate(mol_2)
    fps = MAP4.calculate_many([mol_1, mol_2])

    #return ENC.get_distance(map4_1, map4_2)
    return ENC.get_distance(fps[0], fps[1])



model_path = "./w11_eh_d150_tsz02.hdf5"
data_path  = "w11.hdf5"


with h5py.File(data_path, mode="r") as f5:
    clf = fm.FastMapSVM(
        correlation_distance,
        200,
        model_path
    )
    clf.fit(f5["/X/train"], f5["/y/train"])
    
    #y_true = f5["/y/test"][:]
    #proba = clf.predict_proba(f5["/X/test"])
    print('predicting')
    y_pred = clf.predict(f5["/X/test"])
    
    #print('test accuracy score', accuracy_score(y_true, y_pred))


    #y_true = f5["/y/train"][:]
    #proba = clf.predict_proba(f5["/X/test"])
    #y_pred = clf.predict(f5["/X/train"])

    np.savetxt('pred_ch.txt', y_pred, fmt='%d')

    #print('train accuracy score', accuracy_score(y_true, y_pred))


