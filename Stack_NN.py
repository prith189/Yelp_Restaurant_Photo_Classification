# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 10:45:48 2016

@author: prith189
"""


import numpy as np

import pandas as pd

np.random.seed(1377)


import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adadelta, Adagrad, Adam,RMSprop
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
import sklearn

def np_thresh(x1,thresh):
    
    x1 = x1-thresh
    
    x1 = x1>0
    
    x1 = np.array(x1,dtype=int)
    
    return x1
    

def fscore(x1,x2):

    tru_pos  = np.sum(np.array((np.logical_and(x1,x2)),dtype=int))

    actual_pos = np.sum(x1)
    
    pred_pos = np.sum(x2)
    if(pred_pos>0):
        precision = float(tru_pos)/pred_pos
    else:
        precision = 0
    
    if(actual_pos>0):
        recall = float(tru_pos)/actual_pos
    else:
        recall=0
    
    #print(precision,recall)
    if(precision+recall>0):
        fscore = (2*precision*recall)/(precision+recall)
    else:
        fscore = 0
    
    #print(fscore)   
    
    return [precision,recall,fscore]

#Train.csv
def clean_train():
    train = pd.read_csv('train.csv')
    
    # Are there any businesses without labels
    
    pd.isnull(train['labels'])
    for i in range(9):
        col_name = 'label_'+str(i)
        
        train[col_name] = train['labels'].apply(lambda x: 1 if (str(i) in str(x)) else 0)
    
    train = train.drop(['labels'],axis=1)
    
    train.to_csv('train_cl.csv',index=0)

def get_net(num_feat,num_out):
    clf = Sequential()
    clf.add(Dense(input_dim=num_feat, output_dim=65))
    clf.add(PReLU())
    clf.add(Dense(input_dim = 65, output_dim = 35))
    clf.add(PReLU())
    clf.add(Dense(input_dim = 35, output_dim=num_out, activation='sigmoid'))
    clf.compile(optimizer=Adam(), loss='binary_crossentropy')
    return clf

start_time = time.time()

#train_photos_to_biz_ids

train_to_biz = pd.read_csv('train_photo_to_biz_ids.csv')

uni_bus = train_to_biz['business_id'].unique()

blend1 = np.load('Model1_Full.npy')

blend2 = np.load('Model2_Full_2.npy')

blend3 = np.load('Model2_Full_3.npy')

blend4 = np.load('Model2_Full_4.npy')

blend5 = np.load('Model2_Full_5.npy')

x1 = np.hstack((blend1,blend4,blend2,blend3,blend5))

x1 = pd.DataFrame(x1)

x1['business_id'] = uni_bus

print('time taken for cleaning',time.time()-start_time)

# Start the learning process

print('Starting learning')

cv = 1

submit = 0

num_cv = 5

from sklearn.cross_validation import KFold

skf = list(KFold(x1.shape[0],num_cv,random_state=42))

dataset_blend_train = np.zeros([x1.shape[0],9])

labels = ['label_'+str(i) for i in range(9)]

iter_label = {}

for nb,lb in enumerate(labels):
    iter_n = 0
    train_cl = pd.read_csv('train_cl.csv')

    train_cl = dict(np.array(train_cl[['business_id',lb]]))

    x1[lb] = x1['business_id'].apply(lambda x: train_cl[x])

other_cols = ['business_id','label_0','label_1','label_2','label_3','label_4','label_5','label_6','label_7','label_8']

if(cv):
    for (training,testing) in skf:
        
        df_train = x1.iloc[training]
        
        df_test = x1.iloc[testing]
        
        df_train_values = np.array(df_train[labels])
        
        df_train_features = np.array(df_train.drop(other_cols,axis=1))
        
        df_test_values = np.array(df_test[labels])
        
        df_test_features = np.array(df_test.drop(other_cols,axis=1))
        
        df_train = None
        
        df_test = None
        
        bst = get_net(df_train_features.shape[1],df_train_values.shape[1])
        
        bst.fit(df_train_features,df_train_values,batch_size=30,nb_epoch=26,shuffle=True,validation_data=(df_test_features,df_test_values),callbacks=[EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')])

        preds = bst.predict(df_test_features) 
        
        dataset_blend_train[testing,:] = preds
        
        
        df_train_features = None
        
        df_test_features = None
        
        xg_train = None
        
        xg_test = None
    
        iter_label[lb] = int(float(iter_n)/num_cv)

# calculate precision

np.save('Stack_NN.npy',dataset_blend_train)

fsc = np.zeros(dataset_blend_train.shape[0])

cr_e = np.zeros(dataset_blend_train.shape[1])

truth = np.array(x1[labels])

dataset_blend_train2 = np.zeros(dataset_blend_train.shape)

f1_thresh = [0.43]*9

for f1 in range(9):

    dataset_blend_train2[:,f1] = np_thresh(dataset_blend_train[:,f1],f1_thresh[f1])

for i in range(dataset_blend_train2.shape[0]):
    
    fsc[i] = (fscore(truth[i,:],dataset_blend_train2[i,:]))[2]

for i in range(dataset_blend_train.shape[1]):
    
    cr_e[i] = (log_loss(truth[:,i],dataset_blend_train[:,i]))

print('fscore',np.mean(fsc))

print('time taken for learning',time.time()-start_time)
# train on the entire dataset
if(submit):
    
    train_to_biz = pd.read_csv('train_photo_to_biz_ids.csv')
    
    uni_bus = train_to_biz['business_id'].unique()
    
    blend1 = np.load('Model1_Full.npy')
    
    blend2 = np.load('Model2_Full_2.npy')
    
    blend3 = np.load('Model2_Full_3.npy')
    
    blend4 = np.load('Model2_Full_4.npy')
    
    blend5 = np.load('Model2_Full_5.npy')
    
    x1 = np.hstack((blend1,blend4,blend2,blend3,blend5))

    x1 = pd.DataFrame(x1)
    
    x1['business_id'] = uni_bus
    
    for nb,lb in enumerate(labels):
        train_cl = pd.read_csv('train_cl.csv')
    
        train_cl = dict(np.array(train_cl[['business_id',lb]]))
    
        x1[lb] = x1['business_id'].apply(lambda x: train_cl[x])
            
        
    df_train_values = np.array(x1[labels])
    
    df_train_features = np.array(x1.drop(other_cols,axis=1))

    bst = get_net(df_train_features.shape[1],df_train_values.shape[1])
        
    bst.fit(df_train_features,df_train_values,batch_size=30,nb_epoch=26)
    
    df_train_features = None
    
    df_test_features = None
    
    xg_train = None
    
    # Predict on the test set
    
    test_to_biz = pd.read_csv('test_photo_to_biz.csv')
    
    uni_bus = test_to_biz['business_id'].unique()
        
    blend1 = np.load('Model1_Full_result.npy')
    
    blend2 = np.load('Model2_Full_2_result.npy')
    
    blend3 = np.load('Model2_Full_3_result.npy')
    
    blend4 = np.load('Model2_Full_4_result.npy')
    
    blend5 = np.load('Model2_Full_5_result.npy')
    
    x1 = np.hstack((blend1,blend4,blend2,blend3,blend5))  
    
    x1 = pd.DataFrame(x1)
    
    x1['business_id'] = uni_bus
    
    result = np.zeros([x1.shape[0],9])

    df_test_features = np.array(x1.drop(['business_id'],axis=1))

    result = bst.predict((df_test_features))

    result2 = np.zeros(result.shape)
    
    for f1 in range(9):
    
        result2[:,f1] = np_thresh(result[:,f1],f1_thresh[f1])

    
    bid = np.array(x1['business_id'])
    
    fin = {}
    
    for i in range(result2.shape[0]):
        x = result2[i,:]
        li = [((q)) for q in range(9) if x[q]==1]
        fin[bid[i]] = li
        
    for j in fin.keys():
        fin[j] = ' '.join(str(e) for e in fin[j])
    
    x1 = pd.DataFrame(x1['business_id'])
    
    x1['labels'] = x1['business_id'].apply(lambda x: fin[x] if x in fin.keys() else '0')
    
    x1.to_csv('result.csv',index=0)





    
    