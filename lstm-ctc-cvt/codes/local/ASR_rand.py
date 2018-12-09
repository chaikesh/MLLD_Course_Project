
# coding: utf-8

# In[1]:


from BiGRU_CTC_class import BiGRU_CTC
import numpy as np
import scipy.signal as signal
from os import path
import random
import string
import h5py
import pandas as pd
import time
from sklearn.datasets import load_files
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[2]:


import os 
os.environ['CUDA_VISIBLE_DEVICES']="1";


# In[3]:


def data_extraction():    ### num_features are 13*3=39 mfcc unnormalised
    
    train_file="/home/karanm/librispeech_100hrs/processed_data/train.txt"; ## full data train
    train_x=[];
    train_y=[];
    with open(train_file) as f: 
        for l in f: 
            l=l.strip()
            train_filename='/home/karanm/librispeech_100hrs/processed_data/mfccs_39_un/'+l+'.npy'
            target_filename='/home/karanm/librispeech_100hrs/processed_data/targets/'+l+'.npy'
            train_x.append(np.load(train_filename))
            train_y.append(np.load(target_filename))

    test_file="/home/karanm/librispeech_100hrs/processed_data/test.txt";
    test_x=[];
    test_y=[];
    with open(test_file) as f: 
        for l in f: 
            l=l.strip()
            train_filename='/home/karanm/librispeech_100hrs/processed_data/mfccs_39_un/'+l+'.npy'
            target_filename='/home/karanm/librispeech_100hrs/processed_data/targets/'+l+'.npy'
            test_x.append(np.load(train_filename))
            test_y.append(np.load(target_filename))
    return train_x,train_y,test_x,test_y;


# In[4]:


train_x,train_y,test_x,test_y=data_extraction();
print(len(train_x),len(test_x));


# In[5]:


num_features=train_x[0].shape[1];


# In[ ]:


# # Calculating mean and variance and saving them once done
# train_x_norm=np.zeros((0,num_features));
# train_y_norm=np.zeros((0,num_features));

# for array in train_x:
#     train_x_norm=np.concatenate((train_x_norm,array),axis=0);
# mean=np.mean(train_x_norm,axis=0);
# # std=np.std(train_x_norm,axis=0);

# np.save(mean_path,mean);
# np.save(std_path,std);


# In[6]:


def normalise_data(data_train,data_test):
    mean_path='/home/karanm/librispeech_100hrs/processed_data/mean.npy';
    std_path='/home/karanm/librispeech_100hrs/processed_data/std.npy';
    mean,std=np.load(mean_path),np.load(std_path);
    
    for i in range(len(data_train)):
        data_train[i]= (data_train[i]-mean)/std;
        
    for i in range(len(data_test)):
        data_test[i]= (data_test[i]-mean)/std; ## Training data mean used for test data also
    
    print('Data Normalised');    
    return data_train,data_test;    


# In[7]:


train_x,test_x=normalise_data(train_x,test_x);


# In[30]:


### LSTM CTC object
initial_model_path='/home/karanm/models/libri_100_int_rand/model.ckpt';
uncer_model_path='/home/karanm/models/libri_100_unc/model.ckpt';
rand_model_path='/home/karanm/models/libri_100_rand/model.ckpt';
temp_path='/home/karanm/models/temp/model.ckpt';

ASR=BiGRU_CTC(num_features=num_features,initial_learning_rate=0.001,beam_width=20,top_paths=20); ## ASR object created 


# # Utility Functions

# In[9]:


def labeled_unlabeled_data(train_x,train_y,n_labeled):  ## function to break data into labeled and unlabeled data 
    np.random.seed(1);
    index=np.random.choice(len(train_x),n_labeled,replace=False);
    index_u=range(len(train_x));
    index_u=[i for i in index_u if i not in index];
    #print(len(index),len(index_u));
    train_x_label,train_y_label=np.array(train_x)[index],np.array(train_y)[index];
    train_x_unlabel,train_y_unlabel=np.array(train_x)[index_u],np.array(train_y)[index_u];
    return train_x_label,train_y_label,train_x_unlabel,train_y_unlabel;


## Function to normalising scores of top paths
def normalised_scores(sc):
   # sc=sc.reshape(sc.shape[1]*sc.shape[0],sc.shape[2]);
    sc_scaled=np.zeros((sc.shape[0],sc.shape[1]));
    
    for i in range(sc.shape[0]):
        sc_scaled[i]=sc[i]-np.floor(np.min(sc[i]));
    
    sc=np.copy(sc_scaled);
    
    norm_scores=np.zeros((sc.shape[0],sc.shape[1]))
    for i in range(len(sc)):
        for j in range(len(sc[i])):
            norm_scores[i,j]= sc[i,j]/np.sum(sc[i]);
    return norm_scores[:,0];


# In[31]:


n=int(len(train_x));
x_l,y_l,x_ul,y_ul=labeled_unlabeled_data(train_x,train_y,n_labeled=n);
print(x_l.shape,x_ul.shape)
x_l_r,y_l_r,x_ul_r,y_ul_r=np.copy(x_l),np.copy(y_l),np.copy(x_ul),np.copy(y_ul);


# # Initial Model training 

# In[32]:


ASR.model_train(tr_inputs=x_l,tr_targets=y_l,
                test_inputs=np.array(test_x),test_targets=np.array(test_y),batch_size=100,
                num_epochs=370,model_path_sav=initial_model_path,model_path_res=initial_model_path,re_train=False);

