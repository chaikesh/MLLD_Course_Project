
# coding: utf-8

# In[1]:


from cvt_same_batch import BiGRU_CTC
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
os.environ['CUDA_VISIBLE_DEVICES']="0";


# In[3]:


# ASR=BiGRU_CTC(num_features=39,initial_learning_rate=0.001,beam_width=20,top_paths=3); ## ASR object created 

# exit();

# ##### input flags ################

# tf.app.flags.DEFINE_string("job_name","","'ps' / 'worker'");
# tf.app.flags.DEFINE_integer("task_index",0,"Index of task within the job");
# FLAGS=tf.app.flags.FLAGS;




def data_extraction():    ### num_features are 13*3=39 mfcc unnormalised
    

    # if flags.task_index==0:   ################### DATA SHARDING ################################################################################################
    train_file_l="/home/karanm/mlld/project/lstm-ctc_final/data_lists/labeled.txt"; 
    train_file_u="/home/karanm/mlld/project/lstm-ctc_final/data_lists/unlabeled.txt";


    train_x_l=[];
    train_y_l=[];
    with open(train_file_l) as f: 
        for l in f: 
            l=l.strip()
            train_filename='/home/karanm/librispeech_150hrs/processed_data/mfccs_39_un/'+l+'.npy'
            target_filename='/home/karanm/librispeech_150hrs/processed_data/targets/'+l+'.npy'
            train_x_l.append(np.load(train_filename))
            train_y_l.append(np.load(target_filename))


    train_x_u=[];
    train_y_u=[];
    with open(train_file_u) as f: 
        for l in f: 
            l=l.strip()
            train_filename='/home/karanm/librispeech_150hrs/processed_data/mfccs_39_un/'+l+'.npy'
            target_filename='/home/karanm/librispeech_150hrs/processed_data/targets/'+l+'.npy'
            train_x_u.append(np.load(train_filename))
            train_y_u.append(np.load(target_filename))



    test_file="/home/karanm/mlld/project/lstm-ctc_final/data_lists/test.txt";
    test_x=[];
    test_y=[];
    with open(test_file) as f: 
        for l in f: 
            l=l.strip()
            train_filename='/home/karanm/librispeech_test_clean/processed_data/mfccs_39_un/'+l+'.npy'
            target_filename='/home/karanm/librispeech_test_clean/processed_data/targets/'+l+'.npy'
            test_x.append(np.load(train_filename))
            test_y.append(np.load(target_filename))
    return train_x_l,train_y_l,train_x_u,train_y_u,test_x,test_y;


# In[4]:


train_x_l,train_y_l,train_x_u,train_y_u,test_x,test_y=data_extraction();


print('Data Extracted');
print(len(train_x_l),len(train_x_u),len(test_x));


# In[5]:


num_features=train_x_l[0].shape[1];



def normalise_data(data_train_l,data_train_u,data_test):
    mean_path='/home/karanm/librispeech_150hrs/processed_data/mean.npy';
    std_path='/home/karanm/librispeech_150hrs/processed_data/std.npy';
    mean,std=np.load(mean_path),np.load(std_path);
    
    for i in range(len(data_train_l)):
        data_train_l[i]= (data_train_l[i]-mean)/std;
        
    for i in range(len(data_train_u)):
        data_train_u[i]= (data_train_u[i]-mean)/std;
    

    for i in range(len(data_test)):
        data_test[i]= (data_test[i]-mean)/std; ## Training data mean used for test data also
    
    print('Data Normalised');    
    return data_train_l,data_train_u,data_test;    


# In[7]:


train_x_l,train_x_u,test_x=normalise_data(train_x_l,train_x_u,test_x);


# In[30]:







# # Utility Functions

# In[9]:





# In[31]:

x_l,y_l=np.array(train_x_l),np.array(train_y_l);
x_u=np.array(train_x_u);


# x_test,y_test=np.array(test),np.array(train_y_l);


# # Initial Model training 

# In[32]:



### LSTM CTC object
# initial_model_path='./model_cvt/model.ckpt';
initial_model_path='./model_cvt_same_batch/model.ckpt'
log_file_path='./local_cvt_out2.out'

ASR=BiGRU_CTC(num_features=num_features,initial_learning_rate=0.001,beam_width=20,top_paths=3); ## ASR object created 







ASR.model_train(labeled_tr_inputs=x_l,labeled_tr_targets=y_l,unlabeled_tr_inputs=x_u,
                test_inputs=np.array(test_x[0:1000]),test_targets=np.array(test_y[0:1000]),batch_size=100,
                num_epochs=700,model_path_sav=initial_model_path,model_path_res=initial_model_path,re_train=True,log_file_name=log_file_path);


