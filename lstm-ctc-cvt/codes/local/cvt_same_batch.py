
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import string
import time


# In[17]:


class BiGRU_CTC():
    
    def __init__(self,num_hidden = 200,num_layers = 2,num_features=40,initial_learning_rate = .001,
                reg_constant = 0.05,allow_growth_gpu=True,top_paths=1,beam_width=1):
        
       
        self.num_hidden=num_hidden;
        self.num_units=num_hidden;
        self.num_layers=num_layers;
        self.num_features=num_features;
        self.reg_constant = 0.05;
        self.initializer = tf.contrib.layers.xavier_initializer(seed=4);
        self.regularizer = tf.contrib.layers.l2_regularizer(scale = self.reg_constant);
        self.allow_growth_gpu=allow_growth_gpu;
        self.top_paths=top_paths;
        self.beam_width=beam_width;
        self.initial_learning_rate=initial_learning_rate;
        SPACE_TOKEN = '<space>'
        SPACE_INDEX = 0
        FIRST_INDEX = ord('A') - 1  # 0 is reserved to space ## order is for ascii value
        # Accounting the 0th indice +  space + blank label = 28 characters
        self.num_classes = ord('Z') - ord('A') + 1 + 1 + 1;
        #creating index to charactar mapping
        self.itoc={ v: k for k, v in zip(string.ascii_uppercase, range(1, len(string.ascii_uppercase)+1))}
        self.itoc[0]='<space>';
        ## as targets in ctc loss function goes without blank

        
       #######GRAPH BUILD
        self.config = tf.ConfigProto();
        self.config.gpu_options.allow_growth = self.allow_growth_gpu;
        # #config.gpu_options.per_process_gpu_memory_fraction = 0.90;




        self.graph = tf.Graph();
        with self.graph.as_default():
            #tf.reset_default_graph()
            self.inputs = tf.placeholder(tf.float32, [None, None,self.num_features])

            """Here we use sparse_placeholder that will generate a
            SparseTensor required by ctc_loss op. """

            self.targets = tf.sparse_placeholder(tf.int32)



            self.seq_len = tf.placeholder(tf.int32, [None])

            self.prob = tf.placeholder_with_default(1.0, shape=())

            #stacking LSTM layers(for multiple layers)
            self.output = self.inputs## for initial layers
            for n in range(self.num_layers):

                self.lstm_fw =tf.contrib.rnn.GRUCell(self.num_units)
                self.lstm_fw= tf.nn.rnn_cell.DropoutWrapper(cell=self.lstm_fw, input_keep_prob=self.prob)
                self.lstm_bw = tf.contrib.rnn.GRUCell(self.num_units)
                self.lstm_bw= tf.nn.rnn_cell.DropoutWrapper(cell=self.lstm_bw, input_keep_prob=self.prob)




                _initial_state_fw =self. lstm_fw.zero_state(tf.shape(self.inputs)[0], tf.float32)
                _initial_state_bw = self.lstm_bw.zero_state(tf.shape(self.inputs)[0], tf.float32)

                self.output, self._states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw, self.lstm_bw,self.output, 
                                                          initial_state_fw=_initial_state_fw,
                                                          initial_state_bw=_initial_state_bw, sequence_length =self. seq_len,
                                                          scope='BLSTM_'+str(n+1))


               #print(output[0].shape)
###############################
                self.output_fw=self.output[0];
                self.output_bw=self.output[1];


                self.output = tf.concat([self.output[0], self.output[1]],axis=2)
                
                

                self.outputs=self.output;


            
            shape1 = tf.shape(self.inputs)
            batch_s, max_timesteps = shape1[0], shape1[1] ### as inputs are being padded

            # Reshaping to apply the same weights over the timesteps
            self.outputs = tf.reshape(self.outputs, [-1,2*self.num_hidden])

            self.output_fw = tf.reshape(self.output_fw, [-1,self.num_hidden]);
            self.output_bw = tf.reshape(self.output_bw, [-1,self.num_hidden]);

            # print(self.output_bw.shape,self.output_fw.shape);
            # print(self.outputs.shape);


            # Initializing weights and bias for DNN
            self.W=tf.get_variable(name = "W", shape = [2*self.num_hidden,self.num_classes],initializer=self.initializer,regularizer=self.regularizer) 
            self.b=tf.get_variable(name = "b", shape = [self.num_classes],initializer=self.initializer,regularizer=self.regularizer) 

            



            self.logits = tf.matmul(self.outputs, self.W) + self.b;


            

            self.logits = tf.reshape(self.logits, [batch_s, -1, self.num_classes]) ## batch_size X no of frames X no of classes

                #print(logits.shape)

                # Time major is true by default
            self.logits = tf.transpose(self.logits, (1, 0, 2))## this means dim0 i  s now dim1 and so on
            



            # if(self.flag==1):
                # Reshaping back to the original shape
                

                #print(logits.shape)

                #CTC loss
            self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len) ##### SEE the sparse targets
            
            self.cost = tf.reduce_mean(self.loss)
            ## targets would be of 27X1 without blanks
            #regularization losses
            self.reg_losses_supervised = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) ###################################

            # final loss and optimization
            self.final_cost_supervised=self.cost+self.reg_constant*sum(self.reg_losses_supervised);
            


            self.optimizer_supervised = tf.train.MomentumOptimizer(self.initial_learning_rate,0.9).minimize(self.final_cost_supervised);   

            #decoding(we could also use tf.nn.ctc_greedy_decoder)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len,top_paths=self.top_paths,beam_width=self.beam_width); 
        



            self.cer = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), ## decoded[0] means top path pickup
                                                  self.targets))
            



            # else:


            self.W_fw=tf.get_variable(name = "W_fw", shape = [self.num_hidden,self.num_classes],initializer=self.initializer,regularizer=self.regularizer)
            self.b_fw=tf.get_variable(name = "b_fw", shape = [self.num_classes],initializer=self.initializer,regularizer=self.regularizer) ;

            self.W_bw=tf.get_variable(name = "W_bw", shape = [self.num_hidden,self.num_classes],initializer=self.initializer,regularizer=self.regularizer)
            self.b_bw=tf.get_variable(name = "b_bw", shape = [self.num_classes],initializer=self.initializer,regularizer=self.regularizer) ;



            self.logits_fw = tf.matmul(self.output_fw, self.W_fw) + self.b_fw;
            self.logits_bw = tf.matmul(self.output_bw, self.W_bw) + self.b_bw;




            self.logits_fw = tf.reshape(self.logits_fw, [batch_s, -1, self.num_classes]) ## batch_size X no of frames X no of classes

            self.logits_fw = tf.transpose(self.logits_fw, (1, 0, 2))## this means dim0 is now dim1 and so on

            self.logits_bw = tf.reshape(self.logits_bw, [batch_s, -1, self.num_classes]) ## batch_size X no of frames X no of classes

            
            self.logits_bw = tf.transpose(self.logits_bw, (1, 0, 2))## this means dim0 is now dim1 and so on



            self.gold_truth = tf.cast(self.decoded[0], tf.int32); ########################

            self.loss_unsupervised = tf.reduce_mean(tf.nn.ctc_loss(self.gold_truth, self.logits_fw, self.seq_len)+tf.nn.ctc_loss(self.gold_truth, self.logits_bw, self.seq_len));

            self.reg_losses_unsupervised = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)###########################keys


            self.final_cost_unsupervised=self.loss_unsupervised+self.reg_constant*sum(self.reg_losses_unsupervised);

            self.optimizer_unsupervised = tf.train.MomentumOptimizer(self.initial_learning_rate/50,0.9).minimize(self.final_cost_unsupervised);


            self.saver = tf.train.Saver();        


  
    #function to convert decoded list of indexes back to string 
    def convert(self,x):
        s=''
        for i in x:
            if(self.itoc[i]=='<space>'):
                s=s+' '
            else:
                s=s+self.itoc[i]

        return s;

    
    
    #Creating a sparse representention           
    def sparse_tuple_from(self,sequences,dtype=np.int32):

        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

        return indices, values, shape






    #function to return sequence lengths and pad the sequences to the maximum length(used from utils.py)
    def pad_sequences(self,sequences, maxlen=None, dtype=np.float32,
                      padding='post', truncating='post', value=0.):
        lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        #print(x,lengths)
        return x, lengths
    
    
    def model_train(self,labeled_tr_inputs,labeled_tr_targets,unlabeled_tr_inputs,test_inputs,test_targets,num_epochs,batch_size,re_train,
                    model_path_sav,model_path_res,log_file_name):

        num_examples = len(labeled_tr_inputs); ##################################################### supervised ######################################
        
        num_batches_per_epoch = int(num_examples/batch_size); 

        batch_size_unsup = int(len(unlabeled_tr_inputs)/num_batches_per_epoch); 

        with tf.Session(config=self.config,graph=self.graph) as session:



            if(re_train==True):
                self.saver.restore(session,model_path_res);
                print("Model restored from file: %s" % model_path_res);  ##Restoring model
            else :
                tf.global_variables_initializer().run();  # Initializate the weights and biases

            arr=[]
            #iterating over total number of epochs
            for curr_epoch in range(num_epochs):
                test_cer = 0
                sup_train_cost = 0
                unsup_train_cost = 0

                start = time.time()



                #iterating over total number of batchs
                for batch in range(num_batches_per_epoch):


                    # Getting the index
                    indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
                    batch_train_inputs = labeled_tr_inputs[indexes]

                    # Padding input to max_time_step of this batch
                    batch_train_inputs, batch_train_seq_len = self.pad_sequences(batch_train_inputs)
                    #print(batch_train_inputs.shape)

                    # Converting to sparse representation so as to to feed SparseTensor input
                    batch_train_targets = self.sparse_tuple_from(labeled_tr_targets[indexes])

                    # feed dictionay for training
                    feed = {self.inputs: batch_train_inputs,
                            self.targets: batch_train_targets,
                            self.seq_len: batch_train_seq_len,self.prob:0.6}

                    batch_cost, _ = session.run([self.final_cost_supervised, self.optimizer_supervised], feed)
                    sup_train_cost += batch_cost*batch_size


                    if curr_epoch>1:

                        indexes = [i % num_examples for i in range(batch * batch_size_unsup, (batch + 1) * batch_size_unsup)]
                        batch_train_inputs = unlabeled_tr_inputs[indexes]

                        # Padding input to max_time_step of this batch
                        batch_train_inputs, batch_train_seq_len = self.pad_sequences(batch_train_inputs);
                        #print(batch_train_inputs.shape)

                        # Converting to sparse representation so as to to feed SparseTensor input
                        # batch_train_targets = self.sparse_tuple_from(unlabeled_tr_targets[indexes])

                        # feed dictionay for training
                        
                        ind=np.random.choice(len(batch_train_inputs),batch_size,replace=False);

                        feed = {self.inputs: batch_train_inputs[ind],
                                
                                self.seq_len: batch_train_seq_len[ind],self.prob:.6}

                        batch_cost, _ = session.run([self.final_cost_unsupervised, self.optimizer_unsupervised], feed)
                        unsup_train_cost+= batch_cost*batch_size_unsup







                # Shuffle the labeled data
                shuffled_indexes = np.random.permutation(num_examples)## shuffling data for next epoch
                labeled_tr_inputs = labeled_tr_inputs[shuffled_indexes]
                labeled_tr_targets = labeled_tr_targets[shuffled_indexes]


                ###### Shuffled unlabeled data #######################
                shuffled_indexes = np.random.permutation(len(unlabeled_tr_inputs));## shuffling data for next epoch
                unlabeled_tr_inputs = unlabeled_tr_inputs[shuffled_indexes];
                


                # Metrics mean
                sup_train_cost /= num_examples
                unsup_train_cost/=len(unlabeled_tr_inputs);

                if(curr_epoch%1==0 and sup_train_cost<350 ):
                    batch_test_inputs, batch_test_seq_len = self.pad_sequences(test_inputs)
                    batch_test_targets = self.sparse_tuple_from(test_targets)
                    # feed dictionary for test data to calculate test CER
                    feed_test = {self.inputs: batch_test_inputs,
                                self.targets: batch_test_targets,
                                self.seq_len: batch_test_seq_len}
                    test_cer = session.run(self.cer,feed_test)


                log = "Epoch {}/{}, sup-train_cost = {:.3f},unsup-train_cost = {:.3f}, test_cer = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, num_epochs, sup_train_cost, unsup_train_cost,test_cer, time.time() - start))
                log_file = open(log_file_name,"a");
                log_file.write(log.format(curr_epoch+1, num_epochs, sup_train_cost, unsup_train_cost,test_cer, time.time() - start));
                log_file.write('\n');
                log_file.close();

                #arr.append(log.format(curr_epoch+1, num_epochs, train_cost, test_cer, time.time() - start))
            #np.save('heir2',arr)


            ## for model saving
                if(curr_epoch%5==0):
                    self.saver.save(session, model_path_sav);
                    print("Model saved in path: %s" % model_path_sav);

                if(curr_epoch==250):
                    self.saver.save(session, './model_cvt_same_batch_ssp_sup_local/model.ckpt');
                    print("Model saved in path: %s" % './model_cvt_same_batch_ssp_sup_local/model.ckpt');



    def model_score_ul(self,trn_inputs,path,batch_=100):

        batch_size=batch_;
        num_examples = len(trn_inputs)
        num_batches_per_epoch = int(num_examples/batch_size)

        with tf.Session(config=self.config,graph=self.graph) as session:
            ## restoring model
            self.saver.restore(session,path);
            print("Model restored from file: %s" % path);  ##Restoring model

            scores=[]
            for batch in range(num_batches_per_epoch):
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_train_inputs = trn_inputs[indexes]

                    # Padding input to max_time_step of this batch
                #print('padding');
                batch_train_inputs, batch_train_seq_len = self.pad_sequences(batch_train_inputs);
                #print('padding done');
                    #print(batch_train_inputs.shape)

                # feed dictionay for training
                #print('calculating prob');
                feed = {self.inputs: batch_train_inputs,
                        self.seq_len: batch_train_seq_len}

                scores += session.run([self.log_prob], feed)
                #print('prob calculated');
        print('Scoring Done');
        scores=np.array(scores);
        scores=scores.reshape(scores.shape[0]*scores.shape[1],scores.shape[2]);
        return scores;

    def model_cer(self,test_inputs,test_targets,path):

        with tf.Session(config=self.config,graph=self.graph) as session:
            ## restoring model
          
            self.saver.restore(session,path);
            print("Model restored from file: %s" % path);  ##Restoring model
            batch_test_inputs, batch_test_seq_len = self.pad_sequences(test_inputs);
            batch_test_targets = self.sparse_tuple_from(test_targets);
            # feed dictionary for test data to calculate test CER
            feed_test = {self.inputs: batch_test_inputs,
                        self.targets: batch_test_targets,
                        self.seq_len: batch_test_seq_len}
            test_cer = session.run(self.cer,feed_test);

        return test_cer;
    
    
    def convert(self,x):
        s=''
        for i in x:
            if(self.itoc[i]=='<space>'):
                s=s+' '
            else:
                s=s+self.itoc[i]

        return s;
    
    

    def model_decode(self,test_inputs,test_targets,path):

        with tf.Session(config=self.config,graph=self.graph) as session:
            ## restoring model
            self.saver.restore(session,path);
            print("Model restored from file: %s" % path);  ##Restoring model

            batch_test_inputs, batch_test_seq_len = self.pad_sequences(test_inputs);

            # Converting to sparse representation so as to to feed SparseTensor input
            batch_test_targets = self.sparse_tuple_from(test_targets);

            feed_test = {self.inputs: batch_test_inputs,
                    self.targets: batch_test_targets,
                    self.seq_len: batch_test_seq_len
                    }

            # Decoding
            d = session.run(self.decoded[0],feed_test)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)
            #saver = tf.train.Saver()

            org_list=[];
            dec_list=[];

            for i, seq in enumerate(dense_decoded):

                seq = [s for s in seq if s != -1]

                #print('Sequence %d' % i)
                org_list.append(self.convert(test_targets[i]));
                dec_list.append(self.convert(seq));

                #print('\t Original: ', self.convert(test_targets[i]))
                #print('\t Decoded: ',self.convert(seq))
        return org_list,dec_list;
                
    
    def model_cer_batch(self,test_inputs,test_targets,path,batch_=100):

        batch_size=batch_;
        num_examples = len(test_inputs)
        num_batches_per_epoch = int(num_examples/batch_size)

        with tf.Session(config=self.config,graph=self.graph) as session:
            ## restoring model
            self.saver.restore(session,path);
            print("Model restored from file: %s" % path);  ##Restoring model

            cer_list=[]
            for batch in range(num_batches_per_epoch):
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_train_inputs,batch_train_targets = test_inputs[indexes],test_targets[indexes];

                    # Padding input to max_time_step of this batch
                #print('padding');
                batch_train_inputs, batch_train_seq_len = self.pad_sequences(batch_train_inputs);
                batch_train_targets = self.sparse_tuple_from(batch_train_targets);
                #print('padding done');
                    #print(batch_train_inputs.shape)

                # feed dictionay for training
                #print('calculating prob');
                feed = {self.inputs: batch_train_inputs,
                        self.targets: batch_train_targets,
                        self.seq_len: batch_train_seq_len}

                cer_list += session.run([self.cer], feed)
                #print('prob calculated');
        print('Cer calculated');
        cer_list=np.array(cer_list);
        #scores=scores.reshape(scores.shape[0]*scores.shape[1],scores.shape[2]);
        return cer_list;

    
    
    
    
    
    
    def wer_calc(self,seq1, seq2):  ## seq1 is orginial  
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix [x, 0] = x
        for y in range(size_y):
            matrix [0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix [x,y] = matrix[x-1, y-1];
                else:
                    matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
        #print (matrix)
        max_len=size_x-1;
        ans=matrix[size_x - 1, size_y - 1]
        return (ans/max_len);

    
    def model_wer(self,test_inputs,test_targets,path,batch_=100):
        
        batch_size=batch_;
        num_examples = len(test_inputs)
        num_batches_per_epoch = int(num_examples/batch_size)
        
        wer=0;
        
        print('kaise');
        for batch in range(num_batches_per_epoch):
            
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

            batch_train_inputs,batch_train_targets = test_inputs[indexes],test_targets[indexes];
            
            o,d=self.model_decode(batch_train_inputs,batch_train_targets,path);
            
            for i in range(len(o)):
                o_=o[i].split();
                d_=d[i].split();

                wer+=self.wer_calc(o_,d_)/len(o);
        
        return wer/(num_batches_per_epoch);
            
    
    
    
    def softmax(self,arr):
        arr=arr.reshape(-1,);
        arr2=np.exp(arr)/np.sum(np.exp(arr));
        return arr2;
    
    
        
        
    def model_logits(self,trn_inputs,path,batch_=100): ## function to calculate path probabilities brfore decoding

        batch_size=batch_;
        num_examples = len(trn_inputs)
        num_batches_per_epoch = int(num_examples/batch_size)

        with tf.Session(config=self.config,graph=self.graph) as session:
            
            self.saver.restore(session,path);
            
            print("Model restored from file: %s" % path);  ##Restoring model

            scores=[]; ## path_probability before decoding
            
            for batch in range(num_batches_per_epoch):
                
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_train_inputs = trn_inputs[indexes]

                
                batch_train_inputs, batch_train_seq_len = self.pad_sequences(batch_train_inputs);
                
                
                feed = {self.inputs: batch_train_inputs,
                        self.seq_len: batch_train_seq_len}

                temp= session.run(self.logits, feed);
                temp=np.transpose(temp,(1,0,2));
                
                list_=[]
                
                for i in range(len(temp)):
                    list_.append(temp[i,0:batch_train_seq_len[i],:])
                    #print(list_[i].shape)
                    
                for i in range(len(list_)):
                    for j in range(list_[i].shape[0]):
                        list_[i][j]=self.softmax(list_[i][j]);


                list_=[np.max(i,axis=1) for i in list_]
                #list_[0].shape


                path_prob=np.zeros(len(list_),)

                for i in range(len(list_)):
                    path_prob[i]=1
                    n=len(list_[i]);
                    for j in range(len(list_[i])):
                        path_prob[i]*= ((list_[i][j])**(1/n));

                scores.append(path_prob);
                
                
        scores=np.array(scores);
        scores=scores.reshape(scores.shape[0]*scores.shape[1],);
        
        return scores;

    
    
    
    