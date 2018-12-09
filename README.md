# MLLD_Course_Project
##Abstract
Cross View Training(CVT) is a semi supervised learning technique that has shown improvements for several NLP tasks. For labeled examples, standard supervised learning is performed and for unlabeled examples, CVT teaches the auxiliary prediction modules that see only restricted views of the input to match the prediction of the primary module which has the full input view. Since our approach requires distributing CVT, so to better understand it we first implement CVT for text chunking task in a distributed setting. Recent LSTM-CTC  based end-to-end ASR(Automatic Speech Recogniton) have  been shown to work extremely well when there is an abundance of supervised training data, matching and exceeding the performance of hybrid DNN systems. But, labeling speech utterances is very expensive and time consuming task. So, we propose a method to apply CVT with LSTM-CTC based ASR model to leverage unlabeled raw speech data and train a robust ASR with limited labeled speech data. As this approach requires large amount of unlabeled data, it necessitates training of model in  distributed settings. For this, we explore different settings of distributed training for the above mentioned tasks.
