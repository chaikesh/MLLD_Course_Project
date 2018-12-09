For distributed implementation, first codes could be obtained from https://github.com/tensorflow/models/tree/master/research/cvt_text. Note that, distributed mode is not implemented in above repository.
Now to run distributed, unsupervised data should be sharded manually.
Replace above files in the original code as per the mode to be operated in.
Run preprocessing_distributed.py uploaded here and then follow the normal procedure.
Above codes implement 3 worker nodes and 1 parameter servers.
