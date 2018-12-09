
These are Scripts to run the distributed mode for LSTM-CTC based ASR in local mode with CVT settings and without cvt settings.

Change the number of workers and ip add of the nodes according to the need .


Bulk Synchronous Parallel mode

on ps node : python ASR_cvt_same_batch_bsp.py --job_name="ps" --task_index=0

on worker0 node: python ASR_cvt_same_batch_bsp.py --job_name="worker" --task_index=0

on worker 1 node: python ASR_cvt_same_batch_bsp.py --job_name="worker" --task_index=1



Asynchronous Parallel mode

on ps node : python ASR_cvt_same_batch_asp.py --job_name="ps" --task_index=0

on worker0 node: python ASR_cvt_same_batch_asp.py --job_name="worker" --task_index=0

on worker 1 node: python ASR_cvt_same_batch_asp.py --job_name="worker" --task_index=1




Stale Synchronus Parallel mode(change stale value as required)

on ps node : python ASR_cvt_same_batch_ssp.py --stale=05 --job_name="ps" --task_index=0

on worker0 node: python ASR_cvt_same_batch_ssp.py --stale=05 --job_name="worker" --task_index=0

on worker 1 node: python ASR_cvt_same_batch_ssp.py --stale=05 --job_name="worker" --task_index=1



