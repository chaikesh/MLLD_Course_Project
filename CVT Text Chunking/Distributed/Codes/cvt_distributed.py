# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run training and evaluation for CVT text models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from base import configure
from base import utils
from training import trainer
from training import training_progress


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', '"train" or "eval')
tf.app.flags.DEFINE_string('model_name', 'default_model',
                           'A name identifying the model being '
                           'trained/evaluated')
tf.app.flags.DEFINE_string("job_name","","'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index",0,"Index of task within the job")

#python cvt.py --mode=train --model_name=chunking_model --job_name="ps" --task_index=0
#python cvt.py --mode=train --model_name=chunking_model --job_name="worker" --task_index=0
#python cvt.py --mode=train --model_name=chunking_model --job_name="worker" --task_index=1
#python asp.py --job_name="ps" --task_index=0
def main():
  
  utils.heading('SETUP')
  
  
  parameter_servers=["10.1.1.254:2265"];# 0,4 ,3,1
  workers=["10.1.1.249:2263","10.1.1.251:2264" ,"10.1.1.252:2268"]  ##251 correspond to node 3 253-1 254-0 252-2
  cluster = tf.train.ClusterSpec({"ps":parameter_servers,"worker":workers})

 
  os.environ['OMP_NUM_THREADS'] = '2'
  config_env=tf.ConfigProto();
  #config_env.gpu_options.allow_growth=True
  config_env.allow_soft_placement=True
  config_env.log_device_placement=True

  server=tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index,config=config_env)
  config = configure.Config(mode=FLAGS.mode, model_name=FLAGS.model_name)
  config.write()
  
  
  if FLAGS.job_name=='ps':
    server.join()
  elif FLAGS.job_name=='worker':
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
      
      model_trainer = trainer.Trainer(config,FLAGS.task_index)
      summary_writer = tf.summary.FileWriter(config.summaries_dir)
      checkpoints_saver = tf.train.Saver(max_to_keep=1)
      best_model_saver = tf.train.Saver(max_to_keep=1)
      init = tf.global_variables_initializer()
      global_step= model_trainer._model._global_step
      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),global_step=global_step,init_op=init)
    with sv.prepare_or_wait_for_session(server.target) as sess:
      progress = training_progress.TrainingProgress(
          config, sess, checkpoints_saver, best_model_saver,FLAGS.task_index,
          config.mode == 'train')
      utils.log()
      if config.mode == 'train':
        utils.heading('START TRAINING ({:})'.format(config.model_name))
        model_trainer.train(sess, progress, summary_writer)
      elif config.mode == 'eval':
        utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
            config.checkpoints_dir))
        model_trainer.evaluate_all_tasks(sess, summary_writer, None)
      else:
        raise ValueError('Mode must be "train" or "eval"')

  sv.stop()
if __name__ == '__main__':
  main()
