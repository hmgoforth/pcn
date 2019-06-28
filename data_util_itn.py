import numpy as np
import tensorflow as tf
from tensorpack import dataflow

from transform_util import random_rotation, euldeg2rotm, rotm2quat, quat2rotm

from pdb import set_trace as st

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, sample_size, is_training, train_perturb_list=None, valid_perturb_list=None, so3_perturb=False, use_partial=False):
        super(PreprocessData, self).__init__(ds)
        self.sample_size = sample_size
        self.is_training = is_training
        self.so3_perturb = so3_perturb
        self.use_partial = use_partial

        if valid_perturb_list is not None:
            self.valid_perturb_list = np.loadtxt(valid_perturb_list)
        else:
            self.valid_perturb_list = None

        if train_perturb_list is not None:
            self.train_perturb_list = np.loadtxt(train_perturb_list)
            
            if len(self.train_perturb_list.shape) == 1:
                self.train_perturb_list = np.expand_dims(self.train_perturb_list, axis=0)
        else:
            self.train_perturb_list = None

    def get_data(self):
        itr = 0
        for iden, partial, complete in self.ds.get_data():
            partial_centroid = np.mean(partial, axis=0)

            if self.use_partial:
                itn_input = partial
            else:
                itn_input = resample_pcd(complete, self.sample_size)

            if self.is_training:
                if self.train_perturb_list is not None:
                    R = quat2rotm(self.train_perturb_list[itr])
                else:
                    R = random_rotation(self.so3_perturb)
            else:
                R = quat2rotm(self.valid_perturb_list[itr])
                itr += 1
            
            itn_input = itn_input @ R.T
            itn_input = itn_input - partial_centroid
            gt_transformation = np.eye(4)
            gt_transformation[0:3, 0:3] = R
            gt_transformation[0:3, 3] = -partial_centroid
            yield iden, itn_input, gt_transformation

def lmdb_dataflow(lmdb_path, batch_size, sample_size, is_training, test_speed=False, train_perturb_list=None, valid_perturb_list=None, so3_perturb=False, use_partial=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)
    df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = PreprocessData(df, sample_size, is_training, train_perturb_list=train_perturb_list, valid_perturb_list=valid_perturb_list, so3_perturb=so3_perturb, use_partial=use_partial)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.BatchData(df, batch_size, use_list=True)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size

def get_queued_data(generator, dtypes, shapes, queue_capacity=10):
    assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'
    queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)
    placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=True)
    feed_fn = lambda: {placeholder: value for placeholder, value in zip(placeholders, next(generator))}
    queue_runner = tf.contrib.training.FeedingQueueRunner(queue, [enqueue_op], close_op, feed_fns=[feed_fn])
    tf.train.add_queue_runner(queue_runner)
    return queue.dequeue()