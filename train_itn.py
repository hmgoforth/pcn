'''
Train IT-network on a completion dataset, where each pair is a partial and complete point cloud
in canoncial alignment. Choose whether to train on partial or complete clouds. Transformations
are applied randomly during training. Validation transformations should be defined in a text file.
'''

import argparse
import datetime
import importlib
from models import itn
import os
import pytz
import tensorflow as tf
from tensorflow.python import debug as  tf_debug
import time
from data_util_itn import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views

from pdb import set_trace as st

class TrainProvider:
    def __init__(self, args, is_training):
        df_train, self.num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                                 args.sample_size, is_training=True,
                                                 so3_perturb=args.so3, use_partial=args.use_partial,
                                                 train_perturb_list=args.train_perturbations)
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.sample_size, 3],
                                       [args.batch_size, 4, 4]])
        df_valid, self.num_valid = lmdb_dataflow(args.lmdb_valid, args.batch_size,
                                                 args.sample_size, is_training=False, 
                                                 valid_perturb_list=args.valid_perturbations,
                                                 use_partial=args.use_partial)
        batch_valid = get_queued_data(df_valid.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.sample_size, 3],
                                       [args.batch_size, 4, 4]])
        self.batch_data = tf.cond(is_training, lambda: batch_train, lambda: batch_valid)

def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    provider = TrainProvider(args, is_training_pl)

    ids, inputs, gt_pose = provider.batch_data
    num_eval_steps = provider.num_valid // args.batch_size

    model = itn.ITN(inputs, gt_pose, args.iterations, args.validation_iterations, args.no_batchnorm, args.rot_representation, is_training_pl)
    
    if not args.no_lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    saver = tf.train.Saver()

    now = datetime.datetime.now(pytz.timezone('US/Pacific'))
    hr = '{:02d}'.format(now.hour)
    mn = '{:02d}'.format(now.minute)
    dy = '{:02d}'.format(now.day)
    mt = '{:02d}'.format(now.month)
    yr = '{:04d}'.format(now.year)
    log_dir = args.log_dir + '_'.join(['', hr, mn, dy, mt, yr])

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    else:
        if os.path.exists(log_dir):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                       % log_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s' % log_dir)
                os.makedirs(log_dir)
                os.makedirs(os.path.join(log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(log_dir, 'plots'))
        with open(os.path.join(log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments
        os.system('cp models/itn.py %s' % (log_dir))  # bkp of model def
        os.system('cp train.py %s' % log_dir)                         # bkp of train procedure

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_time = 0
    train_start = time.time()
    step = sess.run(global_step)

    while not coord.should_stop():
        step += 1
        epoch = step * args.batch_size // provider.num_train + 1
        start = time.time()
        __, loss, summary = sess.run([train_op, model.loss, train_summary],
                                     feed_dict={is_training_pl: True})
        total_time += time.time() - start
        writer.add_summary(summary, step)

        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                  (epoch, step, loss, total_time / args.steps_per_print))
            total_time = 0

        if step % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            total_loss = 0
            total_time = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                loss, _ = sess.run([model.loss, model.update],
                                   feed_dict={is_training_pl: False})
                total_loss += loss
                total_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                          (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))
            total_time = 0
        
        if step % args.steps_per_visu == 0:
            model_id, pcds = sess.run([ids[0], model.visualize_ops],
                                      feed_dict={is_training_pl: True})
            model_id = model_id.decode('utf-8')
            plot_path = os.path.join(log_dir, 'plots',
                                     'epoch_%d_step_%d_%s.png' % (epoch, step, model_id))
            plot_pcd_three_views(plot_path, pcds, model.visualize_titles)
        
        if step % args.steps_per_save == 0:
            saver.save(sess, os.path.join(log_dir, 'model'), step)
            print(colored('Model saved at %s' % log_dir, 'white', 'on_blue'))
        
        if step >= args.max_step:
            break

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/shapenet_car/train.lmdb')
    parser.add_argument('--lmdb_valid', default='data/shapenet_car/valid.lmdb')
    parser.add_argument('--valid_perturbations', default='data/shapenet_car/itn_valid_perturb.txt')
    parser.add_argument('--train_perturbations', default=None)
    parser.add_argument('--log_dir', default='log/itn')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=2048, help='num points to sample in input')
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--no_lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=7500)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=75000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=1000)
    parser.add_argument('--steps_per_visu', type=int, default=1000)
    parser.add_argument('--steps_per_save', type=int, default=10000)
    parser.add_argument('--so3', action='store_true', help='use random rotation with 3DoF (default 1DoF SE2 heading)')
    parser.add_argument('--use_partial', action='store_true', help='use partial clouds as input instead of complete clouds')
    parser.add_argument('--iterations', default=1, type=int, help='number of iterations during training')
    parser.add_argument('--no-batchnorm', action='store_true', help='do not use batchnorm')
    parser.add_argument('--validation-iterations', type=int, default=1, help='number of ITN iterations during validation')
    parser.add_argument('--rot-representation', choices=['quat','6dof'], default='quat', help='rotation representation for regression')
    args = parser.parse_args()

    train(args)
