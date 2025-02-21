# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import importlib
import models
import numpy as np
import os
import tensorflow as tf
import time
from io_util import read_pcd, save_pcd
from visu_util import plot_pcd_three_views


def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, gt, tf.constant(1.0))

    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'completions'), exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    car_ids = [filename.split('.')[0] for filename in os.listdir(args.pcd_dir)]
    total_time = 0
    total_points = 0
    for i, car_id in enumerate(car_ids):
        partial = read_pcd(os.path.join(args.pcd_dir, '%s.pcd' % car_id))
        bbox = np.loadtxt(os.path.join(args.bbox_dir, '%s.txt' % car_id))
        total_points += partial.shape[0]

        # Calculate center, rotation and scale
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = (bbox[3, 0] - bbox[0, 0])
        bbox /= scale

        partial = np.dot(partial - center, rotation) / scale
        partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        partial_perturb = partial + [0.2, 0.2, 0]

        start = time.time()
        completion = sess.run(model.outputs, feed_dict={inputs: [partial]})
        completion_perturb = sess.run(model.outputs, feed_dict={inputs: [partial_perturb]})
        total_time += time.time() - start
        completion = completion[0]
        completion_perturb = completion_perturb[0]

        completion_w = np.dot(completion, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        completion_w = np.dot(completion_w * scale, rotation.T) + center

        pcd_path = os.path.join(args.results_dir, 'completions', '%s.pcd' % car_id)
        save_pcd(pcd_path, completion_w)

        if i % args.plot_freq == 0:
            plot_path_perturb = os.path.join(args.results_dir, 'plots', '%s_perturb.png' % car_id)
            plot_path = os.path.join(args.results_dir, 'plots', '%s.png' % car_id)
            
            plot_pcd_three_views(plot_path_perturb, [partial_perturb, completion_perturb], ['input', 'output'],
                                 '%d input points' % partial.shape[0], [5, 0.5])
            plot_pcd_three_views(plot_path, [partial, completion], ['input', 'output'],
                                 '%d input points' % partial.shape[0], [5, 0.5])

    print('Average # input points:', total_points / len(car_ids))
    print('Average time:', total_time / len(car_ids))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_emd_car')
    parser.add_argument('--pcd_dir', default='data/kitti/cars')
    parser.add_argument('--bbox_dir', default='data/kitti/bboxes')
    parser.add_argument('--results_dir', default='data/results/kitti_pcn_emd')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    args = parser.parse_args()

    test(args)
