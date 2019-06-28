'''
Read shapenet_cars, apply random rotations and subtract partial centroid
from each pair. Save as new dataset, including model list, new directory
with partial data, new directory with complete gt, and total number of pairs
for train and validation
'''

import argparse
import numpy as np
from open3d import *
import os
from pdb import set_trace as st
from tensorpack import dataflow
from termcolor import colored
from visu_util import plot_pcd_three_views

def main(args):
    # create directories to store partial and complete gt for train and valid
    train_partial_dir = os.path.join(args.save_dir, 'train_partial')
    train_complete_dir = os.path.join(args.save_dir, 'train_complete')
    valid_partial_dir = os.path.join(args.save_dir, 'valid_partial')
    valid_complete_dir = os.path.join(args.save_dir, 'valid_complete')

    if os.path.exists(args.save_dir):
        delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                    % args.save_dir, 'white', 'on_red'))
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % args.save_dir)
            new_dir = True
    else:
        os.makedirs(args.save_dir)
        new_dir = True

    if new_dir:
        os.makedirs(train_partial_dir)
        os.makedirs(train_complete_dir)
        os.makedirs(valid_partial_dir)
        os.makedirs(valid_complete_dir)

    train_id_list = process_lmdb(args.lmdb_train, train_partial_dir, train_complete_dir, args.se3)
    valid_id_list = process_lmdb(args.lmdb_valid, valid_partial_dir, valid_complete_dir, args.se3)

    save_list(os.path.join(args.save_dir, 'train_list'), train_id_list)
    save_list(os.path.join(args.save_dir, 'valid_list'), valid_id_list)

def save_list(save_path, id_list):
    with open(save_path, 'w') as f:
        for item in id_list:
            f.write("%s\n" % item)

def process_lmdb(lmdb_fn, partial_save_dir, complete_save_dir, rot_3d):
    # for each entry in lmdb, apply random rotation and subtract centroid
    # from both partial and complete

    df = dataflow.LMDBSerializer.load(lmdb_fn, shuffle=False)
    df_gen = df.get_data()
    id_list = []

    size = df.size()
    itr = 0

    for pair_id, partial, complete in df_gen:
        itr += 1
        print('{:d} / {:d}'.format(itr, size))

        id_list.append(pair_id)
        partial_nc, complete_nc = rotate_and_subtract_centroid(partial, complete, rot_3d)

        save_pcd(partial_save_dir, pair_id, partial_nc)
        save_pcd(complete_save_dir, pair_id, complete_nc)

    return id_list 

def rotate_and_subtract_centroid(partial, complete, rot_3d):
    R = random_rotation(rot_3d)
    partial_rot = partial @ R.T
    complete_rot = complete @ R.T

    partial_centroid = np.mean(partial_rot, axis=0, keepdims=True)

    partial_nc = partial_rot - partial_centroid
    complete_nc = complete_rot - partial_centroid

    # plot_pcd_three_views(None, 
    #     [partial, complete, partial_nc, complete_nc],
    #     ['partial', 'complete', 'partial_nc', 'complete_nc'],
    #     disp=True)

    # print('mean(partial_nc):')
    # print(np.mean(partial_nc, axis=0))
    # # print('mean(complete_nc):')
    # # print(np.mean(complete_nc))

    return partial_nc, complete_nc

def random_rotation(rot_3d):
    angle_x = np.random.uniform() * 2 * np.pi if rot_3d else 0
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi if rot_3d else 0

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def save_pcd(save_dir, pair_id, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    write_point_cloud(os.path.join(save_dir, '%s.pcd' % pair_id), pcd)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--lmdb_train',
        default='data/shapenet_car/train.lmdb')
    
    parser.add_argument(
        '--lmdb_valid',
        default='data/shapenet_car/valid.lmdb')

    parser.add_argument(
        '--save_dir',
        default='data/shapenet_car/noncanonical_se2'
    )
    
    parser.add_argument(
        '--se3',
        action='store_true',
        help='use random 3D rotations (default: 2D in x-y plane)')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
