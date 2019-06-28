'''
Given an lmdb file (for partial completion) and number of samples, select subset of lmdb and save as new lmdb.
'''

import argparse
import numpy as np
import os
import tensorflow as tf
from tensorpack import DataFlow, dataflow
from termcolor import colored

from create_noncanonical_shapenet_cars import save_pcd
from lmdb_writer_noncanonical_shapenet_cars import pcd_df

from pdb import set_trace as st

def main(args):
    df = dataflow.LMDBSerializer.load(args.lmdb, shuffle=False)
    df_gen = df.get_data()

    partial_dir = os.path.join(args.save_dir, 'partial')
    complete_dir = os.path.join(args.save_dir, 'complete')

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
        os.makedirs(partial_dir)
        os.makedirs(complete_dir)

    id_list = []
    for i in range(args.num_samples):
        pair_id, partial, complete = next(df_gen)
        save_pcd(partial_dir, pair_id, partial - 0)
        save_pcd(complete_dir, pair_id, complete - 0)
        id_list.append(pair_id)

    new_df = pcd_df(id_list, partial_dir, complete_dir)

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path):
        os.system('rm %s' % save_path)
    
    dataflow.LMDBSerializer.save(new_df, save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', help='lmdb from which samples are taken')
    parser.add_argument('--num-samples', type=int, help='how many samples to take from lmdb')
    parser.add_argument('--save-dir', help='location of new lmdb. All point clouds in new lmdb will be saved as pcd in subfolders here.')
    parser.add_argument('--save-name', help='name of new lmdb')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)