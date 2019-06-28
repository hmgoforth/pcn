'''
Given validation LMBD dataset, creates file of quaternion rotations
'''

import argparse
import numpy as np
import tensorflow as tf
from tensorpack import dataflow

from transform_util import random_rotation, rotm2quat

from pdb import set_trace as st

def main(args):
    df = dataflow.LMDBSerializer.load(args.lmdb_valid, shuffle=False)
    size = df.size()

    rotations = np.zeros((size, 4))

    for i in range(size):
        R = random_rotation(args.so3)
        rotations[i] = rotm2quat(R)

    np.savetxt(args.valid_perturbations, rotations)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_valid', default='data/shapenet_car/valid.lmdb')
    parser.add_argument('--valid_perturbations', default='data/shapenet_car/itn_valid_perturb.txt')
    parser.add_argument('--so3', action='store_true', help='use random rotation with 3DoF (default 1DoF SE2 heading)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)