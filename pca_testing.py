from io_util import read_pcd
import numpy as np
from pdb import set_trace as st
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *
import argparse
import glob
import os

def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)

parser = argparse.ArgumentParser()
parser.add_argument('--point_cloud_dir')
args = parser.parse_args()

for point_cloud in os.listdir(args.point_cloud_dir):
    if point_cloud.endswith('.pcd'):
        pcd = read_pcd(os.path.join(args.point_cloud_dir, point_cloud))
        pcd = pcd - np.mean(pcd, axis=0)

        cov = np.cov(pcd.T)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        evecs_sort = evecs[:, sort_indices]

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        plot_pcd(ax, pcd)
        ax.plot3D([0, evecs_sort[0, 0]],[0, evecs_sort[1,0]],[0, evecs_sort[2,0]],color='blue',zdir='y')
        ax.plot3D([0, evecs_sort[0, 1]],[0, evecs_sort[1,1]],[0, evecs_sort[2,1]],color='red',zdir='y')
        ax.plot3D([0, evecs_sort[0, 2]],[0, evecs_sort[1,2]],[0, evecs_sort[2,2]],color='green',zdir='y')
        plt.show()