# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3), disp=False):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    
    if disp:
        plt.show()
    else:
        fig.savefig(filename)

    plt.close(fig)
