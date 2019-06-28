import argparse
import os
from io_util import read_pcd
from tensorpack import DataFlow, dataflow

# create train.lmdb:
# python3 lmdb_writer_noncanonical_shapenet_cars.py --list_path data/shapenet_car/noncanonical_se2/train_list --partial_dir data/shapenet_car/noncanonical_se2/train_partial --complete_dir data/shapenet_car/noncanonical_se2/train_complete --output_path data/shapenet_car/noncanonical_se2/train.lmbd
# create valid.lmdb
# python3 lmdb_writer_noncanonical_shapenet_cars.py --list_path data/shapenet_car/noncanonical_se2/valid_list --partial_dir data/shapenet_car/noncanonical_se2/valid_partial --complete_dir data/shapenet_car/noncanonical_se2/valid_complete --output_path data/shapenet_car/noncanonical_se2/valid.lmbd

class pcd_df(DataFlow):
    def __init__(self, model_list, partial_dir, complete_dir):
        self.model_list = model_list
        self.partial_dir = partial_dir
        self.complete_dir = complete_dir

    def size(self):
        return len(self.model_list)

    def get_data(self):
        for model_id in self.model_list:
            complete = read_pcd(os.path.join(self.complete_dir, '%s.pcd' % model_id))
            partial = read_pcd(os.path.join(self.partial_dir, '%s.pcd' % model_id))
            yield model_id, partial, complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path')
    parser.add_argument('--partial_dir')
    parser.add_argument('--complete_dir')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    df = pcd_df(model_list, args.partial_dir, args.complete_dir)

    if os.path.exists(args.output_path):
        os.system('rm %s' % args.output_path)
    
    dataflow.LMDBSerializer.save(df, args.output_path)
