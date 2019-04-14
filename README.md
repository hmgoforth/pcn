## PCN: Point Completion Network
#### [[paper]](https://arxiv.org/pdf/1808.00671.pdf) [[data]](https://drive.google.com/open?id=1Af9igOStb6O9YHwjYHOwR0qW4uP3zLA6) [[website]](https://cs.cmu.edu/~wyuan1/pcn)

### Introduction
PCN is a learning-based shape completion method which directly maps a partial point cloud to a dense, complete point cloud without any voxelization. It is based on our 3DV 2018 publication [PCN: Point Completion Network](https://arxiv.org/abs/1808.00671/). Please refer to our [project website](https://cs.cmu.edu/~wyuan1/pcn) or read our paper for more details.

### Usage

#### 1) Prerequisite
1. Under `scripts` folder, run `./build_docker.sh`
3. Download trained models from [Google Drive](https://drive.google.com/open?id=1Af9igOStb6O9YHwjYHOwR0qW4uP3zLA6).

#### 2) Demo
Under 'scripts' folder, run 'run_docker.sh'
Inside docker run `python3 demo.py`. Use `--input_path` option to switch between the input examples in `demo_data`.

#### 3) ShapeNet Completion
1. Download ShapeNet test data in the `shapenet` folder on [Google Drive](https://drive.google.com/open?id=1Af9igOStb6O9YHwjYHOwR0qW4uP3zLA6). Specifically, this experiment requires `test`, `test_novel`, `test.list` and `test_novel.list`.
2. Run `python3 test_shapenet.py`. Use `--model_type` option to choose different model architectures. Type `python3 test_shapenet.py -h` for more options.

#### 4) KITTI Completion
1. Download KITTI data in the `kitti` folder on [Google Drive](https://drive.google.com/open?id=1Af9igOStb6O9YHwjYHOwR0qW4uP3zLA6).
2. Run `python3 test_kitti.py`. Type `python3 test_kitti.py -h` for more options.

#### 5) KITTI Registration
1. Run the KITTI completion experiment first to get complete point clouds.
2. Run `python3 kitti_registration.py`. Type `python3 kitti_registration.py -h` for more options.

#### 6) Training
1. Download training (`train.lmdb`, `train.lmdb-lock`) and validation (`valid.lmdb`, `valid.lmdb-lock`) data from `shapenet` or `shapenet_car` directory on [Google Drive](https://drive.google.com/open?id=1Af9igOStb6O9YHwjYHOwR0qW4uP3zLA6). Note that the training data for all 8 categories in `shapenet` takes up 49G of disk space. The training data for only the car category takes 9G instead.
2. Run `python3 train.py`. Type `python3 train.py -h` for more options.

#### 7) Data Generation
To generate your own data from ShapeNet, first Download [ShapeNetCore.v1](https://shapenet.org). Then, create partial point clouds from depth images (see instructions in `render`) and corresponding ground truths by sampling from CAD models (see instructions in `sample`). Finally, serialize the data using `lmdb_writer.py`.

### Citation
If you find our work useful for your research, please cite:
```
@inProceedings{yuan2018pcn,
  title     = {PCN: Point Completion Network},
  author    = {Yuan, Wentao and Khot, Tejas and Held, David and Mertz, Christoph and Hebert, Martial},
  booktitle = {3D Vision (3DV), 2018 International Conference on},
  year      = {2018}
}
```

### License
This project Code is released under the MIT License (refer to the LICENSE file for details).
