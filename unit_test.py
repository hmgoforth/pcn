import numpy as np
import unittest
import tensorflow as tf; tf.enable_eager_execution()

from transform_util import rotm2quat, quat2rotm, euldeg2rotm, euldeg2quat, mat6d2rotm_tf, rotm2mat6d_tf

from pdb import set_trace as st

class Test_euldeg2quat(unittest.TestCase):
    def test_identity(self):
        quat = euldeg2quat(0, 0, 0)
        quat_gt = np.zeros(4)
        quat_gt[0] = 1
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))
    
    def test_x90(self):
        quat = euldeg2quat(90, 0, 0)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.7071068
        quat_gt[1] = 0.7071068
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))
    
    def test_y90(self):
        quat = euldeg2quat(0, 90, 0)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.7071068
        quat_gt[2] = 0.7071068
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

    def test_z90(self):
        quat = euldeg2quat(0, 0, 90)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.7071068
        quat_gt[3] = 0.7071068
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

    def test_x180(self):
        quat = euldeg2quat(180, 0, 0)
        quat_gt = np.zeros(4)
        quat_gt[1] = 1
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

    def test_y180(self):
        quat = euldeg2quat(0, 180, 0)
        quat_gt = np.zeros(4)
        quat_gt[2] = 1
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

    def test_z180(self):
        quat = euldeg2quat(0, 0, 180)
        quat_gt = np.zeros(4)
        quat_gt[3] = 1
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

    def test_rand(self):
        quat = euldeg2quat(23, 82, 13)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.74961123
        quat_gt[1] = 0.07672064
        quat_gt[2] = 0.65578898
        quat_gt[3] = -0.04623594
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

        quat = euldeg2quat(-240, 428, -888)
        quat_gt = np.zeros(4)
        quat_gt[0] = -0.43829334
        quat_gt[1] = 0.35311284
        quat_gt[2] = -0.68480871
        quat_gt[3] = -0.46286856
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

class Test_rotm2quat(unittest.TestCase):
    def test_identity(self):
        R = np.eye(3)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[0] = 1
        assert np.array_equal(quat, quat_gt)

    def test_x90(self):
        R = euldeg2rotm(90, 0, 0)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.7071068
        quat_gt[1] = 0.7071068
        assert np.all(np.isclose(quat, quat_gt))
    
    def test_y90(self):
        R = euldeg2rotm(0, 90, 0)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.7071068
        quat_gt[2] = 0.7071068
        assert np.all(np.isclose(quat, quat_gt))

    def test_z90(self):
        R = euldeg2rotm(0, 0, 90)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.7071068
        quat_gt[3] = 0.7071068
        assert np.all(np.isclose(quat, quat_gt))

    def test_x180(self):
        R = euldeg2rotm(180, 0, 0)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[1] = 1
        assert np.all(np.isclose(quat, quat_gt))

    def test_y180(self):
        R = euldeg2rotm(0, 180, 0)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[2] = 1
        assert np.all(np.isclose(quat, quat_gt))

    def test_z180(self):
        R = euldeg2rotm(0, 0, 180)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[3] = 1
        assert np.all(np.isclose(quat, quat_gt))

    def test_rand(self):
        R = euldeg2rotm(23, 82, 13)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[0] = 0.74961123
        quat_gt[1] = 0.07672064
        quat_gt[2] = 0.65578898
        quat_gt[3] = -0.04623594
        assert np.all(np.isclose(quat, quat_gt))

        R = euldeg2rotm(-240, 428, -888)
        quat = rotm2quat(R)
        quat_gt = np.zeros(4)
        quat_gt[0] = -0.43829334
        quat_gt[1] = 0.35311284
        quat_gt[2] = -0.68480871
        quat_gt[3] = -0.46286856
        assert np.all(np.isclose(quat, quat_gt)) or np.all(np.isclose(-quat, quat_gt))

class Test_quat2rotm(unittest.TestCase):
    def test_identity(self):
        q = euldeg2quat(0, 0, 0)
        rotm_gt = euldeg2rotm(0, 0, 0)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

    def test_x90(self):
        q = euldeg2quat(90, 0, 0)
        rotm_gt = euldeg2rotm(90, 0, 0)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

    def test_y90(self):
        q = euldeg2quat(0, 90, 0)
        rotm_gt = euldeg2rotm(0, 90, 0)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))
    
    def test_z90(self):
        q = euldeg2quat(0, 0, 90)
        rotm_gt = euldeg2rotm(0, 0, 90)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))
    
    def test_x180(self):
        q = euldeg2quat(180, 0, 0)
        rotm_gt = euldeg2rotm(180, 0, 0)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

    def test_y180(self):
        q = euldeg2quat(0, 180, 0)
        rotm_gt = euldeg2rotm(0, 180, 0)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))
    
    def test_z180(self):
        q = euldeg2quat(0, 0, 180)
        rotm_gt = euldeg2rotm(0, 0, 180)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

    def test_rand(self):
        q = euldeg2quat(23, 82, 13)
        rotm_gt = euldeg2rotm(23, 82, 13)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

        q = euldeg2quat(-240, 428, -888)
        rotm_gt = euldeg2rotm(-240, 428, -888)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

    def test_unnormalized(self):
        q = euldeg2quat(2, -45, 21) * 5
        rotm_gt = euldeg2rotm(2, -45, 21)
        rotm = quat2rotm(q)
        assert np.all(np.isclose(rotm, rotm_gt))

class Test_rotm2quat2rotm(unittest.TestCase):
    def test_rand(self):
        rotm_1 = euldeg2rotm(23, 82, 13)
        q = rotm2quat(rotm_1)
        rotm_2 = quat2rotm(q)
        assert np.all(np.isclose(rotm_1, rotm_2))

        rotm_1 = euldeg2rotm(-240, 428, -888)
        q = rotm2quat(rotm_1)
        rotm_2 = quat2rotm(q)
        assert np.all(np.isclose(rotm_1, rotm_2))

class Test_quat2rotm2quat(unittest.TestCase):
    def test_rand(self):
        q_1 = euldeg2quat(23, 82, 13)
        rotm = quat2rotm(q_1)
        q_2 = rotm2quat(rotm)
        assert np.all(np.isclose(q_1, q_2)) or np.all(np.isclose(-q_1, q_2))

        q_1 = euldeg2quat(-240, 428, -888)
        rotm = quat2rotm(q_1)
        q_2 = rotm2quat(rotm)
        assert np.all(np.isclose(q_1, q_2)) or np.all(np.isclose(-q_1, q_2))

class Test_mat6drotm_tf(unittest.TestCase):
    def test_identity(self):
        rotm = euldeg2rotm(0, 0, 0)
        rotm = tf.convert_to_tensor(np.tile(np.expand_dims(rotm, axis=0), (24, 1, 1)))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))

    def test_x90(self):
        rotm = euldeg2rotm(90, 0, 0)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))

    def test_y90(self):
        rotm = euldeg2rotm(0, 90, 0)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))
    
    def test_z90(self):
        rotm = euldeg2rotm(0, 0, 90)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))
    
    def test_x180(self):
        rotm = euldeg2rotm(180, 0, 0)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))

    def test_y180(self):
        rotm = euldeg2rotm(0, 180, 0)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))
    
    def test_z180(self):
        rotm = euldeg2rotm(0, 0, 180)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))

    def test_rand(self):
        rotm = euldeg2rotm(23, 82, 13)
        rotm = tf.convert_to_tensor(np.expand_dims(rotm, axis=0))
        mat6d_gt = rotm2mat6d_tf(rotm)
        rotm_est = mat6d2rotm_tf(mat6d_gt)
        assert np.all(np.isclose(rotm, rotm_est))