import tensorflow as tf
from tf_util import mlp, mlp_conv, add_train_summary, add_valid_summary
from transform_util import transform_tf, quat2rotm_tf, mat6d2rotm_tf
import sys

from pdb import set_trace as st

class ITN:
    def __init__(self, inputs, gt_pose, iterations, validation_iterations, no_batchnorm, rot_representation, is_training):
        self.iterations = iterations
        self.validation_iterations = validation_iterations
        self.is_training = is_training
        self.rot_representation = rot_representation

        if no_batchnorm:
            self.bn = None
        else:
            self.bn = tf.contrib.layers.batch_norm

        self.gt_inputs = transform_tf(inputs, tf.linalg.inv(gt_pose))
        self.est_pose, self.est_inputs = self.estimate(inputs)
        self.loss, self.update = self.compute_loss(inputs, gt_pose, self.est_pose)
        self.outputs = self.est_pose
        self.visualize_ops = [inputs[0], self.est_inputs[0], self.gt_inputs[0]]
        self.visualize_titles = ['input', 'estimated', 'ground truth']

    def estimate(self, inputs):
        B = tf.shape(inputs)[0]
        with tf.variable_scope('est', reuse=tf.AUTO_REUSE):
            est_pose = tf.tile(tf.expand_dims(tf.eye(4), axis=0), [B, 1, 1])
            est_inputs = tf.identity(inputs)

        num_iter = tf.cond(self.is_training, lambda: self.iterations, lambda: self.validation_iterations)
        i = tf.constant(0)
        while_condition = lambda i, est_pose, est_inputs: tf.less(i, num_iter)

        def while_body(i, est_pose, est_inputs):
            with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
                features = mlp_conv(est_inputs, [128, 256], bn=self.bn)
                features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
                features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
                features = mlp_conv(features, [512, 1024], bn=self.bn)
                features = tf.reduce_max(features, axis=1, name='maxpool_1')
            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
                if self.rot_representation == 'quat':
                    est_pose_rep_i = mlp(features, [1024, 1024, 512, 512, 256, 7], bn=self.bn)
                elif self.rot_representation == '6dof':
                    est_pose_rep_i = mlp(features, [1024, 1024, 512, 512, 256, 9], bn=self.bn)

            with tf.variable_scope('est', reuse=tf.AUTO_REUSE):
                if self.rot_representation == 'quat':
                    t = tf.expand_dims(est_pose_rep_i[:, 4:], axis=2)
                    q = est_pose_rep_i[:, 0:4]
                    R = quat2rotm_tf(q)
                elif self.rot_representation == '6dof':
                    t = tf.expand_dims(est_pose_rep_i[:, 6:], axis=2)
                    mat6d = est_pose_rep_i[:, 0:6]
                    R = mat6d2rotm_tf(mat6d)

                est_pose_T_i = tf.concat([
                    tf.concat([R, t], axis=2),
                    tf.concat([tf.zeros([B, 1, 3]), tf.ones([B, 1, 1])], axis=2)],
                    axis=1)
                est_inputs = transform_tf(est_inputs, est_pose_T_i)
                est_pose = tf.linalg.matmul(est_pose_T_i, est_pose)

            return [tf.add(i, 1), est_pose, est_inputs]

        _, est_pose, est_inputs = tf.while_loop(while_condition, while_body, [i, est_pose, est_inputs])

        return est_pose, est_inputs

    def compute_loss(self, inputs, gt_pose, est_pose):
        # see equation (1) from IT-net
        # est_pose: world -> body coord
        # gt_pose: body -> world coord (to to invert when applying to inputs)
        est_inputs = transform_tf(inputs, est_pose)
        gt_inputs = transform_tf(inputs, tf.linalg.inv(gt_pose))
        sq_dist = tf.reduce_sum(tf.square(est_inputs - gt_inputs), axis=2)
        loss = tf.reduce_mean(tf.reduce_mean(sq_dist, axis=1), axis=0)

        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, update_loss