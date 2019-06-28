import numpy as np
import tensorflow as tf

def random_rotation(so3):
    angle_x = np.random.uniform() * 2 * np.pi if so3 else 0
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi if so3 else 0

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

def euldeg2rotm(x,y,z):
    angle_x = x / 180 * np.pi
    angle_y = y / 180 * np.pi
    angle_z = z / 180 * np.pi

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

def rotm2quat(rotm):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    m00 = rotm[0,0]
    m01 = rotm[0,1]
    m02 = rotm[0,2]
    m10 = rotm[1,0]
    m11 = rotm[1,1]
    m12 = rotm[1,2]
    m20 = rotm[2,0]
    m21 = rotm[2,1]
    m22 = rotm[2,2]

    tr = m00 + m11 + m22

    if (tr > 0): 
        S = ((tr + 1.0) ** 0.5) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = ((1.0 + m00 - m11 - m22) ** 0.5) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif (m11 > m22):
        S = ((1.0 + m11 - m00 - m22) ** 0.5) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = ((1.0 + m22 - m00 - m11) ** 0.5) * 2
        qw = (m10 - m01) / S
        qx = (m02 - m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    quat = np.zeros(4)
    quat[0] = qw
    quat[1] = qx
    quat[2] = qy
    quat[3] = qz

    return quat

def quat2rotm(quat):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    # assume input is [w, x, y, z]
    
    # normalize first
    quat = quat / np.linalg.norm(quat)

    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    rotmat = np.zeros((3,3))
    rotmat[0,0] = 1 - 2*qy**2 - 2*qz**2
    rotmat[0,1] = 2*qx*qy - 2*qz*qw
    rotmat[0,2] = 2*qx*qz + 2*qy*qw
    
    rotmat[1,0] = 2*qx*qy + 2*qz*qw
    rotmat[1,1] = 1 - 2*qx**2 - 2*qz**2
    rotmat[1,2] = 2*qy*qz - 2*qx*qw

    rotmat[2,0] = 2*qx*qz - 2*qy*qw
    rotmat[2,1] = 2*qy*qz + 2*qx*qw
    rotmat[2,2] = 1 - 2*qx**2 - 2*qy**2

    return rotmat

def euldeg2quat(x,y,z):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
    angle_x = x / 180 * np.pi
    angle_y = y / 180 * np.pi
    angle_z = z / 180 * np.pi

    c1 = np.cos(angle_y / 2)
    c2 = np.cos(angle_z / 2)
    c3 = np.cos(angle_x / 2)
    s1 = np.sin(angle_y / 2)
    s2 = np.sin(angle_z / 2)
    s3 = np.sin(angle_x / 2)

    qw = c1 * c2 * c3 + s1 * s2 * s3
    qx = c1 * c2 * s3 - s1 * s2 * c3
    qy = s1 * c2 * c3 + c1 * s2 * s3
    qz = c1 * s2 * c3 - s1 * c2 * s3

    quat = np.zeros(4)
    quat[0] = qw
    quat[1] = qx
    quat[2] = qy
    quat[3] = qz

    return quat

def transform_tf(points, transform):
    # in: B x N x 3, B x 4 x 4
    # out: B x N x 3

    R = transform[:, 0:3, 0:3]
    t = tf.tile(tf.expand_dims(transform[:, 0:3, 3], axis=1), [1, tf.shape(points)[1], 1])

    points_transform = tf.linalg.matmul(points, R, transpose_b=True)
    points_transform = points_transform + t

    return points_transform

def quat2rotm_tf(quat):
    # in: B x 4
    # out: B x 3 x 3

    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    # assume input is [w, x, y, z]
    
    # normalize first
    quat_norm = tf.tile(tf.norm(quat, axis=1, keepdims=True),[1, 4])
    quat = quat / quat_norm

    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    
    rotmat = tf.zeros((tf.shape(quat)[0],3,3))

    rotmat00 = 1 - 2*qy**2 - 2*qz**2
    rotmat01 = 2*qx*qy - 2*qz*qw
    rotmat02 = 2*qx*qz + 2*qy*qw
    rotmat0 = tf.stack([rotmat00, rotmat01, rotmat02], axis=1)

    rotmat10= 2*qx*qy + 2*qz*qw
    rotmat11 = 1 - 2*qx**2 - 2*qz**2
    rotmat12 = 2*qy*qz - 2*qx*qw
    rotmat1 = tf.stack([rotmat10, rotmat11, rotmat12], axis=1)

    rotmat20 = 2*qx*qz - 2*qy*qw
    rotmat21 = 2*qy*qz + 2*qx*qw
    rotmat22 = 1 - 2*qx**2 - 2*qy**2
    rotmat2 = tf.stack([rotmat20, rotmat21, rotmat22], axis=1)

    rotmat = tf.stack([rotmat0, rotmat1, rotmat2], axis=1)

    return rotmat

def mat6d2rotm_tf(mat6d):
    # in: B x 6
    # out: B x 3 x 3

    # https://arxiv.org/pdf/1812.07035.pdf
    # see eq. 15/16 in supp. mat.

    def N(x):
        x_norm = tf.tile(tf.norm(x, axis=1, keepdims=True), [1, x.shape[1]])
        return x / x_norm

    a1 = mat6d[:, 0:3]
    a2 = mat6d[:, 3:6]
    b1 = N(a1)
    dotprod = tf.reduce_sum(tf.multiply(b1, a2), 1, keepdims=True)
    b2 = N(a2 - dotprod * b1)
    b3 = tf.cross(b1, b2)
    
    rotmat = tf.stack([b1, b2, b3], axis=2)

    return rotmat

def rotm2mat6d_tf(rotm):
    # in: B x 3 x 3
    # out: B x 6

    # https://arxiv.org/pdf/1812.07035.pdf
    # see eq. 14 in supp. mat.

    mat6d = rotm[:, :, 0:2]
    mat6d = tf.transpose(mat6d, perm=[0, 2, 1])
    mat6d = tf.reshape(mat6d, [mat6d.shape[0], 6])

    return mat6d