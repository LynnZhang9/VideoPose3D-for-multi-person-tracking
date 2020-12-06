import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.optimize
import functools
from matplotlib import cm
# FILE = "output_3d.npy"
# FILE = r"./temp_output/output_3d_predictions.npy"
# FILE = r"./temp_output/output_3d_keypoints.npy"
# FILE = r"./temp_output/output_3d_after_image_coordinates.npy"
FILE = r"./temp_output/output_3d_after_camera_to_world.npy"

data = np.load(FILE)


#Functions for plane regression
def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def plane_regression(points):
    fun = functools.partial(error, points=points)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    xs, ys, zs = zip(*points)

    point = np.array([0.0, 0.0, c])
    normal = np.array(cross([1,0,a], [0,1,b]))

    d = -point.dot(normal)
    # x yy define the plane area
    xx, yy = np.meshgrid([-4,-2], [-9,-3])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    return xx, yy, z, xs, ys, zs


def define_area(point1, point2, point3):

    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D

def point2area_distance(point1, point2, point3, point4):
    """
    :param point1:A point in the plane
    :param point2:A point in the plane
    :param point3:A point in the plane
    :param point4:A point out of the plane
    :return:distance
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


if __name__ == '__main__':
    # All ankle points
    ankles_exist = False
    for fr in range(data.shape[1]):

        left_ankle = data[:, fr, 3, :]
        right_ankle = data[:, fr, 6, :]
        if not ankles_exist:
            left_ankles = left_ankle
            right_ankles = right_ankle
            ankles_exist = True
        else:
            left_ankles = np.concatenate((left_ankles, left_ankle), axis=0)
            right_ankles = np.concatenate((right_ankles, right_ankle), axis=0)


    left_points = left_ankles[~np.isnan(left_ankles).any(axis=1), :]
    right_points = right_ankles[~np.isnan(right_ankles).any(axis=1), :]
    lx, ly, lz = left_points.T
    rx, ry, rz = right_points.T

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))


    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Left ankle points distribution')
    ax.set(xlabel='Z value', ylabel='Amount of points')
    plt.hist(lz, bins=50)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Right ankle points distribution')
    ax.set(xlabel='Z value', ylabel='Amount of points')
    plt.hist(rz, bins=50)
    plt.show()
