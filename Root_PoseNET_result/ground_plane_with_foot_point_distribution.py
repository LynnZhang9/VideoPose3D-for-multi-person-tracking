import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
import numpy as np
import scipy.optimize
import functools
from matplotlib import cm
import pickle



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


def plane_regression(points, x_interval=[-10,10], y_interval=[-10,10]):
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
    xx, yy = np.meshgrid(x_interval, y_interval)
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
def get_points_on_ground_plane(data_path):
    # data_path = r"../Extrinsics_calculation/output/GP010170.MP4_camera_data.pkl"
    with open(data_path, "rb") as input_file:
        camera_dict = pickle.load(input_file)

    image_inputs = camera_dict['image_points']
    homography = camera_dict['homography']
    shape = list(image_inputs.shape)
    shape[1] = shape[1] + 1
    b = np.ones(shape)
    b[:,:-1] = image_inputs
    world_coordinate = np.dot(homography,b.T)
    aj_world_coordinate = world_coordinate/world_coordinate[2]


    shape2 = list(camera_dict['world_coordinate_origin_offset'].shape)
    shape2[0] = shape2[0] + 1
    world_coor = np.zeros(shape2)
    world_coor[:-1] = camera_dict['world_coordinate_origin_offset']
    cal_utm = aj_world_coordinate + world_coor[:, None]
    # Geotation matrix
    rmatx = cv2.Rodrigues(camera_dict['extrinscis_R'])
    # Get camera coordinate from UTM coordinate
    camera_coor = np.dot(rmatx[0], cal_utm) + camera_dict['extrinscis_T']
    return camera_coor

if __name__ == '__main__':
    FILE = r"./output_pose_GP010170_10_cut"
    with open(FILE, 'rb') as fp:
        data = pickle.load(fp)
    max_num_person = 0
    for num_person_list in data[1]:
        max_num_person = max(max_num_person, len(num_person_list))

    data = data[1]
    # All ankle points
    ankles_exist = False
    for fr in range(len(data)):
        for i in range(len(data[fr])):
            # left_ankle = data[:, fr, 3, :]
            left_ankle = np.array([data[fr][i][10, :]])
            real_left_ankle = left_ankle.copy()
            real_left_ankle[0, 1] = left_ankle[0, 2]
            real_left_ankle[0, 2] = -left_ankle[0, 1]
            # right_ankle = data[:, fr, 6, :]
            right_ankle = np.array([data[fr][i][13, :]])
            real_right_ankle = right_ankle.copy()
            real_right_ankle[0, 1] = right_ankle[0, 2]
            real_right_ankle[0, 2] = -right_ankle[0, 1]
            if not ankles_exist:
                ankles = np.concatenate((real_left_ankle, real_right_ankle), axis=0)
                ankles_exist = True
            else:
                fr_ankles = np.concatenate((real_left_ankle, real_right_ankle), axis=0)
                ankles = np.concatenate((ankles, fr_ankles), axis=0)

    points = ankles[~np.isnan(ankles).any(axis=1), :]
    points = points/1000
    x_interval = [-30,20]
    y_interval = [0, 60]

    # Get ground plane in camera space
    data_path = r"../Extrinsics_calculation/output/GP010170.MP4_camera_data.pkl"
    camera_coor = get_points_on_ground_plane(data_path)
    camera_coor_x = camera_coor[0, :]
    camera_coor_y = camera_coor[2, :]
    camera_coor_z = -camera_coor[1, :]
    camera_coor_array_tuple = (camera_coor_x, camera_coor_y, camera_coor_z)

    camera_coor_right = np.vstack(camera_coor_array_tuple)
    camera_coor_points = camera_coor_right.T
    camera_coor_xx, camera_coor_yy, camera_coor_zz, camera_coor_xs, camera_coor_ys, camera_coor_zs = plane_regression(camera_coor_points, x_interval=x_interval, y_interval=y_interval)

    xx, yy, zz, xs, ys, zs = plane_regression(points, x_interval=x_interval, y_interval=y_interval)
    point1 = np.array([xx[0][0], yy[0][0], zz[0][0]])
    point2 = np.array([xx[0][1], yy[0][1], zz[0][1]])
    point3 = np.array([xx[1][0], yy[1][0], zz[1][0]])
    point4 = np.array([xx[1][1], yy[1][1], zz[1][1]])
    distance = []
    for i in range(points.shape[0]):
        point5 = points[i]
        d1 = point2area_distance(point1, point2, point3, point5)
        distance.append(d1)

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Plot of ankle points distribution
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(elev=15., azim=70)  # initialize the view angle
    ax.set_title('Ankle points distribution')
    scat = ax.scatter(xs, ys, zs, c=distance, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(scat, shrink=0.8, aspect=15)
    # ax.plot_surface(xx, yy, zz, alpha=0.2, color=[0, 1, 0])

    ax.scatter(camera_coor_x, camera_coor_y, camera_coor_z)
    ax.plot_surface(camera_coor_xx, camera_coor_yy, camera_coor_zz, alpha=0.3, color=[0, 1, 0])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)
    # ax.set_zlim(  0,10)
    # Plot histogram of distance between ankle points and plane
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Histogram of distance between ankle points and plane')
    ax.set(xlabel='Distance', ylabel='Amount of points')
    plt.hist(distance, bins=50)
    plt.show()