"""
Plot animation of 3d VideoPose3D output npy file.
Auther: Lin
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.optimize
import functools
import pickle
import cv2


FILE = r"../output_pose_GP010170_10_cut"
with open (FILE, 'rb') as fp:
    data = pickle.load(fp)
max_num_person = 0
for num_person_list in data[1]:
    max_num_person = max(max_num_person, len(num_person_list))

data = data[1]

# # pose color
# keypoint_num = 17
# joints_right_2d = [2, 4, 6, 8, 10, 12, 14, 16]
# colors_2d = np.full(keypoint_num, 'black')
# colors_2d[joints_right_2d] = 'red'
# mul_colors_2d = np.expand_dims(colors_2d, 0).repeat(data.shape[0], axis=0)
# colors_2d = mul_colors_2d.reshape(data.shape[0] * data.shape[2])
# parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
# skeleton_joints_right = [1, 2, 3, 14, 15, 16]

# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2),
(2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))

initialized = False


plane_3d = []
plane_3d.append([])
lines_3d = []
lines_3d.append([])
artists = []


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
    return xx, yy, z
def get_points_on_ground_plane(camera_dict):
    # data_path = r"../Extrinsics_calculation/output/GP010170.MP4_camera_data.pkl"

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

def plot_at_frame(fr):
    # nonlocal initialized
    global artist, ax, lines_3d, plane_3d
    vis_kps = np.array(data[fr])/1000
    x, y, z = vis_kps.T
    kps_lines = skeleton
    kpt_3d = vis_kps
    kpt_3d_vis = np.ones_like(vis_kps)
    print('keyframes: {}'.format(fr))
    if len(artists) == 0:
        #Plane regression

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
        # points = points / 1000


        #Keypoint
        for i in range(len(x)):
                artists.append(ax.scatter(x[i], z[i], -y[i], label=str(i)))

        # Skeleton
        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]

            for n in range(max_num_person):
                if n < kpt_3d.shape[0]:
                    lines_3d[0].append(ax.plot([x[i1][n], x[i2][n]],
                                               [z[i1][n], z[i2][n]],
                                               [-y[i1][n], -y[i2][n]], zdir='z', c=colors[l]))
                else:
                    lines_3d[0].append(ax.plot([np.nan, np.nan],
                                               [np.nan, np.nan],
                                               [np.nan, np.nan], zdir='z', c=colors[l]))
        ax.legend()
    else:
        #Keypoint
        for i, artist in enumerate(artists):
            artist._offsets3d = (x[i], z[i], -y[i])

        #Skeleton

        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]

            # for n in range(person_num):
            for n in range(max_num_person):
                if n < kpt_3d.shape[0]:
                    lines_3d[0][n * 20 + l][0].set_xdata(np.array([x[i1][n], x[i2][n]]))
                    lines_3d[0][n * 20 + l][0].set_ydata(np.array([z[i1][n], z[i2][n]]))
                    lines_3d[0][n * 20 + l][0].set_3d_properties(np.array([-y[i1][n], -y[i2][n]]),zdir='z')
                else:
                    lines_3d[0][n * 20 + l][0].set_xdata(np.array([np.nan, np.nan]))
                    lines_3d[0][n * 20 + l][0].set_ydata(np.array([np.nan, np.nan]))
                    lines_3d[0][n * 20 + l][0].set_3d_properties(np.array([np.nan, np.nan]),zdir='z')



    return artists

if __name__ == '__main__':
    # Get ground plane in camera space
    data_path = r"../../Extrinsics_calculation/output/GP010170.MP4_camera_data.pkl"
    with open(data_path, "rb") as input_file:
        camera_dict = pickle.load(input_file)


    x_interval = [-30, 20]
    y_interval = [0, 60]
    # xx, yy, zz = plane_regression(points, x_interval, y_interval)
    #
    # plane_3d[0] = ax.plot_surface(xx, yy, zz, alpha=0.2, color=[0, 1, 0])
    camera_coor = get_points_on_ground_plane(camera_dict)
    camera_coor_x = camera_coor[0, :]
    camera_coor_y = camera_coor[2, :]
    camera_coor_z = -camera_coor[1, :]
    # camera_coor_y = camera_coor[1, :]
    # camera_coor_z = camera_coor[2, :]
    camera_coor_array_tuple = (camera_coor_x, camera_coor_y, camera_coor_z)

    camera_coor_right = np.vstack(camera_coor_array_tuple)
    camera_coor_points = camera_coor_right.T
    camera_coor_xx, camera_coor_yy, camera_coor_zz = plane_regression(
        camera_coor_points, x_interval=x_interval, y_interval=y_interval)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    # ax.view_init(elev=15., azim=70) # initialize the view angle

    ax.scatter(camera_coor_x, camera_coor_y, camera_coor_z)
    ax.plot_surface(camera_coor_xx, camera_coor_yy, camera_coor_zz, alpha=0.2, color=[0, 1, 0])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim([-10, -3])
    # ax.set_ylim([-10, -3])
    # ax.set_zlim([0, 7])

    a = animation.FuncAnimation(fig, plot_at_frame, frames=range(len(data)), repeat=True)
    plt.show()