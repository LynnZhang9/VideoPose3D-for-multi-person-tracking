"""
Plot animation of 3d VideoPose3D output npy file.
Auther: Patrick and Lin

"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.optimize
import functools

# FILE = "output_3d.npy"
# FILE = r"./temp_output/output_3d_predictions.npy"
# FILE = r"./temp_output/output_3d_keypoints.npy"
# FILE = r"./temp_output/output_3d_after_image_coordinates.npy"
FILE = r"./temp_output/output_3d_after_camera_to_world.npy"

data = np.load(FILE)

# pose color
keypoint_num = 17
joints_right_2d = [2, 4, 6, 8, 10, 12, 14, 16]
colors_2d = np.full(keypoint_num, 'black')
colors_2d[joints_right_2d] = 'red'
mul_colors_2d = np.expand_dims(colors_2d, 0).repeat(data.shape[0], axis=0)
colors_2d = mul_colors_2d.reshape(data.shape[0] * data.shape[2])
parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
skeleton_joints_right = [1, 2, 3, 14, 15, 16]

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
    return xx, yy, z




def plot_at_frame(fr):
    # nonlocal initialized
    global artist, ax, lines_3d, plane_3d
    x, y, z = data[:, fr, :, :].T
    if len(artists) == 0:
        #Plane regression
        # plane_3d.remove()
        # left_ankle = data[:, fr, 15, :]
        # right_ankle = data[:, fr, 16, :]
        # ankles =np.concatenate((left_ankle, right_ankle), axis=0)
        # points = ankles[~np.isnan(ankles).any(axis=1), :]

        # All ankle points
        ankles_exist = False
        for fr in range(data.shape[1]):
            left_ankle = data[:, fr, 3, :]
            right_ankle = data[:, fr, 6, :]
            if not ankles_exist:
                ankles = np.concatenate((left_ankle, right_ankle), axis=0)
                ankles_exist = True
            else:
                fr_ankles = np.concatenate((left_ankle, right_ankle), axis=0)
                ankles = np.concatenate((ankles, fr_ankles), axis=0)

        points = ankles[~np.isnan(ankles).any(axis=1), :]

        xx, yy, zz = plane_regression(points)

        plane_3d[0] = ax.plot_surface(xx, yy, zz, alpha=0.2, color=[0, 1, 0])

        #Keypoint
        for i in range(len(x)):
            artists.append(ax.scatter(x[i], y[i], z[i], label=str(i)))
        #Skeleton
        for mul in range(x.shape[1]):
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = 'red' if j in skeleton_joints_right else 'black'

                lines_3d[0].append(ax.plot([x[j][mul], x[j_parent][mul]],
                                           [y[j][mul], y[j_parent][mul]],
                                           [z[j][mul], z[j_parent][mul]], zdir='z', c=col))
        ax.legend()
    else:
        # #Plane regression
        # plane_3d[0].remove()
        # left_ankle = data[:, fr, 15, :]
        # right_ankle = data[:, fr, 16, :]
        # ankles =np.concatenate((left_ankle, right_ankle), axis=0)
        # points = ankles[~np.isnan(ankles).any(axis=1), :]
        # xx, yy, zz = plane_regression(points)
        # plane_3d[0] = ax.plot_surface(xx, yy, zz, alpha=0.2, color=[0, 1, 0])

        #Keypoint
        for i, artist in enumerate(artists):
            artist._offsets3d = (x[i], y[i], z[i])

        #Skeleton
        for mul in range(x.shape[1]):
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                lines_3d[0][mul * 16 + j - 1][0].set_xdata(np.array([x[j][mul], x[j_parent][mul]]))
                lines_3d[0][mul * 16 + j - 1][0].set_ydata(np.array([y[j][mul], y[j_parent][mul]]))
                lines_3d[0][mul * 16 + j - 1][0].set_3d_properties(np.array([z[j][mul], z[j_parent][mul]]),zdir='z')

    return artists


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15., azim=70) # initialize the view angle
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_xlim([-10, -3])
# ax.set_ylim([-10, -3])
# ax.set_zlim([0, 7])

a = animation.FuncAnimation(fig, plot_at_frame, frames=range(data.shape[1]), repeat=True)
plt.show()