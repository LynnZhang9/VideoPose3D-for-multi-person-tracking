"""
Plot animation of 3d VideoPose3D output npy file.
Auther: Patrick and Lin

"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# FILE = "output_3d.npy"
# FILE = "output_3d_predictions.npy"
# FILE = "output_3d_keypoints.npy"
# FILE = "output_3d_after_image_coordinates.npy"
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

lines_3d = []
lines_3d.append([])
artists = []






def plot_at_frame(fr):
    # nonlocal initialized
    global artist, ax, lines_3d
    x, y, z = data[:, fr, :, :].T
    if len(artists) == 0:
        for i in range(len(x)):
            artists.append(ax.scatter(x[i], y[i], z[i], label=str(i)))

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
        # keypoint
        for i, artist in enumerate(artists):
            artist._offsets3d = (x[i], y[i], z[i])

        # pose
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
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])

a = animation.FuncAnimation(fig, plot_at_frame, frames=range(data.shape[1]), repeat=True)
plt.show()