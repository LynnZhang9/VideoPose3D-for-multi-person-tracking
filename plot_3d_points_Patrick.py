"""
Plot animation of 3d VideoPose3D output npy file.
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# FILE = "output_3d.npy"
# FILE = "output_3d_predictions.npy"
# FILE = "output_3d_keypoints.npy"
# FILE = "output_3d_after_image_coordinates.npy"
FILE = "output_3d_after_camera_to_world.npy"

data = np.load(FILE)

artists = []

def plot_at_frame(fr):
    global artist, ax
    x, y, z = data[:, fr, :, :].T
    if len(artists) == 0:
        for i in range(len(x)):
            artists.append(ax.scatter(x[i], y[i], z[i], label=str(i)))
        ax.legend()
    else:
        for i, artist in enumerate(artists):
            artist._offsets3d = (x[i], y[i], z[i])
    return artists


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

a = animation.FuncAnimation(fig, plot_at_frame, frames=range(data.shape[1]), repeat=True)
plt.show()