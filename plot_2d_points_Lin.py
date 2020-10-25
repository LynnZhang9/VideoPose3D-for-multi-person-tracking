import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

FILE = "output_3d_keypoints.npy"
# FILE = "output_3d_after_image_coordinates.npy"
data = np.load(FILE)

dataa = np.insert(data, 2, values=0, axis=3)
# def update_plot(i, data, scat):
#     scat.set_array(data[i])
#     return scat,
#
#
# numframes = data.shape[1]
# numpoints = 10
# color_data = np.random.random((numframes, numpoints))
# x, y, c = np.random.random((3, numpoints))
#
# fig = plt.figure()
# scat = plt.scatter(x, y, c=c, s=100)
#
# ani = animation.FuncAnimation(fig, update_plot, frames=range(data.shape[1]),
#                               fargs=(data, scat))
# plt.show()
artists = []

def plot_at_frame(fr):
    global artist, ax
    x, y, z = dataa[:, fr, :, :].T
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
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])

a = animation.FuncAnimation(fig, plot_at_frame, frames=range(dataa.shape[1]), repeat=True)
plt.show()