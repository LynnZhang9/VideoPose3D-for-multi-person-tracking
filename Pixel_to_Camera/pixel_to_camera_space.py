import pickle
import numpy as np
import cv2
import functools
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


data_path = r"../Extrinsics_calculation/output/GP010170.MP4_camera_data.pkl"
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


print('plot starts')
x = camera_coor[0,:]
y = camera_coor[1,:]
z = camera_coor[2,:]

points = camera_coor.T
xx, yy, zz, xs, ys, zs = plane_regression(points, x_interval=[-30,10], y_interval=[-10,10])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz, alpha=0.2, color=[0, 1, 0])
ax.set_title('Ground plane in camera space')
ax.scatter(x,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
print('plot ends')
