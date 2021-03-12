import os, glob
import json
import numpy as np
import cv2
import pickle

# Input parameters
#VIDEO_FILE = "F:\\Marienplatz-videos_Tag_3&4\\Verkehrsbeobachtung_20190523\\NEU\\Poses_Kamera 1 (Stativ)\\GH013110.mp4"
VIDEO_FILE = "/media/lin/Seagate Backup Plus Drive/Lin_videos/Lin_videos/Theresien_Ludwig/GP010170.MP4"
# POINT_FILE = os.path.join("georectify", "geo_points.geojson")
POINT_FILE = os.path.join("georectify", "Theresien_Ludwig_UTM.geojson")
# Hero3-matrix
# [2.12902613e+03   0.00000000e+00   9.50917979e+02
#  0.00000000e+00   2.10204718e+03   5.35954497e+02
#  0.00000000e+00   0.00000000e+00   1.00000000e+00]
# Hero3-dist
# -8.98305625e-01   6.06958917e-01 0  0  0
# fx, fy = 2.422557950067380e+03, 2.453137175698948e+03  # focal lengths
fx, fy = 2.12902613e+03, 2.10204718e+03  # focal lengths
# cx, cy = 1.620388695920248e+03, 9.925668485438835e+02  # principal point
cx, cy = 9.50917979e+02, 5.35954497e+02  # principal point
# k1, k2, k3 = 0.031290733197451, -0.011385417172631, 0.001015770876776  # radial distortion coefficients
k1, k2, k3 = -8.98305625e-01, 6.06958917e-01, 0  # radial distortion coefficients
# p1, p2 = 0.002185718337131, 0.003178018767407  # tangential distortion coefficients
p1, p2 = 0, 0   # tangential distortion coefficients

# Get input video and frame
cap = cv2.VideoCapture(VIDEO_FILE)
success, frame = False, None
while not success:
    success, frame = cap.read()

# Read world-point coordinates from geojson file
with open(POINT_FILE, "r") as f:
    point_json = json.load(f)
# UTM coordinate with 33N zone
world_points = [tuple(c["geometry"]["coordinates"][0]) for c in point_json["features"]]
image_points = []
world_pointswith_z = [tuple(c["geometry"]["coordinates"][0])+(0,) for c in point_json["features"]]

def on_click(event, x, y, flags, param):
    global image_points
    if event == cv2.EVENT_LBUTTONDOWN:
        image_points.append(tuple([x, y]))
        print("Point " + str(len(image_points)) + ": " + str(x) + ", " + str(y))
# Point 1: 421, 284
# Point 2: 764, 346
# Point 3: 241, 362
# Point 4: 552, 441
# Point 5: 184, 394
# Point 6: 512, 478
# Point 7: 348, 706
# Point 8: 392, 724
# Point 9: 152, 339
# Point 10: 94, 368

# Undistort image
h, w, _ = frame.shape
camera_matrix = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]], dtype="double")
distortion_coeffs = np.array([k1, k2, p1, p2, k3])
uFrame = cv2.undistort(frame, camera_matrix, distortion_coeffs)

# Get User-Selected Image Points
cv2.namedWindow("Select Points", cv2.WINDOW_NORMAL)
cv2.imshow("Select Points", uFrame)
cv2.setMouseCallback("Select Points", on_click)
cv2.resizeWindow("Select Points", 1080, 720)
while True:
    k = cv2.waitKey(1)
    if k == ord("q"):
        exit(1)
    elif len(image_points) == len(world_points):
        cv2.destroyAllWindows()
        break
    else:
        disp_img = uFrame.copy()
        for i, pt in enumerate(image_points):
            cv2.circle(disp_img, pt, 6, (255, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(disp_img, str(i+1), pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 3)
            cv2.imshow("Select Points", disp_img)

world_points = np.array(world_points)
world_pointswith_z = np.array(world_pointswith_z)
world_points_origin = np.min(world_points, axis=0)
print("World Coordinates Origin Offset:", world_points_origin)
world_points = world_points - world_points_origin
max_world_coords = np.max(world_points, axis=0)
world_view_buffer = 60
image_points = np.array(image_points)
homography, mask = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
print("Homography:\n", homography)

# Get extrinscis parameters
image_points = np.array(image_points, dtype=np.float32)
_, rvecs, tvecs, inliers = cv2.solvePnPRansac(world_pointswith_z, image_points, camera_matrix, distortion_coeffs)
print("Rotation:\n", rvecs)
print("Translation:\n", tvecs)


output = dict()
output["image_points"] = image_points
output["camera_matrix"] = camera_matrix
output["distortion_coeffs"] = distortion_coeffs
output["img_shape"] = frame.shape
output["world_coordinate_origin_offset"] = world_points_origin
output["homography"] = homography
output["extrinscis_R"] = rvecs
output["extrinscis_T"] = tvecs
with open(os.path.join("output", VIDEO_FILE.split(os.sep)[-1]+"_camera_data.pkl"), "wb") as of:
    pickle.dump(output, of)

cv2.destroyAllWindows()