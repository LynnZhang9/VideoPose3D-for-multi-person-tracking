# Introduction
1. This folder shows the extra scripts and folders, which are different from the public 3DMPPE_ROOTNET_RELEASE github project.
2. With those scripts, people can export the root points for each frame of input video.

# How to run the scripts?
1. git clone public 3DMPPE_ROOTNET_RELEASE github project from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE.
2. copy all scripts and folder to the same path respect to root of public 3DMPPE_ROOTNET_RELEASE github project.

# Directory

## Root
The ${POSE_ROOT} is described as below.

${POSE_ROOT}
|-- data
|-- demo
|-- common
|-- main
|-- output

## demo
${POSE_ROOT}
|-- demo
|   |-- snapshot_7.pth.tar (a trained model extracted from RootNet-20210103T143641Z-001.zip)
|   |-- snapshot_18.pth.tar (a trained model extracted from RootNet-20210103T143641Z-001.zip)
|   |-- snapshot_19.pth.tar (a trained model extracted from RootNet-20210103T143641Z-001.zip)
|   |-- demo_test.py
|   |-- data_mul
|   |   |-- all_keyframes_root_3d_GP010170 (root points exported from demo_test.py)
|   |   |-- data_2d_custom_myvideos_GP010170.npz (2d skeletons exported from detectron_pose_predictor_demo_no_interpolation.py)

Note: RootNet-20210103T143641Z-001.zip is uploaed in Teams channel

# demo_test.py
## Input
data_path = 'data_mul/data_2d_custom_myvideos_GP010170.npz'

focal = [2.12902613e+03, 2.10204718e+03]

img_generator = read_video('/home/lin/Videos/GP010170.MP4')

## Output
output_path = 'data_mul/all_keyframes_root_3d_GP010170'


