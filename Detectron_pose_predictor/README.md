# This folder contains the scripts of 2d skeletons extraction and visualization. The model are obtained from detectron2 project, in order to extract 2d skeletons in pixel coordinate for each frame of imported video.

# detectron_pose_predictor_demo_no_interpolation.py
Export the data of 2d skeletons in pixel coordinate for each frame of imported video
## Input
model_config_path = '/your/path/to/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'

model_weights_path = '/your/path/to/model_final_5ad38f.pkl'

img_generator = read_video('/your/path/to/GP010170.MP4')

output_path = './npz_output/data_2d_custom_myvideos_GP010170'

traj_output_path = './npz_output/dummy_traj_GP010170'

Note: 
1. dummy_traj_GP010170 should be the output of trajectory but can not be exported, since the trajectory algorithm was not implemented sucessfully in this script. Hence, all relative codes are commented.
2. model_config and model_weights are uploaded in Teams channel named 2d_skeleton_extraction_models.zip.

## Output
npz data: /npz_output/data_2d_custom_myvideos_GP010170

# visualization_demo_no_interpolation.py
Visualize 2d skeletons in pixel coordinate for each frame of imported cutted video
## Input
keypoints_npz_path = './npz_output/data_2d_custom_myvideos_GP010170_10_cut.npz'

img_generator = read_video('/home/lin/Videos/GP010170_10_cut.MP4')

traj_output_path = './npz_output/dummy_traj_GP010170_10_cut'

## Output
mp4_output_path = './rendered_video_output/rendered_GP010170_10_cut.MP4'



