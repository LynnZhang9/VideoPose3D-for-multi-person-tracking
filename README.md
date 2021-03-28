## VideoPose3D-for-multi-person-tracking
This project aims at adjusting the VideoPose3D project from Dario Pavllo, in order to track the trajectories of multiple people and predict the keypoints of their skeletons with respective to their root points.

## How to use it?

1. Git clone the VideoPose3D project:
	https://github.com/facebookresearch/VideoPose3D
2. After we finish Step 4: creating a custom dataset mentioned in INFERENCE.md of VideoPose3D project, we replace run.py and relative files (arguments.py and visualization.py in 'common' folder)


## New update: New models Root_PoseNET_result and Root_ROOTNET_result are applied to extract the mutiple 2d pixel poses and predict 3d poses in camera coordinate.
Steps of Pipeline for getting 3d pose estimation in cameraspace with Root-pose net model. 

1. Using detectron2 to extract the 2d pixel coordinate of keypoints from GP010170.MP4 (by Detectron_pose_predictor/detectron_pose_predictor_demo_no_interpolation.py). Output npz is created in Detectron_pose_predictor/npz_output/data_2d_custom_myvideos_GP010170.npz. 

2. Create a rendered video in rendered_video_output/rendered_GP010170.MP4 by applying Detectron_pose_predictor/visualization_demo_no_interpolation.py. Hence, you can see the recognized objects in data_2d_custom_myvideos_GP010170.npz vividly. 

3. Move data_2d_custom_myvideos_GP010170.npz to 3DMPPE_ROOTNET_RELEASE/demo/data_mul (create a folder named data_mul). Adjust the data_path, output_path, focal and video path of img_generator in 3DMPPE_ROOTNET_RELEASE/demo/demo_test.py, then run it. All_keyframes_root_3d_GP010170 is created in data_mul folder.

4. Copy the data_2d_custom_myvideos_GP010170.npz and all_keyframes_root_3d_GP010170 to 3DMPPE_POSENET_RELEASE/demo/data_mul folder. Adjust the data_path, root_depth_path, output_path, focal and video path of img_generator in 3DMPPE_POSENET_RELEASE/demo/demo_test.py, then run it. output_pose_GP010170 is created in 3DMPPE_POSENET_RELEASE/demo folder. 

5. Copy the output_pose_GP010170_10_cut to VideoPose3D-for-multi-person-tracking/Root_PoseNET_result 

6. Run foot_point_distribution_root_pose_net.py with data output_pose_GP010170. Regressed plane of footpoints and footpoints can be display in foot_point_distribution_root_pose_net.py. 

7. Run ground_plane_with_foot_point_distribution.py with data output_pose_GP010170. Then we can show the ground plane and foot points by  ground_plane_with_foot_point_distribution.py.

8. Run ground_plane_with_skeletons.py with data output_pose_GP010170. It can indicate both ground plane and moving skeletons in camera coordiante.

9. Run ground_truth_generation.py with data output_pose_GP010170. This script can move skeletons back on the true ground plane in camera coordiante and export the adjusted 3d skeletons.

10. After 9.step, run plot_adjusted_skeletons.py with data Ground_truth_GP010170. It can indicate both ground plane and adjusted moving skeletons in camera coordiante.
 


