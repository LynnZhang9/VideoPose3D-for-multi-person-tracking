import os
import numpy as np
import cv2
import subprocess as sp
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#NN-tracking
from scipy.optimize import linear_sum_assignment
import pandas as pd
from PIL import Image

# SETTINGS =============================================================================================================

MAX_LOOKAHEAD = 4  # Number of frames ahead of current frame for which to consider candidate points
MAX_SPEED = 10000  # 100  # Maximum distance points can move per frame to be considered for connection
INF = 10000  # "Infinite cost" value for point pairs which should not be considered for connection.


# MAIN CODE ============================================================================================================


def get_img_paths(imgs_dir):
	img_paths = []
	for dirpath, dirnames, filenames in os.walk(imgs_dir):
		for filename in [f for f in filenames if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]:
			img_paths.append(os.path.join(dirpath,filename))
	img_paths.sort()

	return img_paths

def read_images(dir_path):
	img_paths = get_img_paths(dir_path)
	for path in img_paths:
		yield cv2.imread(path)


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)


def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def init_pose_predictor(config_path, weights_path, cuda=True):
	cfg = get_cfg()
	cfg.merge_from_file(config_path)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = weights_path
	if cuda == False:
		cfg.MODEL.DEVICE='cpu'
	predictor = DefaultPredictor(cfg)

	return predictor


#NN-tracking
def get_connection_back(m, j, connections):
    """
    Finds the frame-pid pair which precedes and connects to the given frame-pid pair.
    :param m: current frame
    :param j: current pid
    :param connections: list of connections of form ((k, i), (m, j)), where k & m are frame numbers, and i & j are pids
    :return: None if no previous connection, else (k, i), where k is the frame number and i is the pid
    """
    for cxn in reversed(connections):
        if cxn[1] == (m, j):
            return cxn[0]
    return None

def has_connection_back(m, j, connections):
    """
    Checks whether a given frame-pid pair has a preceding connection.
    :param m: current frame
    :param j: current pid
    :param connections: list of connections of form ((k, i), (m, j)), where k & m are frame numbers, and i & j are pids
    :return:
    """
    return get_connection_back(m, j, connections) is not None

def calculate_cost_matrix(skeletons, k):
	"""
	Calculates a cost matrix for the linear assignment optimization.
	Each row corresponds to a point in the current frame.
	Each column corresponds to a point in the next MAX_LOOKAHEAD frames.
	:param skeletons: Skeletons object containing the skeleton data from the input file.
	:param k: current frame
	:return: cost matrix
	:type skeletons: SkeletonTools.Skeletons
	"""
	# Get current and candidate points
	points = skeletons[k, :, :]
	# print('k: {}'.format(k))
	# print("points: {}".format(points))
	candidates = skeletons[k+1:min(k+1+MAX_LOOKAHEAD, skeletons.shape[0]), :, :]
	# Flatten candidate points along frame-axis
	candidates_flat = np.reshape(candidates, newshape=(1, -1, 2), order="C")[0]
	# Calculate frame difference penalties for each point: MAX_SPEED*(m-k-1)
	time_penalties = np.floor(np.arange(candidates_flat.shape[0])/MAX_LOOKAHEAD) * MAX_SPEED
	# Create cost matrix
	cost_matrix = np.zeros((points.shape[0], candidates_flat.shape[0]))
	for i in range(points.shape[0]):  # here, i is row index (current points)
		if True in np.isnan(points[i]):
			cost_matrix[i, :] = INF  # Give high cost to missing points
			continue
		for j in range(candidates_flat.shape[0]):  # here, j is column index (candidate points)
			# Calculate cost as Euclidean distance plus time penalty
			cost = np.linalg.norm(candidates_flat[j]-points[i]) + time_penalties[j]
			cost_matrix[i, j] = cost if not np.isnan(cost) else INF
	return cost_matrix

def penalize_connected_points(cost_matrix, k, connections):
	"""
	Adjust a cost matrix by assigning high cost to candidate points which have already been connected.
	:param cost_matrix: the cost matrix to adjust
	:param k: current frame
	:param connections: list of connections
	:return: None
	"""
	n_points, n_candidates = cost_matrix.shape
	for n in range(n_points):
		j = int(n % n_points)
		m = int(k + 1 + np.floor(n/n_points))
		if has_connection_back(m, j, connections):
			cost_matrix[n, :] = INF

def make_connections(k, cost_matrix):
	"""
	Perform linear sum assignment and return list of connections made.
	:param k: current frame
	:param cost_matrix: cost matrix
	:return: list of connections of form ((k, i), (m, j)), where k & m are frame numbers, and i & j are pids
	"""
	# Pad the cost matrix with one extra column per current point. These signify "no connection"
	padded_cost_matrix = np.pad(cost_matrix, ((0, 0), (0, cost_matrix.shape[0])), "constant", constant_values=MAX_SPEED*(MAX_LOOKAHEAD+1))
	# sns.heatmap(padded_cost_matrix, cmap="rainbow", linewidths=1, annot=True, vmax=MAX_SPEED*(MAX_LOOKAHEAD+1))
	# plt.show()
	# Perform linear sum assignment
	row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
	# Create list of connections from result of linear sum assignment
	cxns = []
	n_points, n_candidates = cost_matrix.shape
	for i in range(len(row_ind)):
		if col_ind[i] >= cost_matrix.shape[1]:
			continue  # Skip "no connection" connections (padded columns)
		if cost_matrix[row_ind[i], col_ind[i]] < (MAX_LOOKAHEAD+1)*MAX_SPEED:
			j = int(col_ind[i] % n_points)
			m = int(k + 1 + np.floor(col_ind[i]/n_points))
			cxns.append(((k, row_ind[i]), (m, j)))
	return cxns

def NN_tracking():
	return

def get_next_index(connections, point):
	"""
	Gets the index in the given connections list for which the corresponding connection starts at the given point.
	:param connections: list of connections of form ((k, i), (m, j)), where k & m are frame numbers, and i & j are pids
	:param point: (k, i) pair for which to search connections list
	:return: None if no matching connection found, else the index of the matching connection
	"""
	for i, cxn in enumerate(connections):
		if cxn[0] == point:
			return i
	return None


def trace(connections, tr):
	"""
	Recursively generate continuous "traces" from a connections list.
	A trace is of the form [(k0, i0), (k1, i1), ..., (kn, in)], where k are frame numbers and i are pids
	:param connections: list of connections of form ((k, i), (m, j)), where k & m are frame numbers, and i & j are pids
	:param tr: currently in-progress trace
	:return: (trace, connections), where used connections have been removed from connections list
	"""
	start = tr[-1]
	next_point_index = get_next_index(connections, start)
	if next_point_index is None:
		return tr, connections
	else:
		_, next_point = connections.pop(next_point_index)
		tr.append(next_point)
		return trace(connections, tr=tr)

def trace_connections(skeletons, connections):
	"""
	Generate DataFrame containing the traced trajectories.
	:param skeletons: Skeletons object containing data from input file.
	:param connections: list of connections to trace.
	:return: DataFrame with three columns per trajectory: skeleton id, x, and y. Index is frame number.
	"""
	# Generate traces by calling trace() on the first connection in connections list.
	# This is done until connections list is empty (calls to trace() remove used connections from list)
	traces = []
	while len(connections) > 0:
		start_pt = connections[0][0]
		tr, connections = trace(connections, tr=[start_pt])
		traces.append(tr)
	# Generate dict from trace list, which will allow fast generation of output DataFrame
	output_dict = dict()
	for i, tr in enumerate(traces):
		frames, pids = zip(*tr)
		frames, pids = list(frames), list(pids)
		output_dict["traj"+str(i)+"_pid"] = {frame: pid for frame, pid in tr}
		output_dict["traj"+str(i)+"_x"] = {frame: x for frame, x in zip(frames, skeletons[frames, pids, 0])}
		output_dict["traj"+str(i)+"_y"] = {frame: y for frame, y in zip(frames, skeletons[frames, pids, 1])}
	# Generate output DataFrame from dict
	output_df = pd.DataFrame(output_dict)
	output_df.index.name = "frame"
	return output_df

def NN_tracing():
	return

def encode_for_videpose3d(boxes,keypoints,resolution, dataset_name, max_num_person, traj_output_path):
	# Generate metadata:
	metadata = {}
	metadata['layout_name'] = 'coco'
	metadata['num_joints'] = 17
	metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
	metadata['video_metadata'] = {dataset_name: resolution}

	
	prepared_boxes_all = []
	prepared_keypoints_all = []
	for k in range(len(boxes)):
		prepared_boxes = []
		prepared_keypoints = []
		#for i in range(len(boxes[k])):#number of person in this frame
		for i in range(max_num_person):
			if i < len(boxes[k]):
				if len(boxes[k][i]) == 0 or len(keypoints[k][i]) == 0:
					# No bbox/keypoints detected for this frame -> will be interpolated
					prepared_boxes.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
					prepared_keypoints.append(np.full((17, 3), np.nan, dtype=np.float32)) # 17 COCO keypoints
					
					continue

				prepared_boxes.append(boxes[k][i])
				prepared_keypoints.append(keypoints[k][i])

			else:
				prepared_boxes.append(np.full(4, np.nan, dtype=np.float32))
				prepared_keypoints.append(np.full((17, 3), np.nan, dtype=np.float32))
			
		prepared_boxes = np.array(prepared_boxes, dtype=np.float32)
		prepared_keypoints = np.array(prepared_keypoints, dtype=np.float32)


		
		if len(boxes[k]) == 0 or len(keypoints[k]) == 0:
			prepared_boxes_all.append(np.full((max_num_person,4,), np.nan, dtype=np.float32))
			prepared_keypoints_all.append(np.full((max_num_person,17,2), np.nan, dtype=np.float32))

			continue


		prepared_boxes_all.append(prepared_boxes)
		prepared_keypoints_all.append(prepared_keypoints[:,:,:2])



	boxes_all = np.array(prepared_boxes_all, dtype=np.float32)
	#keypoints = np.array(prepared_keypoints, dtype=np.float32)
	keypoints_all = np.array(prepared_keypoints_all, dtype=np.float32)
	keypoints_all = keypoints_all[:, :, :, :2] # Extract (x, y)
	
	# Fix missing bboxes/keypoints by linear interpolation
	mask = ~np.isnan(boxes_all[:, :, 0])
	mask = np.squeeze(mask)

	# indices = np.arange(len(boxes_all))


	
	# for m in range(len(mask[0])):
	# 	for i in range(4):
	# 		boxes_all[:, m, i] = np.interp(indices, indices[mask[:,m]], boxes_all[mask[:,m], m, i])
	# 	for i in range(17):
	# 		for j in range(2):
	# 			keypoints_all[:, m, i, j] = np.interp(indices, indices[mask[:,m]], keypoints_all[mask[:,m], m, i, j])
	
	print('{} total frames processed'.format(len(boxes_all)))
	print('{} frames were interpolated'.format(np.sum(~mask)))
	print('----------')
	
	print("Loading skeletons from keypoints_all")

	########################################
	skeletons = keypoints_all[:,:,11,:]# read in skeletons as single points

	connections = []  # list of connections of form ((k, i), (m, j)) where k and m are frame numbers and i and j are point indices
	MAX_FRAME = len(skeletons)-1
	print("Performing NN Tracking ("+str(MAX_FRAME)+" frames)...")
	progress_frames = np.linspace(0, MAX_FRAME, num=11).astype(int)  # frames at which to show progress message (every ~10%)
	# Main loop: iterate through each frame and find connections
	for k in range(MAX_FRAME):
		if k in progress_frames:
			print("\tk = " + str(k))
		cost_matrix = calculate_cost_matrix(skeletons, k)
		penalize_connected_points(cost_matrix, k, connections)
		connections += make_connections(k, cost_matrix)
	print()
	print("...NN Tracking complete")
	print("Tracing Trajectories...")
	trajectories = trace_connections(skeletons, connections)
	
	print("Saving trajectory output file ")
	trajectories.to_hdf(traj_output_path, "trajectories")

	
	return [{
		'start_frame': 0, # Inclusive
		'end_frame': len(keypoints_all), # Exclusive
		'bounding_boxes': boxes_all,
		'keypoints': keypoints_all,
		'trajectories': trajectories
	}], metadata


def predict_pose(pose_predictor, img_generator, output_path, traj_output_path, dataset_name='detectron2'):
	'''
		pose_predictor: The detectron's pose predictor
		img_generator:  Images source
		output_path:    The path where the result will be saved in .npz format
	'''
	boxes = []
	keypoints = []
	resolution = None
	max_num_person = 0
	# Predict poses:
	for i, img in enumerate(img_generator):
		pose_output = pose_predictor(img)
		max_num_person = max(max_num_person, len(pose_output["instances"].pred_boxes.tensor))
		if len(pose_output["instances"].pred_boxes.tensor) > 0:

			cls_boxes = pose_output["instances"].pred_boxes.tensor.cpu().numpy()

			cls_keyps = pose_output["instances"].pred_keypoints.cpu().numpy()

			# print("cls_keyps:kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
			# print(cls_keyps)

		else:
			#cls_boxes = np.full((4,), np.nan, dtype=np.float32)
			cls_boxes = np.full((1,4,), np.nan, dtype=np.float32)
			#cls_keyps = np.full((17,3), np.nan, dtype=np.float32)   # nan for images that do not contain human
			cls_keyps = np.full((1,17,3), np.nan, dtype=np.float32)
			
		boxes.append(cls_boxes)
		keypoints.append(cls_keyps)

		# Set metadata:
		if resolution is None:
			resolution = {
				'w': img.shape[1],
				'h': img.shape[0],
			}
		# test
		# test_img = Image.fromarray(img, 'RGB')
		# test_img.save('input.jpg')
		# test_img.show()
		print('{}      '.format(i+1), end='\r')

	# Encode data in VidePose3d format and save it as a compressed numpy (.npz):
	data, metadata = encode_for_videpose3d(boxes, keypoints, resolution, dataset_name, max_num_person, traj_output_path)
	output = {}
	output[dataset_name] = {}
	output[dataset_name]['custom'] = [data[0]['keypoints'].astype('float32')]

	output[dataset_name]['custom_boxes'] = [data[0]['bounding_boxes'].astype('float32')]

	np.savez_compressed(output_path, positions_2d=output, metadata=metadata)

	print ('All done!')



if __name__ == '__main__':
	# Todo: adjust all paths
	# Init pose predictor:
	model_config_path = '/home/lin/workspace/detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'

	model_weights_path = '/home/lin/workspace/detectron2/configs/COCO-Keypoints/model_final_5ad38f.pkl'
	pose_predictor = init_pose_predictor(model_config_path, model_weights_path, cuda=True)

	# Predict poses and save the result:
	# img_generator = read_images('./images')    # read images from a directory
	#img_generator = read_video('./video.mp4')  # or get them from a video
	img_generator = read_video('/home/lin/Videos/GP010170.MP4')
	# img_generator = read_video('/home/lin/workspace/OpenTraj/datasets/ETH/seq_eth/video_cut.avi')
	# output_path = './pose2d_cut_testtt_no_interpolation_27_09'
	output_path = '/home/lin/workspace/VideoPose3D-for-multi-person-tracking/Detectron_pose_predictor/npz_output/data_2d_custom_myvideos_GP010170'
	# output_path = '/home/lin/workspace/3DMPPE_ROOTNET_RELEASE/demo/data_mul/data_2d_custom_myvideos_ETH_cut'
	traj_output_path = '/home/lin/workspace/VideoPose3D-for-multi-person-tracking/Detectron_pose_predictor/npz_output/dummy_traj_GP010170'
	# traj_output_path = '/home/lin/workspace/3DMPPE_ROOTNET_RELEASE/demo/data_mul/dummy_traj_ETH_cut'
	predict_pose(pose_predictor, img_generator, output_path, traj_output_path)




