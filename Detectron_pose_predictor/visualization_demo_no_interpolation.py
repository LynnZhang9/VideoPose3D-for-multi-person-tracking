import os
import shutil
import numpy as np
import cv2
import subprocess as sp

import pandas as pd

def load_keypoints_boxes_from_npz(path_to_npz, dataset_name='detectron2'):
	data = np.load(path_to_npz, encoding='latin1', allow_pickle=True)
	meta = data['metadata'].item()
	keypoints = data['positions_2d'].item()[dataset_name]['custom_keypoints'][0]
	boxes = data['positions_2d'].item()[dataset_name]['custom_boxes'][0]

	return keypoints, boxes, meta


def remove_dir(dir):
	try:
		shutil.rmtree(dir)
	except OSError as e:
		print ("Error: %s - %s." % (e.filename, e.strerror))


def frames_to_video(src_path, dst_path, fps=30):
	os.system("ffmpeg -framerate %s -pattern_type glob -f image2 -i '%s/*.jpeg' %s" % (fps, src_path, dst_path))


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
		yield cv2.imread(path), path


def read_video(filename):
	w, h = get_resolution(filename)

	command = ['ffmpeg',
			'-i', filename,
			'-f', 'image2pipe',
			'-pix_fmt', 'bgr24',
			'-vsync', '0',
			'-vcodec', 'rawvideo', '-']

	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
	i = 0
	while True:
		i += 1
		data = pipe.stdout.read(w*h*3)
		if not data:
			break
		yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3)), str(i-1).zfill(5)


def get_resolution(filename):
	command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
			   '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
	for line in pipe.stdout:
		w, h = line.decode().strip().split(',')
		return int(w), int(h)


def draw_body_joints_2d(img_orig, pts2d, bones=None, draw_indices=None):
	img = img_orig.copy()
	for m in range(len(pts2d)):
		for i in range(len(pts2d[m])):
			if np.isnan(pts2d[m][i][0]) or np.isnan(pts2d[m][i][1]):
				
				continue
			

			x = int(round(pts2d[m][i][0]))
			y = int(round(pts2d[m][i][1]))
			img = cv2.circle(img, (x,y), 5, (0,0,255), 5)

			if draw_indices is not None:
					font = cv2.FONT_HERSHEY_SIMPLEX
					bottomLeftCornerOfText = (x,y)
					fontScale = 1
					fontColor = (255,255,255)
					lineType = 2
					cv2.putText(img,str(i), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

			if bones is not None:
				for bone in bones:
					pt1 = (int(round(pts2d[m][bone[0]][0])), int(round(pts2d[m][bone[0]][1])))
					pt2 = (int(round(pts2d[m][bone[1]][0])), int(round(pts2d[m][bone[1]][1])))
					img = cv2.line(img,pt1,pt2,(255,0,0),4)

	return img

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=range(24), offset=(0,0)):
	for i,box in enumerate(bbox):
		if np.isnan(box[0]):
			continue
		print('i={}'.format(i))
		print('len(box)')
		print(len(box))
		print(box)
		x1,y1,x2,y2 = [int(i) for i in box]
		print(x1,y1,x2,y2)
		x1 += offset[0]
		x2 += offset[0]
		y1 += offset[1]
		y2 += offset[1]
		# box text and bar
		# id = int(identities[i]) if identities is not None else 0  
		id = i
		print('id')
		print(id)
		color = compute_color_for_labels(id)
		label = '{}{:d}'.format("", id)
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
		cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
		cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
		cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
	return img

def draw_trajectories(img, trajectories, offset=(0,0)):
	# for i,trajectory in enumerate(trajectories):
		# if np.isnan(trajectory[0]):
		# 	continue
	for i in range(int(len(trajectories)/3)):
		if np.isnan(trajectories["traj" + str(i) + "_pid"]):
			continue
		
		print('pid={}'.format(trajectories["traj" + str(i) + "_pid"]))
		print('x={}'.format(trajectories["traj" + str(i) + "_x"]))
		print('y={}'.format(trajectories["traj" + str(i) + "_y"]))
		# x1,y1,x2,y2 = [int(i) for i in box]
		# print(x1,y1,x2,y2)
		x1 = trajectories["traj" + str(i) + "_x"]
		y1 = trajectories["traj" + str(i) + "_y"]
		pid = trajectories["traj" + str(i) + "_pid"]

		x1 += offset[0]
		x = int(round(x1))

		y1 += offset[0]
		y = int(round(y1))

		pid = int(pid)
		
		# box text and bar
		# id = int(identities[i]) if identities is not None else 0  
		# id = i
		# print('id')
		# print(id)
		color = compute_color_for_labels(pid)
		label = '{}{:d}'.format("", pid)
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
		# cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
		img = cv2.circle(img, (x,y), 5, (0,0,255), 5)
		img = cv2.rectangle(img,(x, y),(x+t_size[0]+3,y+t_size[1]+4), color,-1)
		img = cv2.putText(img,label,(x,y+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
	return img

def visualize_keypoints_boxes(img_generator, keypoints, boxes, trajectories, mp4_output_path, fps=30, draw_joint_indices=None):
	'''
	Visualize keypoints (2d body joints) detected by Detectron2:
		img_generator:      Images source (images or video)
		keypoints:          Body keypoints detected by Detectron2
		mp4_output_path:    The path where the result will be saved in .mp4 format
		fps:                FPS of the result video
		draw_joint_indices: Draw body joint indices (in COCO format)
	'''
	body_edges_17 = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],
							  [11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])

	#Create a temp_dir to save intermediate results:
	temp_dir = './temp'
	if os.path.exists(temp_dir):
		remove_dir(temp_dir)
	os.makedirs(temp_dir)

	#Draw keypoints and save the result:
	for i, (img, img_path) in enumerate(img_generator):
		print('frame={}'.format(i))
		frame_joint2d = keypoints[i]
		frame_boxes = boxes[i]
		try:
			frame_trajectories = trajectories.loc[i+1]
		except Exception:
			print("Error: No corresponding frames")
			img = draw_body_joints_2d(img, frame_joint2d, bones=body_edges_17, draw_indices=draw_joint_indices)
			#img = draw_boxes(img, frame_boxes, identities=None, offset=(0,0))
		else:
			img = draw_body_joints_2d(img, frame_joint2d, bones=body_edges_17, draw_indices=draw_joint_indices)
			#img = draw_boxes(img, frame_boxes, identities=None, offset=(0,0))
			img = draw_trajectories(img, frame_trajectories, offset=(0,0))

		img_name = img_path.split('/')[-1].split('.')[0]
		out_path = os.path.join(temp_dir,img_name + '.jpeg')
		cv2.imwrite(out_path,img)

		print('{}      '.format(i+1), end='\r')

	#Convert images to video:
	frames_to_video(temp_dir, mp4_output_path, fps=fps)
	remove_dir(temp_dir)

	print ('All done!')


if __name__ == '__main__':
	#Load keypoints from .npz:
	keypoints_npz_path = './pose2d_cut_testtt_no_interpolation_06_09.npz'
	keypoints,boxes,_ = load_keypoints_boxes_from_npz(keypoints_npz_path, dataset_name='detectron2')

	# img_generator = read_images('./images')    # read images from a directory
	img_generator = read_video('/home/lin/Videos/GH013110_original_cut_cut.MP4')  # or get them from a video

	trajectories = pd.read_hdf("./output/_cut_cut_traj_20_09.h5")
	#Visualize the keypoints:
	# visualize_keypoints(img_generator, keypoints, mp4_output_path='./GH013110_original_output_cut_cut_testtt_no_interpolation_16_08.mp4')
	visualize_keypoints_boxes(img_generator, keypoints, boxes, trajectories, mp4_output_path='./GH013110_original_output_cut_cut_testtt_no_interpolation_20_09.mp4')
