import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

import subprocess as sp
import pickle

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox
from dataset import generate_patch_image


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

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    # parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--gpu', default='0', type=str, dest='gpu_ids')
    # parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--test_epoch', default=19, type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def load_model():
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    # snapshot load
    model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_pose_net(cfg, False)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()
    return model



def import_data(data_path):
    print('Loading 2D detections...')
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    # joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
    keypoints = keypoints['detectron2']['custom_boxes'][0]
    original_img_height, original_img_width = keypoints_metadata['video_metadata']['detectron2']['h'], keypoints_metadata['video_metadata']['detectron2']['w']
    # for i in range(len(keypoints)):
    #     bbox_list = keypoints[i]
    #     print('done')
    return keypoints, original_img_height, original_img_width
def preprocessing_bbox(keypoints):
    keypoints[:, :, 2] = keypoints[:, :, 2] - keypoints[:, :, 0]
    keypoints[:, :, 3] = keypoints[:, :, 3] - keypoints[:, :, 1]
    return keypoints
    # for i in range(len(keypoints)):
    #     bbox_list = keypoints[i]
    #     for i in bbox_list:
    #         i[2] = i[2] - i[0]
    #         i[3] = i[3] - i[1]
    #     bbox_list = bbox_list


def rootpoint_prediction(keypoints, transform, model, img_generator, output_path):
    keyframes_root_3d_test = []
    for i, original_img in enumerate(img_generator):
        bbox_list = keypoints[i]
        bbox_list = bbox_list[~np.isnan(bbox_list).any(axis=1), :]
        person_num = len(bbox_list)

        root_3d_test = []
        # for cropped and resized human image, forward it to RootNet
        if person_num < 1:
            root_3d_test.append(np.nan)
            print('Root joint depth: ' + str(np.nan) + ' mm')
        else:
            for n in range(person_num):
                bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
                img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0)# Todo: get keyframes
                img = transform(img).cuda()[None, :, :, :]
                k_value = np.array(
                    [math.sqrt(cfg.bbox_real[0] * cfg.bbox_real[1] * focal[0] * focal[1] / (bbox[2] * bbox[3]))]).astype(
                    np.float32)
                k_value = torch.FloatTensor([k_value]).cuda()[None, :]
                # forward
                with torch.no_grad():
                    root_3d = model(img, k_value)  # x,y: pixel, z: root-relative depth (mm)
                img = img[0].cpu().numpy()
                root_3d = root_3d[0].cpu().numpy()

                # save output in 2D space (x,y: pixel)
                # vis_img = img.copy()
                # vis_img = vis_img * np.array(cfg.pixel_std).reshape(3, 1, 1) + np.array(cfg.pixel_mean).reshape(3, 1, 1)
                # vis_img = vis_img.astype(np.uint8)
                # vis_img = vis_img[::-1, :, :]
                # vis_img = np.transpose(vis_img, (1, 2, 0)).copy()
                # vis_root = np.zeros((2))
                # vis_root[0] = root_3d[0] / cfg.output_shape[1] * cfg.input_shape[1]
                # vis_root[1] = root_3d[1] / cfg.output_shape[0] * cfg.input_shape[0]
                # cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5, color=(0, 255, 0), thickness=-1,
                #            lineType=cv2.LINE_AA)
                # cv2.imwrite('output_root_2d_' + str(n) + '.jpg', vis_img)

                root_3d_test.append(root_3d[2])
                print('Root joint depth: ' + str(root_3d[2]) + ' mm')
        keyframes_root_3d_test.append(root_3d_test)


    with open(output_path, 'wb') as fp:
        pickle.dump(keyframes_root_3d_test, fp)

if __name__ == '__main__':
    # Todo: adjust the path
    data_path = 'data_mul/data_2d_custom_myvideos_GP010170.npz'
    output_path = 'data_mul/all_keyframes_root_3d_GP010170'

    keypoints, original_img_height, original_img_width = import_data(data_path)
    keypoints = preprocessing_bbox(keypoints)
    # normalized camera intrinsics
    # focal = [1500, 1500]  # x-axis, y-axis
    # Todo: adjust focal lengths fx, fy of GH013110.mp4 = 2.422557950067380e+03, 2.453137175698948e+03
    focal = [2.12902613e+03, 2.10204718e+03]
    princpt = [original_img_width / 2, original_img_height / 2]  # x-axis, y-axis
    print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
    print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

    model = load_model()
    # prepare input image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    # img_path = 'input.jpg'
    # original_img = cv2.imread(img_path)
    # original_img_height, original_img_width = original_img.shape[:2]
    # Todo: adjust the path
    img_generator = read_video('/home/lin/Videos/GP010170.MP4')
    # img_generator = read_video('/home/lin/workspace/OpenTraj/datasets/ETH/seq_eth/video_cut.avi')
    rootpoint_prediction(keypoints, transform, model, img_generator, output_path)


    print('done')