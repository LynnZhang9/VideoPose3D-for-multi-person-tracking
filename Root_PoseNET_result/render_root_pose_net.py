import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
import pickle

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)

def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

def multi_person_render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + 1), size))
    ax_in = fig.add_subplot(1, 1 + 1, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    # for index, (title, data) in enumerate(poses.items()):
    ax = fig.add_subplot(1, 1 + 1, 0 + 2, projection='3d')
    ax.view_init(elev=15., azim=azim)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    ax.set_title("Reconstruction")  # , pad=35
    ax_3d.append(ax)
    lines_3d.append([])
    trajectories.append(poses[:, :, 0, [0, 1]])
    poses = [poses]

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        # keypoints = keypoints[input_video_skip:]  # todo remove
        # for mul_idx in range(len(poses)):
        #     for idx in range(len(poses[mul_idx])):
        #         poses[mul_idx][idx] = poses[mul_idx][idx][input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    # if downsample > 1:
    #     keypoints = downsample_tensor(keypoints, downsample)
    #     all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
    #     # for idx in range(len(poses)):
    #     #     poses[idx] = downsample_tensor(poses[idx], downsample)
    #     #     trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
    #     fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    # parents = skeleton.parents()
    joint_num = skeleton['joint_num']
    joints_name = skeleton['joints_name']
    flip_pairs = skeleton['flip_pairs']
    skeleton = skeleton['skeleton']

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
        #     # if np.isnan(trajectories[n][i, 0]) or np.isnan(trajectories[n][i, 1]):
        #     #     continue
        #     min_x = 0
        #     max_x = 0
        #     min_y = 0
        #     max_y = 0
        #     for mul in range(len(trajectories[n][i])):
        #         min_x = min(min_x, trajectories[n][i][mul, 0])
        #         max_x = max(max_x, trajectories[n][i][mul, 0])
        #         min_y = min(min_y, trajectories[n][i][mul, 1])
        #         max_y = max(max_y, trajectories[n][i][mul, 1])
            ax.set_xlim3d([-15000,5000])
            ax.set_xlabel('X')
            ax.set_ylim3d([0,30000])
            ax.set_ylabel('Y')
            ax.set_zlim3d([-5000,15000])
        #         # add label on coordinate


        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 1)]
        # colors_ = [np.array((c[2], c[1], c[0])) for c in colors]
        all_colors = []
        for col in range(keypoints.shape[1]):
            all_colors += [np.array((c[2], c[1], c[0])) for c in colors]


        keypoints_i_reshape = keypoints[i, :, :, :].reshape(keypoints.shape[1] * keypoints.shape[2], keypoints.shape[3])

        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            #right
            for n, ax in enumerate(ax_3d):
                for mul in range(poses[n].shape[1]):
                # for mul in range(3):
                    for j, j_parent in enumerate(skeleton):
                        # if j_parent == -1:
                        #     continue

                        # if len(parents) == keypoints.shape[2] and keypoints_metadata['layout_name'] != 'coco':
                        if len(skeleton) == (keypoints.shape[2] - 1):

                            lines.append(ax_in.plot([keypoints[i, mul, j_parent[0], 0], keypoints[i, mul, j_parent[1], 0]],
                                                    [keypoints[i, mul, j_parent[0], 1], keypoints[i, mul, j_parent[1], 1]], color='pink'))

                        # col = 'red' if j in skeleton.joints_right() else 'black'
                        # for mul in range(poses[n].shape[0]):
                        #     for n, ax in enumerate(ax_3d):
                                # poses_i_reshape = poses[n][:, i, :, :]
                                # pos = poses_i_reshape.reshape(poses_i_reshape.shape[0] * poses_i_reshape.shape[1], poses_i_reshape.shape[2])
                                # for mul in range(len(poses[n][0])):
                        pos = poses[n][i][mul]
                        lines_3d[n].append(ax.plot([pos[j_parent[0], 0], pos[j_parent[1], 0]],
                                                   [pos[j_parent[0], 2], pos[j_parent[1], 2]],
                                                   [-pos[j_parent[0], 1], -pos[j_parent[1], 1]], zdir='z', c='red'))
                        # lines_3d[0].append(ax.plot([x[i1][n], x[i2][n]],
                        #                            [z[i1][n], z[i2][n]],
                        #                            [-y[i1][n], -y[i2][n]], zdir='z', c=colors[l]))


            points = ax_in.scatter(*keypoints_i_reshape.T, 10, color=all_colors, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])
            for n, ax in enumerate(ax_3d):
                for mul in range(poses[n].shape[1]):
                # for mul in range(3):
                    for j, j_parent in enumerate(skeleton):
                        # if j_parent == -1:
                        #     continue

                        if len(skeleton) == (keypoints.shape[2]-1):

                            lines[20*mul+j][0].set_data([keypoints[i, mul, j_parent[0], 0], keypoints[i, mul, j_parent[1], 0]],
                                                        [keypoints[i, mul, j_parent[0], 1], keypoints[i, mul, j_parent[1], 1]])

                        pos = poses[n][i][mul]
                        lines_3d[n][mul*20 + j][0].set_xdata(np.array([pos[j_parent[0], 0], pos[j_parent[1], 0]]))
                        lines_3d[n][mul*20 + j][0].set_ydata(np.array([pos[j_parent[0], 2], pos[j_parent[1], 2]]))
                        lines_3d[n][mul*20 + j][0].set_3d_properties(np.array([-pos[j_parent[0], 1], -pos[j_parent[1], 1]]), zdir='z')


            points.set_offsets(keypoints_i_reshape)

        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


def list2array(data,num_keypoint=21, num_point_dimenssion=3):
    max_num = 0
    for i, list in enumerate(data):
        max_num = max(max_num, len(list))

    for i, list in enumerate(data):
        print(i)
        print(max_num - len(list))
        num_to_add = max_num - len(list)
        for i in range(num_to_add):
            list.append(np.full([num_keypoint, num_point_dimenssion], np.nan))

    data = np.array(data)

    return data


if __name__ == '__main__':
    FILE = r"./output_pose_12_02_focal_GH013110" # Todo: adjust the input 3d pose data
    with open(FILE, 'rb') as fp:
        data = pickle.load(fp)
    max_num_person = 0
    for num_person_list in data[1]:
        max_num_person = max(max_num_person, len(num_person_list))

    data_2d = data[0]
    data_3d = data[1]
    predicted_multi_input_keypoints = list2array(data_2d, num_point_dimenssion=2)
    anim_output = list2array(data_3d, num_point_dimenssion=3)
    fps = None
    keypoints_metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]], 'video_metadata': {'detectron2': {'w': 2704, 'h': 1520}}}


    # MuCo joint set
    skeleton_info = {
        "joint_num": 21,
        "joints_name": ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe'),
        "flip_pairs": ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20)),
        "skeleton": ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
    }

    viz_bitrate = 3000
    azim = -70
    viz_output = "root_pose_net_output_12_02_focal_GH013110.mp4" # Todo: adjust the output video name
    viz_limit = -1
    viz_downsample = 1
    viz_size = 6
    viz_video = '/home/lin/Videos/GH013110_original_cut_cut.MP4'
    viewport = (2704, 1520)
    viz_input = 0
    multi_person_render_animation(predicted_multi_input_keypoints, keypoints_metadata, anim_output,
                                  skeleton_info, fps, viz_bitrate, azim, viz_output,
                                  limit=viz_limit, downsample=viz_downsample, size=viz_size,
                                  input_video_path=viz_video, viewport=viewport,
                                  input_video_skip=viz_input)
