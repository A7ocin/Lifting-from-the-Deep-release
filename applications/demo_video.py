#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Nicola Garau'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath

import json

from absl import app
from absl import flags

FLAGS = flags.FLAGS

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

flags.DEFINE_string('VIDEO_FILE_PATH', PROJECT_PATH + '/data/videos/skate.mp4', 'VIDEO_FILE_PATH')
flags.DEFINE_string('JSON_OUTPUT', PROJECT_PATH + '/data/json/out.json', 'JSON_OUTPUT_PATH')
flags.DEFINE_bool('DISPLAY', False, 'Display results')


def main(argv):
    cap = cv2.VideoCapture(FLAGS.VIDEO_FILE_PATH)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image_size = (height, width, 3)
    ret, image = cap.read()
    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)
    pose_estimator.initialise()
    
    frame_num = 0   
    poses = {} 
    with open(FLAGS.JSON_OUTPUT, 'w') as outfile:
        try:
            while(True):
                ret, image = cap.read()
                if(ret==False):
                    print("End of video")
                    break
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

                try:
                    # estimation
                    pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

                    print(str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)*100/cap.get(cv2.CAP_PROP_FRAME_COUNT))) + "%")
                    poses[str(frame_num)] = pose_3d.tolist()
                    #print(poses[str(frame_num)])

                    if(FLAGS.DISPLAY):
                        # Show 2D and 3D poses
                        display_results(image, pose_2d, visibility, pose_3d)
                except ValueError:
                    print('No visible people in the image. Change CENTER_TR in packages/lifting/utils/config.py ...')
                frame_num+=1
        except KeyboardInterrupt:
            pass 
        json.dump(poses, outfile)

    # close model
    pose_estimator.close()


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show(block=False)

if __name__ == '__main__':
    import sys
    app.run(main)
