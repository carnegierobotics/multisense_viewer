#!/usr/bin/env python3

"""
images_to_ros1_bagfile.py

A script which converts a directory of images logged by the MultiSense Viewer into a ROS1 bagfile
"""

import argparse
import csv
import cv2
import json
import numpy as np
import os
import sys

from cv_bridge import CvBridge
import rosbag
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

def write_bag_file(bag_path, image_paths, left_camera_info, right_camera_info, aux_camera_info):
    bag = rosbag.Bag(bag_path, 'w')

    bridge = CvBridge()

    for topic, path, timestamps in image_paths:

        frame = "left_optical"
        camera_info_base = left_camera_info
        if 'right' in topic:
            camera_info_base = right_camera_info
            frame = "right_optical"
        elif 'aux' in topic:
            camera_info_base = aux_camera_info
            frame = "aux_optical"


        with open(timestamps, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for name, time in reader:
                image_path = os.path.join(path, name)
                if 'aux' in topic:
                    image = cv2.imread(image_path)
                else:
                    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

                sec, microsec = time.split('.')
                microsec = microsec.zfill(6)

                ros_time = rospy.Time.from_sec(float(sec + '.' + microsec))

                header = Header()
                header.frame_id = frame
                header.stamp = ros_time

                ros_image = bridge.cv2_to_imgmsg(image, "passthrough")
                ros_image.header = header

                camera_info_base.header = header
                camera_info_base.width = ros_image.width
                camera_info_base.height = ros_image.height

                bag.write(topic, ros_image, ros_time)
                bag.write(topic + '/camera_info', camera_info_base, ros_time)

    bag.close()


def get_camera_info(root_images_directory):
    left_camera_info = CameraInfo()
    right_camera_info = CameraInfo()
    aux_camera_info = CameraInfo()
    for path in [os.path.join(root_images_directory, p) for p in os.listdir(root_images_directory)]:
        if path.endswith("intrinsics.yml"):
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            left_camera_info.K = fs.getNode("M1").mat().flatten().tolist()
            left_camera_info.D = fs.getNode("D1").mat().flatten().tolist()

            right_camera_info.K = fs.getNode("M2").mat().flatten().tolist()
            right_camera_info.D = fs.getNode("D2").mat().flatten().tolist()

            try:
                aux_camera_info.K = fs.getNode("M3").mat().flatten().tolist()
                aux_camera_info.D = fs.getNode("D3").mat().flatten().tolist()
            except cv2.error:
                print("no aux intrinsic information")

        if path.endswith("extrinsics.yml"):
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            left_camera_info.P = fs.getNode("P1").mat().flatten().tolist()
            left_camera_info.R = fs.getNode("R1").mat().flatten().tolist()

            right_camera_info.P = fs.getNode("P2").mat().flatten().tolist()
            right_camera_info.R = fs.getNode("R2").mat().flatten().tolist()

            try:
                aux_camera_info.P = fs.getNode("P3").mat().flatten().tolist()
                aux_camera_info.R = fs.getNode("R3").mat().flatten().tolist()
            except cv2.error:
                print("no aux extrinsic information")

    left_camera_info.distortion_model = 'rational_polynomial' if len(left_camera_info.D) == 8 else 'plumb_bob'
    right_camera_info.distortion_model = 'rational_polynomial' if len(right_camera_info.D) == 8 else 'plumb_bob'
    aux_camera_info.distortion_model = 'rational_polynomial' if len(aux_camera_info.D) == 8 else 'plumb_bob'

    return left_camera_info, right_camera_info, aux_camera_info

def parse_image_directories(root_images_directory, topic_prefix):
    paths = []
    for root, dirs, files in os.walk(root_images_directory):
        image_path = None
        timestamps_path = None
        topic = None
        for d in dirs:
            if d == "png" or d == "tiff" or d == "ppm":
                image_path = os.path.join(root, d)
                topic = topic_prefix + "/" + os.path.basename(root).lower()
                break
        for f in files:
            if f.endswith("timestamps.csv"):
                timestamps_path = os.path.join(root, f)
                break

        if image_path is not None and timestamps_path is not None and topic is not None:
            paths.append((topic, image_path, timestamps_path))

    return paths

def parse_arguments():
   parser = argparse.ArgumentParser(description='Convert logged images to a ROS1 bagfile.')
   parser.add_argument('-r', '--root-images-directory', type=str, required=True, help='Root directory created by the viewer logger')
   parser.add_argument('-p', '--topic-prefix', type=str, default="/multisense", help='The string prefix to apply to all topics')
   parser.add_argument('-o', '--output-bag-file', type=str, default='output.bag', help='Output bag file.')

   args = parser.parse_args()

   return args.root_images_directory, args.output_bag_file, args.topic_prefix

def main(root_images_directory, output_bag_file, topic_prefix):
    metadata_path = os.path.join(root_images_directory, 'metadata.json')

    if not os.path.exists(metadata_path):
        raise RuntimeError('Invalid root directory')

    image_paths = parse_image_directories(root_images_directory, topic_prefix)
    left_camera_info, right_camera_info, aux_camera_info = get_camera_info(root_images_directory)

    write_bag_file(output_bag_file, image_paths, left_camera_info, right_camera_info, aux_camera_info)

if __name__ == "__main__":
   root_images_directory, output_bag_file, topic_prefix = parse_arguments()
   main(root_images_directory, output_bag_file, topic_prefix)

