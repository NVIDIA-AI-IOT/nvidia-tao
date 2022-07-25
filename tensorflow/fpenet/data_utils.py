# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
"""FPENet data conversion utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import numpy as np
import json


def get_keypoints_from_file(keypoints_file):
    '''
    This function reads the keypoints file from afw format.

    Input:
        keypoints_file (str): Path to the keypoints file.
    Output:
        keypoints (np.array): Keypoints in numpy format [[x, y], [x, y]].
    '''
    keypoints = []
    with open(keypoints_file) as fid:
        for line in fid:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                keypoints.append([float(loc_x), float(loc_y)])
    keypoints = np.array(keypoints, dtype=np.float)
    assert keypoints.shape[1] == 2, "Keypoints should be 2D."
    return keypoints


def convert_dataset(afw_data_path, output_json_path, afw_image_save_path):
    '''
    Function to convert afw dataset to Sloth format json.

    Input:
        afw_data_path (str): Path to afw data folder.
        output_json_path (str): Path to output json file.
        afw_image_save_path (str): Image paths to use in json.
    Returns:
        None 
    '''
    # get dataset file lists
    all_files = os.listdir(afw_data_path)
    images = [x for x in all_files if x.endswith('.jpg')]
    keypoint_files = [img_path.split(".")[-2] + ".pts" for img_path in images]

    output_folder = os.path.dirname(output_json_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read and convert to sloth format
    sloth_data = []

    for image in images:
        image_path = os.path.join(afw_data_path, image)
        image_read = cv2.imread(image_path)
        if image_read is None:
            print('Bad image:{}'.format(image_path))
            continue
        # convert image to png
        image_png = image.replace('.jpg', '.png')
        cv2.imwrite(os.path.join(afw_data_path, image_png), image_read)        
        image_data = {}
        image_data['filename'] = os.path.join(afw_image_save_path, image_png)
        image_data['class'] = 'image'

        annotations = {}
        annotations['tool-version'] = '1.0'
        annotations['version'] = 'v1'
        annotations['class'] = 'FiducialPoints'

        keypoint_file = image.split(".")[-2] + ".pts"
        image_keypoints = get_keypoints_from_file(os.path.join(afw_data_path, keypoint_file))
        
        for num, keypoint in enumerate(image_keypoints):
            annotations["P{}x".format(num+1)] = keypoint[0]
            annotations["P{}y".format(num+1)] = keypoint[1]

        # fill in dummy keypoints for keypoints 69 to 80
        for num in range(69, 81, 1):
            annotations["P{}x".format(num)] = image_keypoints[0][0]
            annotations["P{}y".format(num)] = image_keypoints[0][1]
            annotations["P{}occluded".format(num)] = True

        image_data['annotations'] = [annotations]
        sloth_data.append(image_data)

    # save json
    with open(output_json_path, "w") as config_file:
        json.dump(sloth_data, config_file, indent=4)


def parse_args(args=None):
    """parse the arguments."""
    parser = argparse.ArgumentParser(
        description='Transform dataset for FPENet tutorial')

    parser.add_argument(
        "--afw_data_path",
        type=str,
        required=True,
        help="Input directory to AFW dataset imnages and ground truth keypoints."
    )

    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="Output json file path to save to."
    )

    parser.add_argument(
        "--afw_image_save_path",
        type=str,
        required=True,
        help="Image path to use in jsons."
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(args.afw_data_path, args.output_json_path, args.afw_image_save_path)
