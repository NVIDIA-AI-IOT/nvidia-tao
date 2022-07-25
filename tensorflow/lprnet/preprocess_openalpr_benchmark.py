# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Script to prepare train/val dataset for LPRNet tutorial."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2


def parse_args(args=None):
    """parse the arguments."""
    parser = argparse.ArgumentParser(description='Prepare train/val dataset for LPRNet tutorial')

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory to OpenALPR's benchmark end2end us license plates."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Ouput directory to TLT train/eval dataset."
    )

    return parser.parse_args(args)


def prepare_data(input_dir, img_list, output_dir):
    """Crop the license plates from the orginal images."""

    target_img_path = os.path.join(output_dir, "image")
    target_label_path = os.path.join(output_dir, "label")

    if not os.path.exists(target_img_path):
        os.makedirs(target_img_path)

    if not os.path.exists(target_label_path):
        os.makedirs(target_label_path)

    for img_name in img_list:
        img_path = os.path.join(input_dir, img_name)
        label_path = os.path.join(input_dir,
                                  img_name.split(".")[0] + ".txt")

        img = cv2.imread(img_path)
        with open(label_path, "r") as f:
            label_lines = f.readlines()
            assert len(label_lines) == 1
            label_items = label_lines[0].split()

        assert img_name == label_items[0]
        xmin = int(label_items[1])
        ymin = int(label_items[2])
        width = int(label_items[3])
        xmax = xmin + width
        height = int(label_items[4])
        ymax = ymin + height
        lp = label_items[5]

        cropped_lp = img[ymin:ymax, xmin:xmax, :]

        # save img and label
        cv2.imwrite(os.path.join(target_img_path, img_name), cropped_lp)
        with open(os.path.join(target_label_path,
                               img_name.split(".")[0] + ".txt"), "w") as f:
            f.write(lp)


def main(args=None):
    """Main function for data preparation."""

    args = parse_args(args)

    img_files = []
    for file_name in os.listdir(args.input_dir):
        if file_name.split(".")[-1] == "jpg":
            img_files.append(file_name)

    total_cnt = len(img_files)
    train_cnt = int(total_cnt/2)
    val_cnt = total_cnt - train_cnt
    train_img_list = img_files[0:train_cnt]
    val_img_list = img_files[train_cnt + 1:]
    print("Total {} samples in benchmark dataset".format(total_cnt))
    print("{} for train and {} for val".format(train_cnt, val_cnt))

    train_dir = os.path.join(args.output_dir, "train")
    prepare_data(args.input_dir, train_img_list, train_dir)

    val_dir = os.path.join(args.output_dir, "val")
    prepare_data(args.input_dir, val_img_list, val_dir)


if __name__ == "__main__":
    main()
