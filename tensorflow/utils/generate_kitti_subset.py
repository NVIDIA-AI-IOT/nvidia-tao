import argparse
import os
from tqdm import tqdm

"""
Usage:
python generate_kitti_subset.py --source-data-dir=/home/user/data/training --out-data-dir=/home/user/subset_data/training/ --training True --num-images=100
python generate_kitti_subset.py --source-data-dir=/home/user/data/testing --out-data-dir=/home/user/subset_data/testing/ --num-images=100
"""

def main():
    parser = argparse.ArgumentParser(description='Create a subset for kitti')
    parser.add_argument("--source-data-dir", type=str)
    parser.add_argument("--out-data-dir", type=str)
    parser.add_argument("--training", type=bool, default=False)
    parser.add_argument("--num-images", type=int)
    args = parser.parse_args()
    
    source_data_dir = args.source_data_dir
    out_data_dir = args.out_data_dir
    training_flag = bool(args.training)
    print(training_flag)
    num_images = args.num_images

    # source_data_dir should contain the folders image_2, label_2
    if not os.path.exists(os.path.join(source_data_dir,"image_2")):
        print("Download and extract kitti image_2")
        exit()
    if training_flag:
        if not os.path.exists(os.path.join(source_data_dir,"label_2")):
            print("Download and extract kitti label_2")
            exit()

    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    if not os.path.exists(os.path.join(out_data_dir,"image_2")):
        os.makedirs(os.path.join(out_data_dir,"image_2"))
    if training_flag:
      if not os.path.exists(os.path.join(out_data_dir,"label_2")):
          os.makedirs(os.path.join(out_data_dir,"label_2"))

    all_ids = os.listdir(os.path.join(source_data_dir,"image_2"))

    selected_ids = all_ids[:num_images]
    selected_ids = [id.replace('.png', '') for id in selected_ids]

    for id in tqdm(selected_ids):
        image = source_data_dir + "/image_2/" + str(id) + ".png"
        os.system("cp " + image + " " + out_data_dir + "/image_2/")
        if training_flag:
          label = source_data_dir + "/label_2/" + str(id) + ".txt"
          os.system("cp " + label + " " + out_data_dir + "/label_2/")

if __name__ == "__main__":
    main()
