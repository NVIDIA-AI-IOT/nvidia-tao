import argparse
import os
import json
from tqdm import tqdm

"""
Usage:
python generate_coco_subset.py --source-image-dir=/home/user/data/train2017 --source-annotation-file=/home/user/data/annotations/instances_train2017.json --out-data-dir=/home/user/subset_data/ --num-images=100
python generate_coco_subset.py --source-image-dir=/home/user/data/val2017 --source-annotation-file=/home/user/data/annotations/instances_val2017.json --out-data-dir=/home/user/subset_data/ --num-images=100
"""


def main():
    parser = argparse.ArgumentParser(description='Create a subset for coco')
    parser.add_argument("--source-image-dir", type=str)
    parser.add_argument("--source-annotation-file", type=str)
    parser.add_argument("--out-data-dir", type=str)
    parser.add_argument("--num-images", type=int)
    args = parser.parse_args()
    
    source_image_dir = args.source_image_dir
    source_annotation_file = args.source_annotation_file
    out_data_dir = args.out_data_dir
    num_images = args.num_images

    if not os.path.exists(source_image_dir):
        raise Exception("Download and extract coco train2017/val2017/test2017")

    if source_annotation_file:
        if not os.path.exists(os.path.join(source_annotation_file)):
            raise Exception("Download and extract coco annotations.zip")

    if source_annotation_file:
        out_image_dir = os.path.join(out_data_dir,source_image_dir.split("/")[-1])
        os.makedirs(out_image_dir, exist_ok=True)
        out_json_dir = os.path.join(out_data_dir,"annotations")
        os.makedirs(out_json_dir, exist_ok=True)

        out_json_path = os.path.join(out_data_dir,"annotations",source_annotation_file.split("/")[-1])
        inp_json_dict = json.load(open(source_annotation_file))
        print("Loaded inp annotations")

        out_json_dict = {}

        out_json_dict["images"] = []
        out_json_dict["annotations"] = []

        id_set = set()

        for idx in tqdm(range(num_images)):
            if(idx == len(inp_json_dict["images"])):
                break
            img_info = inp_json_dict["images"][idx]
            out_json_dict["images"].append(img_info)
            src_file_name = os.path.join(source_image_dir, img_info["file_name"])
            os.system("cp " + src_file_name + " " + out_image_dir)
            id_set.add(img_info["id"])

        debug_set = set()
        for annot_info in tqdm(inp_json_dict["annotations"]):
            if(annot_info["image_id"] in id_set):
                annot_info["area"] = float(annot_info["bbox"][2]*annot_info["bbox"][3])
                if(annot_info["bbox"][0] <= 0.0):
                    annot_info["bbox"][0] = 0.0
                if(annot_info["bbox"][1] <= 0.0):
                    annot_info["bbox"][1] = 0.0
                if(annot_info["area"] > 0.0):
                    out_json_dict["annotations"].append(annot_info)

        out_json_dict["categories"] = inp_json_dict["categories"]

        json_out_str = json.dumps(out_json_dict, indent = 4)
        with open(out_json_path, "w") as json_out_file:
            json_out_file.write(json_out_str)

if __name__ == "__main__":
    main()
