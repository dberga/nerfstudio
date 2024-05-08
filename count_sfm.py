import argparse
import os
import json
import glob
import sys

def find_all(path, formats=["jpeg","jpg","png","tif"]):
    # get all images files
    result = []
    #for root, dirs, files in os.walk(path):
    #    if name in files:
    #        result.append(os.path.join(root, name))
    for fmt in formats:
        glob_token="*."+fmt
        glob_path=os.path.join(path,glob_token)
        for filename in glob.glob(glob_path):
            result.append(filename)
    return result

def parse_scene_path():
    # get folder
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", nargs='?', type=str, default="data"
    )
    parser.add_argument(
        "--dataset", nargs='?', type=str, default="nerfstudio"
    )
    parser.add_argument(
        "--scene", nargs='?', type=str, default="desolation"
    )
    parser.add_argument(
        "--transforms", nargs='?', type=str, default="transforms.json"
    )
    args, unknown_args = parser.parse_known_args()
    data_folder = args.data
    dataset_name = args.dataset
    scene_name = args.scene
    transforms_name = args.transforms
    return scene_name, dataset_name, data_folder, transforms_name


if __name__ == "__main__":
    #print("Use as 'python count_sfm.py --dataset heritage_data_patrick --scene hotel_international'")
    scene_name, dataset_name, data_folder, transforms_name = parse_scene_path()

    dataset_path = os.path.join(data_folder,dataset_name)
    scene_path = os.path.join(dataset_path,scene_name)
    transforms_path = os.path.join(dataset_path,scene_name,transforms_name)

    if os.path.exists(transforms_path):
        # get number of available images
        list_image_paths = find_all(scene_path,["jpeg","jpg","png","tif"])
        if len(list_image_paths) == 0:
            scene_path = os.path.join(scene_path,"images") # for sip (skip image processing) case
            list_image_paths = find_all(scene_path,["jpeg","jpg","png","tif"])
        list_image_names = [os.path.basename(image_path) for image_path in list_image_paths]
        num_images = len(list_image_paths)
        
        # get number of images matched in transforms.json
        with open(transforms_path) as f:
            data = json.load(f)
            num_matches = len(data['frames'])

        match_percentage = int((num_matches/num_images)*100)

        print(f"({dataset_name}/{scene_name}) Total images {num_images}, Found matches {num_matches} -> [{match_percentage}%]")
        sys.exit(match_percentage)
    else:
        print(f"ERROR!!! {transforms_path} does not exist")