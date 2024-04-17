import argparse
import os
import shutil
import glob
import cv2
import numpy as np

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
	    "--dataset", nargs='?', type=str, default="heritage_data_patrick"
	)
	parser.add_argument(
	    "--scene", nargs='?', type=str, default="hotel_international"
	)
	args, unknown_args = parser.parse_known_args()
	data_folder = args.data
	dataset_name = args.dataset
	scene_name = args.scene
	return scene_name, dataset_name, data_folder


if __name__ == "__main__":
	print("Use as 'python padding.py --dataset heritage_data_patrick --scene hotel_international'")
	scene_name, dataset_name, data_folder = parse_scene_path()

	dataset_path = os.path.join(data_folder,dataset_name)
	scene_path = os.path.join(dataset_path,scene_name)
	dataset_padded_path = os.path.join(data_folder,dataset_name+"_padded")
	scene_padded_path = os.path.join(dataset_padded_path,scene_name)
	os.makedirs(scene_padded_path,exist_ok=True)

	list_image_paths = find_all(scene_path,["jpeg","jpg","png","tif"])
	list_image_names = [os.path.basename(image_path) for image_path in list_image_paths]

	print(list_image_names)

	# Get maximum size matrix (maximum width and maximum height)
	max_width = 0
	max_height = 0
	for idx, image_path in enumerate(list_image_paths):
		cv_image = cv2.imread(image_path)
		height, width, channels = cv_image.shape
		if width > max_width:
			max_width = width
		if height > max_height:
			max_height = height

	# Read images and Add padding
	padding_color = (0,0,0)
	for idx, image_path in enumerate(list_image_paths):
		padded_image_path = os.path.join(scene_padded_path,list_image_names[idx])
		print("padding "+padded_image_path)
		cv_image = cv2.imread(image_path)
		height, width, channels = cv_image.shape
		padded_image = np.full((max_height,max_width, channels), padding_color, dtype=np.uint8) # prepare result
		x_center = (max_width - width) // 2
		y_center = (max_height - height) // 2
		padded_image[y_center:y_center+height,x_center:x_center+width] = cv_image

		cv2.imwrite(padded_image_path,padded_image)
