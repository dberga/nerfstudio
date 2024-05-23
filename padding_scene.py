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
	parser.add_argument('--use_masks', action='store_true')
	parser.add_argument('--force_squared', action='store_true')
	args, unknown_args = parser.parse_known_args()
	data_folder = args.data
	dataset_name = args.dataset
	scene_name = args.scene
	use_masks = args.use_masks
	force_squared = args.force_squared
	return scene_name, dataset_name, data_folder, use_masks, force_squared


if __name__ == "__main__":
	print("Use as 'python padding_scene.py --dataset heritage_data_patrick --scene hotel_international'")
	scene_name, dataset_name, data_folder,use_masks, force_squared = parse_scene_path()

	# prepare folders
	dataset_path = os.path.join(data_folder,dataset_name)
	scene_path = os.path.join(dataset_path,scene_name)
	dataset_padded_path = os.path.join(data_folder,dataset_name+"_padded")
	scene_padded_path = os.path.join(dataset_padded_path,scene_name)
	os.makedirs(scene_padded_path,exist_ok=True)
	
	# find images
	list_image_paths = find_all(scene_path,["jpeg","jpg","png","tif"])
	list_image_names = [os.path.basename(image_path) for image_path in list_image_paths]

	if use_masks:
		# prepare folders
		scene_masks_path = os.path.join(dataset_path,scene_name,"masks")
		scene_padded_masks_path = os.path.join(dataset_padded_path,scene_name,"masks")	
		os.makedirs(scene_padded_masks_path,exist_ok=True)
		# find masks
		list_masks_paths = find_all(scene_masks_path,["jpeg","jpg","png","tif"])
		list_masks_names = [os.path.basename(mask_path) for mask_path in list_masks_paths]

	# intersect masks and images to make sure there is the same list
	if use_masks: 
		list_image_names_noext = [os.path.splitext(os.path.basename(name))[0] for name in list_image_names]
		list_masks_names_noext = [os.path.splitext(os.path.basename(name))[0] for name in list_masks_names]
		list_images_ext = [os.path.splitext(os.path.basename(name))[1] for name in list_image_names]
		list_masks_ext = [os.path.splitext(os.path.basename(name))[1] for name in list_masks_names]
		image_extension = list_images_ext[0]
		mask_extension = list_masks_ext[0]

		list_intersect_names_noext = list(set(list_image_names_noext).intersection(set(list_masks_names_noext)))
		list_intersect_images_names = [name+image_extension for name in list_intersect_names_noext]
		list_intersect_images_paths = [os.path.join(dataset_path,scene_name,name+image_extension) for name in list_intersect_names_noext]
		list_intersect_images_masks_names = [name+mask_extension for name in list_intersect_names_noext]
		list_intersect_images_masks_paths = [os.path.join(dataset_padded_path,scene_name,name+mask_extension) for name in list_intersect_names_noext]

		list_image_names = list_intersect_images_names
		list_image_paths = list_intersect_images_paths
		list_masks_names = list_intersect_images_masks_names
		list_masks_paths = list_intersect_images_masks_paths

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

	if force_squared and (max_width != max_height):
		max_width=np.max([max_width,max_height])
		max_height=np.max([max_width,max_height])

	# Read images and Add padding
	padding_color = (0,0,0) # black
	mask_key = np.uint8(255) # white
	for idx, image_path in enumerate(list_image_paths):
		# read img
		cv_image = cv2.imread(image_path)
		height, width, channels = cv_image.shape
		x_center = (max_width - width) // 2
		y_center = (max_height - height) // 2
		
		# create padded image
		padded_image_path = os.path.join(scene_padded_path,list_image_names[idx])
		print("padding "+padded_image_path)
		padded_image = np.full((max_height,max_width, channels), padding_color, dtype=np.uint8) # prepare result
		padded_image[y_center:y_center+height,x_center:x_center+width] = cv_image
		cv2.imwrite(padded_image_path,padded_image)

		if use_masks:
			# create mask
			if os.path.exists(scene_masks_path): # masks already exist in original scene, adapt padded masks
				mask_image_path = os.path.join(scene_padded_masks_path,list_masks_names[idx])
				print("creating mask "+mask_image_path)
				input_mask_path = os.path.join(scene_masks_path,list_masks_names[idx])
				cv_mask = cv2.imread(input_mask_path)
				mask_image = np.full((max_height,max_width, channels), padding_color, dtype=np.uint8) # prepare result
				mask_image[y_center:y_center+height,x_center:x_center+width] = cv_mask
				cv2.imwrite(mask_image_path,mask_image)
			else: # masks do not exist in original scene, make only padded mask
				mask_image_path = os.path.join(scene_padded_masks_path,list_masks_names[idx])
				print("creating mask "+mask_image_path)
				mask_image = np.full((max_height,max_width, channels), padding_color, dtype=np.uint8) # prepare result
				mask_image[y_center:y_center+height,x_center:x_center+width] = mask_key
				cv2.imwrite(mask_image_path,mask_image)