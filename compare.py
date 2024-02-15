import open3d as o3d
import numpy as np

import argparse
import os
import shutil
import pandas as pd # make sure you also have installed 'tabulate' for markdown

EXPORTS_FOLDER="exports" # exports
MESHES_FOLDER="mesh" # exports/mesh
POINTCLOUDS_FOLDER="pcd" # exports/pcd

def find_all(name, path):
        # get all results files
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def parse_scene_path():
        # get folder
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exports", nargs='?', type=str, default=EXPORTS_FOLDER
        )
        parser.add_argument(
            "--scene", nargs='?', type=str, default="kitchen"
        )

        parser.add_argument(
            "--type", nargs='?', type=str, default="pcd" # pcd, mesh
        )
        parser.add_argument(
            "--visualize", nargs='?', type=str, default=False
        )
        args, unknown_args = parser.parse_known_args()
        exports_folder = args.exports
        scene_name = args.scene
        comparison_type = args.type
        vis_flag = args.visualize

        scene_path_mesh = os.path.join(exports_folder,MESHES_FOLDER,scene_name)
        scene_path_pcd = os.path.join(exports_folder,POINTCLOUDS_FOLDER,scene_name)
        return scene_path_mesh,scene_path_pcd, scene_name, comparison_type, exports_folder, vis_flag

def read_mesh(mesh_path):
    data = o3d.io.read_triangle_mesh(mesh_path,enable_post_processing=True)
    return data

def read_pcd(pcd_path):
    data = o3d.io.read_point_cloud(pcd_path)
    return data

def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(2.5, 0.0)
        return False

if __name__ == "__main__":        
    scene_path_mesh,scene_path_pcd, scene_name, comparison_type, exports_folder, vis_flag = parse_scene_path()
    df_mesh = pd.DataFrame()
    df_pcd = pd.DataFrame()

    if comparison_type == "mesh":
        path_mesh = find_all('poisson_mesh.ply',scene_path_mesh)
        list_algorithms = [os.path.basename(os.path.dirname(results_path)) for results_path in path_mesh]
        df_mesh = pd.DataFrame(columns=[scene_name]+list_algorithms).set_index(scene_name)
        '''
        for results_path in path_mesh:
	            # read mesh
	            algorithm = os.path.basename(os.path.dirname(results_path))
	            results_data = read_mesh(results_path)
	            o3d.visualization.draw_geometries_with_animation_callback([results_data], rotate_view, window_name=algorithm) # draw_geometries([results_data])
	    '''
    if comparison_type == "pcd":
        path_pcd = find_all('point_cloud.ply',scene_path_pcd)
        list_algorithms = [os.path.basename(os.path.dirname(results_path)) for results_path in path_pcd]
        columns_df = [scene_name]+list_algorithms
        df_pcd = pd.DataFrame(columns=columns_df).set_index(scene_name)
        #df_pcd = pd.DataFrame(columns=list_algorithms)

        for idx_a,results_path_a in enumerate(path_pcd):
        	algorithm_a = os.path.basename(os.path.dirname(results_path_a))
        	results_data_a = read_pcd(results_path_a)
        	list_distances = []
        	for idx_b,results_path_b in enumerate(path_pcd):
        		algorithm_b = os.path.basename(os.path.dirname(results_path_b))
        		results_data_b = read_pcd(results_path_b)

        		# metric: pcd distance
        		distances = results_data_a.compute_point_cloud_distance(results_data_b)
        		mean_distance = np.nanmean(distances)
        		list_distances.append(mean_distance)
            
        	# save comparisons
        	df_pcd.loc[algorithm_a]=list_distances
            
        	# visualize comparison
        	if vis_flag is True:
        		for idx_b,results_path_b in enumerate(path_pcd):
        			algorithm_b = os.path.basename(os.path.dirname(results_path_b))
        			results_data_b = read_pcd(results_path_b)
        			# visualize red-green
        			results_data_a.paint_uniform_color([1, 0, 0])
        			results_data_b.paint_uniform_color([0, 1, 0])
        			window_name = algorithm_a + " (red)" + " & " + algorithm_b + " (green)"
        			o3d.visualization.draw_geometries_with_animation_callback([results_data_a,results_data_b], rotate_view, window_name=window_name) # draw_geometries([results_data])
        print(df_pcd)
        csv_name = "comparison_pcd_"+scene_name+'.csv'
        df_pcd.to_csv(csv_name,sep=',')
        md_name = "comparison_pcd_"+scene_name+'.md'
        with open(md_name, 'w') as md:
            df_pcd.to_markdown(buf=md) # tablefmt="grid"
