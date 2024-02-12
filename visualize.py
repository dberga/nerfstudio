import open3d as o3d

import argparse
import os
import shutil

EXPORTS_FOLDER="exports" # exports
MESHES_FOLDER="meshes" # exports/meshes
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
            "--scene", nargs='?', type=str, default="CEPA01"
        )

        parser.add_argument(
            "--type", nargs='?', type=str, default="mesh" # pcd
        )
        args, unknown_args = parser.parse_known_args()
        exports_folder = args.exports
        scene_name = args.scene
        file_type = args.type
        scene_path_mesh = os.path.join(exports_folder,MESHES_FOLDER,scene_name)
        scene_path_pcd = os.path.join(exports_folder,POINTCLOUDS_FOLDER,scene_name)
        return scene_path_mesh,scene_path_pcd, scene_name, file_type, exports_folder

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
    scene_path_mesh,scene_path_pcd, scene_name, file_type, exports_folder = parse_scene_path()

    if file_type == "mesh":
        path_mesh = find_all('poisson_mesh.ply',scene_path_mesh)
        for results_path in path_mesh:
            # read mesh
            algorithm = os.path.basename(os.path.dirname(results_path))
            results_data = read_mesh(results_path)
            o3d.visualization.draw_geometries_with_animation_callback([results_data], rotate_view, window_name=algorithm) # draw_geometries([results_data])
    if file_type == "pcd":
        path_pcd = find_all('point_cloud.ply',scene_path_pcd)
        for results_path in path_pcd:
            algorithm = os.path.basename(os.path.dirname(results_path))
            results_data = read_pcd(results_path)
            o3d.visualization.draw_geometries_with_animation_callback([results_data], rotate_view, window_name=algorithm) # draw_geometries([results_data])
