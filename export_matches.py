import argparse

import numpy as np

import sqlite3
import os
import sys
from scipy.io import savemat
from subprocess import run, PIPE
from collections import defaultdict

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def image_ids_to_pair(image_id1, image_id2):
    pair_id = image_id2 + 2147483647 * image_id1
    return pair_id

def get_keypoints(cursor, image_id):
    cursor.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))        
    image_idx, n_rows, n_columns, raw_data = cursor.fetchone()
    kypnts = np.frombuffer(raw_data, dtype=np.float32).reshape(n_rows, n_columns).copy()
    kypnts = kypnts[:,0:2]  
    return kypnts

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', required=True, help='Path to the database')
    parser.add_argument('--outdir', required=True, help='Name of the output directory')
    args = parser.parse_args()
    
    filename_db = args.database_path

    print("Opening database: " + filename_db)
    if not os.path.exists(filename_db):
        print('Error db does not exist!')
        exit()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    connection = sqlite3.connect(filename_db)
    cursor = connection.cursor()
    
    list_image_ids = []
    img_ids_to_names_dict = {}
    cursor.execute('SELECT image_id, name, cameras.width, cameras.height FROM images LEFT JOIN cameras ON images.camera_id == cameras.camera_id;')
    for row in cursor:
        image_idx, name, width, height = row        
        list_image_ids.append(image_idx)
        img_ids_to_names_dict[image_idx] = name
    
    num_image_ids = len(list_image_ids)

    # Iterate over entries in the two-view geometry table
    cursor.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries;')
    all_matches = {}
    for row in cursor:
        pair_id = row[0]
        rows = row[1]
        cols = row[2]
        raw_data = row[3]
        if (rows < 5):
            continue

        matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(rows, cols)

        if matches.shape[0] < 5:
            continue

        all_matches[pair_id] = matches

    for key in all_matches:
        pair_id = key
        matches = all_matches[key]
        id1, id2 = pair_id_to_image_ids(pair_id)
        image_name1 = img_ids_to_names_dict[id1]
        image_name2 = img_ids_to_names_dict[id2]

        keys1 = get_keypoints(cursor, id1)
        keys2 = get_keypoints(cursor, id2)

        match_positions = np.empty([matches.shape[0], 4])
        for i in range(0, matches.shape[0]):
            match_positions[i,:] = np.array([keys1[matches[i,0]][0], keys1[matches[i,0]][1], keys2[matches[i,1]][0], keys2[matches[i,1]][1]])

        #outfile = os.path.join(args.outdir, image_name1.split("/")[3].split(".png")[0] + "_" + image_name2.split("/")[3].split(".png")[0] + ".txt");
        outfile = os.path.join(args.outdir, image_name1.split("/")[0].split(".jpg")[0] + "_" + image_name2.split("/")[0].split(".jpg")[0] + ".txt")
        np.savetxt(outfile, match_positions, delimiter=' ')

    cursor.close()
    connection.close()
