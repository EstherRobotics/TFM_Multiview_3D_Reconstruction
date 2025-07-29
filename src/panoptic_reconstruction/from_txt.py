#!/usr/bin/env python

import os
import sys
import cv2
import time
import glob
import copy 
import scipy
import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm 

from .utils import delete_auxiliar_data, read_df_file
from .utils import panutils
from .utils import OriginReferenceHandler
from .utils import SequenceInfo

from .config import ManagePaths
from .panoptic_cameras import ManageCameras
from .panoptic_points import PanopticBodyPoints
from .face_processor import FaceDetection
from .prediction import ScaledLandmarksProcessor
from .reconstruction import ReconstructProcessor


def main(file_method: str): 

	# To calculate execution time
	mean_time = []

	# Paths initalization ands variables
	manage_paths = ManagePaths()
	paths = manage_paths.get_setup_paths()
	df_path = manage_paths.get_df_path(file_method)
	df = read_df_file(df_path)

	# For every sequence 
	# ===============================================================================================
	for path_img in tqdm(df['image']):
		seq_name = path_img.split("/")[0]
		seq = SequenceInfo.get_seq_info_init(seq_name)
		paths = manage_paths.get_general_paths(seq_name)
		manage_paths.recreate_directories()

		# For every HD image in the sequence folder
		for name_img in sorted(os.listdir(paths['hdimgs_path'])):	
			start_time = time.time()

			print("\n\nPROCESSING:", name_img)
			seq = manage_paths.get_path_img_seq_info(seq, name_img)
			#print(seq['path_img'])

			# Save origin
			# ===============================================================================================

			# Save the coordinates from which the previous images were cropped,
			# if the current sequence is different from the previous one
			seq = OriginReferenceHandler.save_json_origin(paths['save_origin_dir'], seq)

			# Extract faces
			# ===============================================================================================

			# Load the corresponding HD image
			img = manage_paths.load_hd_image(seq)
			if img is None: continue

			# Define paths
			paths = manage_paths.add_specific_paths(seq_name)

			# Load cameras number and calibration parameters for the current camera
			manage_cams = ManageCameras(paths['calib_path'], seq['cam_num'])
			cameras, cam = manage_cams.get_cameras()

			# Check that camera is working
			if(manage_cams.not_camera_working(img)): continue

			# Get the panoptic reprojected points for the body, eyes, face and hands 
			# ========================================================	
			rpp = PanopticBodyPoints(paths, seq['hd_idx'], cam)
			ret, points = rpp.get_all_points()
			if not ret: continue

			# Detect person position and manage occlusions
			# =========================================================
			face_detection = FaceDetection(seq, paths['save_cropped_imgs_dir'], img, points, cam)
			seq = face_detection.process_face_detection()


		# Detect landmarks and perform reconstruction 
		# ===============================================================================
		seq = OriginReferenceHandler.save_json_origin(paths['save_origin_dir'], seq, True)

		# Get scaled landmarks and perform 3D reconstruction 
		rec = ReconstructProcessor(manage_paths, manage_cams, paths, cameras)
		rec.process_reconstruction(seq)

		# Reset variables and get execution time 
		# ===============================================================================
		delete_auxiliar_data(paths['save_origin_dir'], paths['save_cropped_imgs_dir'])
		seq =  SequenceInfo.reset_seq_info(seq)

		final_time = time.time()
		mean_time.append(final_time-start_time)		
		print("\nExecution times: ", mean_time, "\nTotal mean time",  np.mean(mean_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running panoptic reconstruction from txt file.")
    parser.add_argument(
        "file_method", choices=["train", "test", "val"],
        help="Which txt file with names images to process: 'train', 'test', or 'val'."
    )

    args = parser.parse_args()
    main(args.file_method)
