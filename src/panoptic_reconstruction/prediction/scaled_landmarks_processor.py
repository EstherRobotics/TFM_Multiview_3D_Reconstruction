import os 
import cv2
import pandas as pd 
from pytorch_toolbelt.utils import read_rgb_image

from .predictor.predictor import FaceMeshPredictor 
from panoptic_reconstruction.utils import OriginReferenceHandler


class ScaledLandmarksProcessor: 
	"""
    A class to process and manage landmarks for images. The prediction is done in the cropped images, to have a clear visualization of the faces.
	The landmarks are scaled to the real size of the original images.
	"""
	def __init__(self, paths: dict) -> None:
		"""
        Initialize the processor with directories for cropped images, origin files and sequence images.
        """
		self._save_cropped_imgs_dir = paths['save_cropped_imgs_dir']
		self._save_origin_dir =  paths['save_origin_dir']
		self._save_seq_images_folder =  paths['save_seq_images_folder']

	
	def predict_head_mesh(self, ref_origin: pd.DataFrame, hd_ori: int) -> list:
		"""
        Predict facial landmarks and scale their coordinates to the original image size, before it was cropped. 

        Args:
            ref_origin (pd.DataFrame): DataFrame containing origin references.
            hd_ori (int): Index of the current image in the reference DataFrame.

        Returns:
            list: A list of scaled landmark coordinates.
		"""
		# Read and predict the facial landmarks
		image = read_rgb_image(ref_origin['file'][hd_ori])
		predictor = FaceMeshPredictor.dad_3dnet()
		predictions = predictor(image)

        # Scale the coordinates of the predicted landmarks
		new_projected = []
		for pt in predictions['projected_vertices'][0]:        
			new_projected.append([pt[0]+ref_origin['x'][hd_ori], pt[1]+ref_origin['y'][hd_ori]])

		return new_projected
	

	def get_all_slandmarks(self, all_cropped_imgs: list) -> tuple:
		"""
        Process a list of cropped images to extract their facial landmarks and save them.

        Args:
            all_cropped_imgs (list): List of filenames for cropped images.

        Returns:
            tuple: 
                - all_slandmarks (list): Scaled landmark coordinates for all images.
                - all_ref_origin (list): List of reference origin DataFrames for each image.
                - all_hd_ori (list): List of indices pointing to the origins of each image.
		"""
		all_slandmarks, all_ref_origin, all_hd_ori = [], [], []

		for c, cropped_img_name in enumerate(all_cropped_imgs): 
			current_cam = '_'.join([cropped_img_name.split('_')[0], cropped_img_name.split('_')[1]])

			# Get origin coordinates for cropped images. These are the coordinates where the face was cropped from the original image.
			origin_path = os.path.join(self._save_origin_dir, f'origin_{current_cam}.csv')
			ret, ref_origin, hd_ori = OriginReferenceHandler.get_ref_origin(origin_path, cropped_img_name)

			if ret:
				# Saves name and hd of the cropped image
				all_ref_origin.append(ref_origin)
				all_hd_ori.append(hd_ori)

				# Resize and save image
				save_image_path = os.path.join(self._save_seq_images_folder, cropped_img_name)
				image_res = cv2.resize(cv2.imread(ref_origin['file'][hd_ori]), (256,256))
				cv2.imwrite(save_image_path, image_res)
				
				# Predict landmarks for cropped image 
				print("PREDICTING LANDMARKS")
				print(cropped_img_name)
				predicted_slandmarks = self.predict_head_mesh(ref_origin, hd_ori)
				all_slandmarks.append(predicted_slandmarks)

		return all_slandmarks, all_ref_origin, all_hd_ori

