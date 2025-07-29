import os
import cv2
import scipy 
import numpy as np 


class FaceCropper: 
	"""
    A class for detecting and cropping faces from an image based on detected points and poses.
    Saves cropped images and metadata for further use.
	"""
	
	def __init__(self, seq: dict, save_cropped_imgs_dir: str, img: np.ndarray, points: dict, cam: str) -> None:
		"""
		Initializes the FaceCropper with input data and configurations.

        Args:
            save_cropped_imgs_dir (str): Directory to save cropped face images.
            path_img (str): Path to the input image.
            img (ndarray): The input image as a NumPy array.
            points (dict): A dictionary with detected landmarks and body points.
            cam (str): Camera identifier.
            seq_info (dict): Metadata and sequence information for cropped faces.
			
		"""
		self._save_cropped_imgs_dir = save_cropped_imgs_dir
		self._path_img = seq['path_img']
		self._img = img
		self._points = points
		self._cam = cam	
		self._seq = seq
		
		
	def get_limits_and_crop_face(self, clear_faces: dict, poses: dict) -> dict:
		"""
		Calculates the cropping limits for each detected face and saves cropped face images.

		Args:
			clear_faces (dict): Dictionary indicating visibility status for each detected face (True/False).
			poses (dict): Dictionary mapping face IDs to their pose classification.

		Returns:
			dict: Updated sequence information after cropping.
		"""
		
		for id_ in clear_faces: 
			nel_pts = np.array([self._points[id_]['nose'], self._points[id_]['l_ear'], self._points[id_]['r_ear'], self._points[id_]['lips']]).T

			# Calculate cropping coordinates (limits)
			limits, scale, w_max = self.calculate_face_limits(nel_pts)

			# Adjust cropping coordinates (limits) based on the pose and calculated size
			limits = self.adjust_limits_by_pose(poses, limits, w_max, scale, id_)			

			# Crop and save the face image along with metadata
			self.crop_and_save_face_image(self._save_cropped_imgs_dir, id_, limits)

		return  self._seq
		
		
	def calculate_face_limits(self, nel_pts: np.ndarray) -> tuple:
		"""	
        Determines the cropping limits for a face based on detected landmarks.

        Args:
            nel_pts (ndarray): Array of coordinates for the nose, ears, and lips.

        Returns:
            tuple: 
                - limits (list): Cropping boundaries [x_min, x_max, y_min, y_max].
                - scale (float): Scale factor indicating relative size of the face.
                - w_max (bool): Whether the face width is greater than its height.
        """
		# Get indices of the x and y landmarks of the previous parts within the image boundaries
		indices_x = np.array(np.where(np.logical_and(nel_pts[0,:] >= 30, nel_pts[0,:] <= self._img.shape[1]-30))).flatten()
		indices_y = np.array(np.where(np.logical_and(nel_pts[1,:] >= 30, nel_pts[1,:] <= self._img.shape[0]-30))).flatten()

        # Get min/max values for x and y based on nose, ears, and lips
		min_idx = indices_x[np.argmin(nel_pts[0,:][indices_x])]
		max_idx = indices_x[np.argmax(nel_pts[0,:][indices_x])]

		min_idy = indices_y[np.argmin(nel_pts[1,:][indices_y])]
		max_idy = indices_y[np.argmax(nel_pts[1,:][indices_y])]

        # Calculate approximate width and height of the face with added limits
		w_face = nel_pts[0,:][max_idx]-nel_pts[0,:][min_idx] + 40
		h_face = nel_pts[1,:][max_idy]-nel_pts[1,:][min_idy] + 100 + 35
		w_max = w_face>h_face
		maxim_face = max(w_face, h_face) 

        # Calculate scaling factor depending on approximate face size
		scale = maxim_face/180

        # Calculate and save cropping coordinates
		x_min = int(nel_pts[0,:][min_idx]-scale*20)
		x_max = int(nel_pts[0,:][max_idx]+scale*20)
		y_min = int(nel_pts[1,:][min_idy]-scale*100) 
		y_max = int(nel_pts[1,:][max_idy]+scale*35)

		limits = [x_min, x_max, y_min, y_max]

		return limits, scale, w_max
		
	
	def crop_and_save_face_image(self, save_cropped_imgs_dir: str, id_: str, limits: list) -> dict:
		"""
        Crops the face from the image based on the calculated limits and saves it.

        Args:
            save_cropped_imgs_dir (str): Directory to save the cropped face images.
            id_ (str): Identifier for the detected face.
            limits (list): Cropping boundaries [x_min, x_max, y_min, y_max].

        Returns:
            dict: Updated sequence information with saved metadata.
        """		
		x_min, x_max, y_min, y_max = limits
		h = y_max-y_min
		w = x_max-x_min

		if(h==w and x_min>=0 and y_min>=0 and x_max<=self._img.shape[1] and y_max<=self._img.shape[0]):
			# Crop face from image
			face = self._img[y_min:y_max, x_min:x_max]

			# Save cropped face image and metadata
			name_img = f"{self._path_img.split('/')[-1].split('.')[0]}_{id_}.jpg"
			output_path = os.path.join(save_cropped_imgs_dir, str(id_), name_img)		
			self._seq = self.save_image_and_metadata(output_path, face, id_, limits)
	
		return self._seq
		
		
	def adjust_limits_by_pose(self, poses: dict, limits: list, w_max: bool, scale: float, id_: str) -> list:	
		"""
        Adjusts cropping limits based on face pose.

        Args:
            poses (dict): Pose classification for each face ID.
            limits (list): Initial cropping boundaries.
            w_max (bool): Whether the face width is greater than its height.
            scale (float): Scaling factor.
            id_ (str): Identifier for the detected face.

        Returns:
            list: Updated cropping boundaries [x_min, x_max, y_min, y_max].
        """
		x_min, x_max, y_min, y_max = limits
		w = x_max-x_min
		h = y_max-y_min

		# Extra limits to add to make sure the complete face cropped will be visible 
		add_lim1 = abs(h-w)//2
		add_lim2 = abs(h-w) - add_lim1
		
		# If the width is larger, the limits are added to the y values
		if(w_max):
			y_min = y_min - add_lim1
			y_max = y_max + add_lim2

		# If the height is larger, the limits are added to the x values
		else:
			x_min = x_min - add_lim1
			x_max = x_max + add_lim2
			
		# Depending on the pose, move the limits
		if(poses[id_] == 2):   # Right profile
			x_min = x_min - int(scale*20)
			x_max = x_max - int(scale*20)

		elif(poses[id_]==1):   # Left profile
			x_min = x_min + int(scale*20)
			x_max = x_max + int(scale*20) 

		limits = [x_min, x_max, y_min, y_max]
		return limits 
		
		
	def save_image_and_metadata(self, output_path: str, face: np.ndarray, id_: str, limits: list) -> dict:	
		"""
        Saves the cropped face image and its metadata.

        Args:
            output_path (str): Path to save the cropped image.
            face (ndarray): Cropped face image.
            id_ (str): Identifier for the detected face.
            limits (list): Cropping boundaries [x_min, x_max, y_min, y_max].

        Returns:
            dict: Updated sequence information with saved metadata.
        """
		# Creates folder to save face image
		os.makedirs(os.path.dirname(output_path), exist_ok=True) 
		cv2.imwrite(output_path, face) 
		print("Image saved in ", output_path)

		if(id_ not in self._seq['all_ppl_idx']):
			self._seq['all_ppl_idx'].append(int(id_))

		# Saves coordinates where it was cropped with file name	for origin json 
		self._seq['x'].append(limits[0])  # Saves x_min
		self._seq['y'].append(limits[2])  # Saves y_min
		self._seq['output_files'].append(output_path)
		
		# Saves camera and face size for actual cropped face
		if id_ in self._seq['sel_scams']:
			self._seq['sel_scams'][id_].append(self._cam)
			self._seq['sel_scams_id'][id_].append(self._seq['cam_num'])
			self._seq['all_tams'][id_].append(face.shape[0])
		else: 
			self._seq['sel_scams'][id_] = [self._cam]
			self._seq['sel_scams_id'][id_] = [self._seq['cam_num']]
			self._seq['all_tams'][id_] = [face.shape[0]]

		return self._seq