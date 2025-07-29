import os
import cv2
import scipy 
import numpy as np


class FacePose: 
	"""
	A class to detect and classify the orientation of a face in an image.

	The face orientation can be classified into the following:
		- FRONT: `x_left_ear < nose < x_right_ear`
		- BACK or WRONG: `x_right_ear < nose < x_left_ear` or special cases/out of limits/etc.
		- LEFT PROFILE: `nose < x_left_ear and nose < x_right_ear`
		- RIGHT PROFILE: `nose > x_left_ear and nose > x_right_ear`
	"""

	def __init__(self, img: np.ndarray, points: dict) -> None:
		"""
		Initializes the FacePose class with the input image and facial points.

		Args:
			img (ndarray): The image containing the face(s).
			points (dict): A dictionary of facial feature points for each face ID.
		"""

		self._points = points
		self._img = img
		self._ids = list(points.keys())
		self._clear_faces = {} # It saves if the face with the associated ID is seen correctly in the image
		self._poses = {} # Saves the pose of the face detected


	def detect_face_pose(self) -> tuple: 
		"""
		Detects the orientation of each face and determines its visibility.

		Returns:
			tuple: 
				- dict: Visibility status of each face (True/False).
				- dict: Pose classification of each face (-1 for undetermined,
						0 for frontal, 1 for left profile, 2 for right profile).
		"""
		# Check if face is seen correctly depending on points 
		for id_ in self._ids: 

			# By default, the face is not seen correctly
			self._clear_faces[id_] = False
			p = self._points[id_]

			if(self.has_required_points(p)):  
				self._poses[id_] = -1  # Saves the pose of the face as undetermined as default
				if not self.ear_nose_distance_correct(p, id_):
					continue

				# Distance nose-chest
				dist_nc = np.linalg.norm(np.linalg.norm(p['nose'] - p['chest']))

				if(self.is_wrong_pose(p, dist_nc)):  
					self._clear_faces[id_] = False  # Face is not visible
					print("Back pose / Wrong detection")
					
				elif(self.is_left_profile(p)):
					self._clear_faces[id_] = True
					self._poses[id_] = 1                    
					print("Left pose")

				elif(self.is_right_profile(p)):
					self._clear_faces[id_] = True
					self._poses[id_] = 2
					print("Right pose")
				else: 
					self._clear_faces[id_] = True
					self._poses[id_] = 0
					print("Frontal pose")

		return self._clear_faces, self._poses


	def ear_nose_distance_correct(self, p: dict, id_: int) -> bool:
		"""
		Checks if the distance between the nose and ears is correct.

		Args:
			p (dict): Points corresponding to a face's features.
			id_ (int): The ID of the face being checked.

		Returns:
			bool: True if the distances are acceptable, False otherwise.
		"""
		# Distance nose - right/left ear 
		dist_r = np.linalg.norm(p['nose'] - p['r_ear'])
		dist_l = np.linalg.norm(p['nose'] - p['l_ear'])  

		# If distance is large, face points are not correct 
		if(dist_l>200 or dist_r>200):
			self._clear_faces[id_] = False
			print("Incorrect detection of nose or ears")
			return False 

		return True


	def has_required_points(self, p: dict) -> bool:
		"""
		Checks if the face has all required points for orientation detection.

		Args:
			p (dict): Points corresponding to a face's features.

		Returns:
			bool: True if all required points are present, False otherwise.
		"""
		required_keys = {'lips', 'r_eye', 'l_eye', 'nose', 'l_ear', 'r_ear', 'chest'}
		return all(key in p for key in required_keys)


	def is_wrong_pose(self, p: dict, dist_nc: float) -> bool:
		"""
		Determines if the face is in a back pose or incorrectly detected.

		Args:
			p (dict): Points corresponding to a face's features.
			dist_nc (float): Distance between the nose and chest.

		Returns:
			bool: True if the face is in a back pose or misaligned, False otherwise.
		"""	
		h,w = self._img.shape[0:2]

		# Check if nose or other key points exceed image boundaries
		points_out_of_bounds = (
			p['nose'][0] + 40 > w or p['nose'][1] + 40 > h or
			p['l_ear'][0] + 30 > h and p['r_ear'][0] + 30 > h or
			p['l_ear'][1] + 30 > w and p['r_ear'][1] + 30 > w or
			p['l_ear'][0] - 30 < 0 and p['r_ear'][0] - 30 < 0 or
			p['l_ear'][1] - 30 < 0 and p['r_ear'][1] - 30 < 0
		)

		# Check vertical misalignment of eyes
		eyes_misaligned = abs(p['l_eye'][1] - p['r_eye'][1]) > 50

		# Check if nose-chest distance is excessive
		nose_chest_too_far = (
			(p['l_ear'][1]>p['chest'][1] or p['r_ear'][1]>p['chest'][1]) and
			dist_nc > 70
		)
		
		# Check if chest is above facial features
		chest_above_face = (
			p['chest'][1] - 10 < p['nose'][1] or
			p['chest'][1] - 10 < p['r_ear'][1] or
			p['chest'][1] - 10 < p['l_ear'][1] or
			p['chest'][1] - 10 < p['r_eye'][1] or
			p['chest'][1] - 10 < p['l_eye'][1]
		)

		# Check for oblique profiles and misaligned facial features
		profile_oblique = (

			# Frontal-like face but oblique profile detected
			(p['l_ear'][0] - 25 < p['nose'][0] < p['r_ear'][0] + 25) or

			# Oblique right profile
			(
				(p['l_eye'][0] - 15) < p['r_eye'][0] and
				(p['l_eye'][1] - 10) < p['r_eye'][1] and
				(
					(p['r_ear'][0] - p['l_ear'][0]) > 15 or
					abs(p['r_ear'][1] - p['l_ear'][1]) > 20
				)
			) or

			# Oblique left profile
			(
				(p['l_eye'][0] - 15) < p['r_eye'][0] and
				(p['r_eye'][1] - 10) < p['l_eye'][1] and
				(
					(p['r_ear'][0] - p['l_ear'][0]) > 15 or
					abs(p['r_ear'][1] - p['l_ear'][1]) > 20
				)
			) or

			# Both ears behind the nose with significant distance
			(
				(p['l_ear'][0] - 25 < p['nose'][0]) and
				(p['r_ear'][0] - 25 < p['l_ear'][0]) and
				(p['r_ear'][0] - p['l_ear'][0]) > 30
			)
		)

		# Combine all conditions
		return (
			eyes_misaligned or
			chest_above_face or
			points_out_of_bounds or
			nose_chest_too_far or
			profile_oblique
		)


	def is_left_profile(self, p: dict) -> bool:
		"""
		Determines if the face is a left profile.

		Args:
			p (dict): Points corresponding to a face's features.

		Returns:
			bool: True if the face is a left profile, False otherwise.
		"""
		return p['nose'][0] - 20 < min(p['l_ear'][0], p['r_ear'][0])


	def is_right_profile(self, p: dict) -> bool:
		"""
		Determines if the face is a right profile.

		Args:
			p (dict): Points corresponding to face features.

		Returns:
			bool: True if the face is a right profile, False otherwise.
		"""			
		return p['nose'][0] + 20 > max(p['l_ear'][0], p['r_ear'][0])

