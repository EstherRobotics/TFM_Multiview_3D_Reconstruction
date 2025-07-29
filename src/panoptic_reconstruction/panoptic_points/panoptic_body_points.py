import json 
from panoptic_reconstruction.utils import panutils
import numpy as np


class PanopticBodyPoints: 
	"""
    Class that handles the reprojection of 3D points to 2D for the human body from the panoptic database. 
	It uses JSON files containing 3D point coordinates as well as camera information to project those points onto a 2D plane.
	"""
	def __init__(self, paths: dict, hd_idx: int, cam: dict) -> None:
		"""
        Initializes the ReprojectionPanopticPoints object with the file paths and camera information.

        Args:
            hd_json_paths (list): 
            hd_idx (int): Frame index used to load specific data files.
            cam (dict): Camera calibration data including 'K' (intrinsic matrix), 'R' (rotation), 't' (translation), and 'distCoef' (distortion coefficients).
		"""
		self._points = {}
		self._hd_skel_json_path = paths['hd_skel_json_path']
		self._hd_face_json_path = paths['hd_face_json_path']
		self._hd_hand_json_path = paths['hd_hand_json_path']
		self._hd_idx = hd_idx
		self._cam = cam


	def get_all_points(self) -> tuple:
		"""
        Retrieves and projects all 3D points (face, chest, lips and hands) to 2D.

		Returns:
            tuple: 
                - ret (bool): True if all points were successfully extracted and projected, False otherwise.
                - points (dict): A dictionary containing all projected 2D points for each person.
		"""
		# Body points			
		ret, points = self.get_body_points()
		if ret:
			# Lip points
			ret, points = self.get_lip_points()
			if ret:
				# Hand points 
				ret, points = self.	get_hand_points()
		return ret, points 
		

	def get_body_points(self)-> tuple:
		"""
        Extracts and projects the 3D face and chest keypoints to 2D.

        Returns:
            tuple: 
                - ret (bool): True if points were successfully extracted and projected, False otherwise.
                - _points (dict): A dictionary with the projected 2D face points for each detected person.
        """
		pt_body = {}
		ret = True

		try:
			# Load the json file with this frame's skeletons
			skel_json_fname = '{0}/body3DScene_{1:08d}.json'.format(self._hd_skel_json_path, self._hd_idx)
			with open(skel_json_fname) as dfile:
				bframe = json.load(dfile)

			# Cycle through all detected bodies
			for person in bframe['bodies']:
				
				# There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
				# where c1 ... c19 are per-joint detection confidences
				skel = np.array(person['joints19']).reshape((-1,4)).transpose()
				
				# Projectar punto de la nariz, oreja izq, oreja dcha, pecho, ojo izq, ojo dcho y chest
				face_points = [[subskel[1], subskel[16], subskel[18], subskel[15], subskel[17], subskel[0]] for subskel in skel[0:3,:]]

				# Project skeleton into view (this is like cv2.projectPoints)
				pt_body[person['id']] = panutils.projectPoints(face_points,
									self._cam['K'], self._cam['R'], self._cam['t'], 
									self._cam['distCoef'])

				# Save face points
				self._points[person['id']] = {}
				self._points[person['id']]['nose'] = pt_body[person['id']][0:2,0]
				self._points[person['id']]['l_ear'] = pt_body[person['id']][0:2,1]
				self._points[person['id']]['r_ear'] = pt_body[person['id']][0:2,2]
				self._points[person['id']]['l_eye'] = pt_body[person['id']][0:2,3]
				self._points[person['id']]['r_eye'] = pt_body[person['id']][0:2,4]
				self._points[person['id']]['chest'] = pt_body[person['id']][0:2,5]

		except IOError as e:
			print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)
			ret = False

		# If any point was detected
		if(not pt_body):
			ret = False
		return ret, self._points


	def get_lip_points(self) -> tuple:
		"""
        Extracts and projects the 3D lip keypoints to 2D.

        Returns:
            tuple: 
                - ret (bool): True if lip points were successfully extracted and projected, False otherwise.
                - _points (dict): A dictionary with the projected 2D lip points for each detected person.
        """
		pt_lips = {}
		ret = True
		
		try:
			# Load the json file with this frame's face
			face_json_fname = '{0}/faceRecon3D_hd{1:08d}.json'.format(self._hd_face_json_path, self._hd_idx)
			with open(face_json_fname) as dfile:
				fframe = json.load(dfile)

			# Cycle through all detected faces
			for person in fframe['people']:
				
				# 3D Face has 70 3D joints, stored as an array [x1,y1,z1,x2,y2,z2,...]
				face3d = np.array(person['face70']['landmarks']).reshape((-1,3)).transpose()
				lips_points = [[f[67]] for f in face3d[0:3,:]]

				# Project skeleton into view (this is like cv2.projectPoints)
				pt_lips[person['id']] = panutils.projectPoints(lips_points,
									self._cam['K'], self._cam['R'], self._cam['t'], 
									self._cam['distCoef'])
				
				# Save lips points
				self._points[person['id']]['lips'] = np.array(pt_lips[person['id']][0:2]).reshape(-1)

		except IOError as e:
			print('Error reading {0}\n'.format(face_json_fname)+e.strerror)
			ret = False

		if(not pt_lips):
			ret = False

		return ret, self._points


	def get_hand_points(self) -> tuple:
		"""
        Extracts and projects the 3D hand keypoints (left and right) to 2D.

        This method processes the 3D hand data and projects the left and right hand keypoints to 2D.

        Returns:
            tuple: 
                - ret (bool): True if hand points were successfully extracted and projected, False otherwise.
                - _points (dict): A dictionary with the projected 2D hand points for each detected person.
        """
		pt_lh = {}
		pt_rh = {}
		ret = True

		try:
			# Load the json file with this frame's face
			hand_json_fname = '{0}/handRecon3D_hd{1:08d}.json'.format(self._hd_hand_json_path, self._hd_idx)
			with open(hand_json_fname) as dfile:
				hframe = json.load(dfile)

			# Cycle through all detected hands
			for person in hframe['people']:
				
				# 3D hands, right_hand and left_hand, have 21 3D joints, stored as an array [x1,y1,z1,x2,y2,z2,...]
				if 'right_hand' in person: 
					hand3d = np.array(person['right_hand']['landmarks']).reshape((-1,3)).transpose()

					# Project skeleton into view (this is like cv2.projectPoints)
					pt_rh[person['id']] = panutils.projectPoints(hand3d,
								self._cam['K'], self._cam['R'], self._cam['t'], 
								self._cam['distCoef'])
					self._points[person['id']]['pt_rh'] = pt_rh[person['id']][0:2].T

				if 'left_hand' in person: 
					hand3d = np.array(person['left_hand']['landmarks']).reshape((-1,3)).transpose()

					# Project skeleton into view (this is like cv2.projectPoints)
					pt_lh[person['id']] = panutils.projectPoints(hand3d,
							self._cam['K'], self._cam['R'], self._cam['t'], 
							self._cam['distCoef'])
					self._points[person['id']]['pt_lh'] = pt_lh[person['id']][0:2].T

		except IOError as e:
			print('Error reading {0}\n'.format(hand_json_fname)+e.strerror)
			ret = False

		return ret, self._points



