import os
import cv2
import scipy 
import numpy as np 


class FaceOcclusion: 
	"""
    A class to handle face occlusion detection, including occlusions caused by hands or other people.
    """
	
	def __init__(self, points: dict) -> None:
		"""
        Initializes the FaceOcclusion class with the provided points.

        Args:
            points (dict): A dictionary where each key represents a face ID and 
                           the value contains the coordinates of facial features and body parts.
        """
		self._points = points
		self._ids = list(points.keys())
		
	
	def process_occlusions(self, clear_faces: dict, poses: dict) -> tuple:		
		"""
        Processes occlusions by analyzing hand-face and inter-person occlusions.

        Args:
            clear_faces (dict): A dictionary indicating the initial visibility status
                                of faces (True/False for each face ID).
            poses (dict): A dictionary mapping face IDs to their pose classification 
                          (e.g., frontal, left profile, right profile).

        Returns:
            tuple:
                - bool: True if at least one face remains clearly visible, False otherwise.
                - dict: Updated visibility status for all faces after processing occlusions.
        """
		clear_faces = self.hand_face_occlusion(clear_faces, poses)
		clear_faces = self.occlusion_people(clear_faces)
		clear = any(clear_faces.values())
			
		return clear, clear_faces
		
	
	def hand_face_occlusion(self, clear_faces: dict, poses: dict) -> dict:
		"""
        Detects whether hands are occluding the face for each detected individual.

        Args:
            clear_faces (dict): Current visibility status of faces (True/False).
            poses (dict): Pose classification for each face (frontal, left profile, right profile).

        Returns:
            dict: Updated visibility status of faces after analyzing hand-face occlusions.
        """
		for id_ in self._ids: 
			if(clear_faces[id_] == True):
				# Array with the points of the nose and eyes
				nose_eyes_points = np.array([self._points[id_]['nose'], self._points[id_]['r_eye'], self._points[id_]['l_eye']])

				# Check if the right hand of the detected person is occluding his/her own face
				if poses[id_]==2 or poses[id_]==0: # Front and right profile
					if('pt_rh' in self._points[id_]):
						dist_rh = scipy.spatial.distance.cdist(self._points[id_]['pt_rh'], nose_eyes_points, 'euclidean')
						min_right = min(np.amin(dist_rh, axis=1))
						if(min_right < 22):
							clear_faces[id_] = False  
							#print("Right hand is in the face")

				# Check if the left hand of the detected person is occluding his/her own face	
				if poses[id_]==1 or poses[id_] == 0: # Front and left profile
					if('pt_lh' in self._points[id_]):
						dist_lh = scipy.spatial.distance.cdist(self._points[id_]['pt_lh'], nose_eyes_points, 'euclidean')
						min_left = min(np.amin(dist_lh, axis=1))
						if(min_left < 22):
							clear_flear_faces[id_] = False  
							#print("Left hand is in the face")
		return clear_faces
		
	
	def occlusion_people(self, clear_faces: dict) -> dict:
		"""
        Detects whether one person is occluding another person's face.

        Args:
            clear_faces (dict): Current visibility status of faces (True/False).

        Returns:
            dict: Updated visibility status of faces after analyzing inter-person occlusions.
        """		

		# Get face points
		face_points = {
					id_: np.vstack((
						self._points[id_]['nose'][0:2],
						self._points[id_]['l_eye'][0:2],
						self._points[id_]['r_eye'][0:2],
    				))
			for id_ in self._points
		}

		still_clear_faces = clear_faces.copy()
 		# Occlusions will only be checked if more than one face is detected
		for i in range(len(self._ids)): 
			for j in range(i+1, len(self._ids)):
				if(still_clear_faces[self._ids[i]] or still_clear_faces[self._ids[j]]):
					min_right = np.iinfo(np.int64).max
					min_left  = np.iinfo(np.int64).max
					min_faces = np.iinfo(np.int64).max

                    # Check face-to-face occlusions
					dist_faces = scipy.spatial.distance.cdist(face_points[self._ids[i]],face_points[self._ids[j]], 'euclidean')
					min_faces = min(np.amin(dist_faces, axis=1))

					# Check hand-to-face occlusions
					if('pt_rh' in self._points[self._ids[j]]):
						dist_rh = scipy.spatial.distance.cdist(self._points[self._ids[j]]['pt_rh'], face_points[self._ids[i]], 'euclidean')
						min_right = min(np.amin(dist_rh, axis=1))
					
					if('pt_lh' in self._points[self._ids[j]]):
						dist_lh = scipy.spatial.distance.cdist(self._points[self._ids[j]]['pt_lh'], face_points[self._ids[i]], 'euclidean')
						min_left = min(np.amin(dist_lh, axis=1))
				
					# If any occlusion is detected, mark the faces as unclear
					if(min_faces < 80 or min_right < 35 or min_left < 35):
						#print(self._ids[i], 'and', self._ids[j], "are overlapped" )
						still_clear_faces[self._ids[i]] = False
						still_clear_faces[self._ids[j]] = False

		clear_faces = {id_: cf for id_, cf in still_clear_faces.items() if cf is True}

		return clear_faces


