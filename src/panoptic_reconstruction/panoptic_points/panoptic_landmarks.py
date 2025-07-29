import json
import os
import torch
import numpy as np
from panoptic_reconstruction.utils import panutils  


class PanopticLandmarks:
    """Class for handling panoptic facial landmarks processing."""
    
    # Class constant defining facial landmark indices grouped by facial features
    FACIAL_LANDMARK_INDICES = {
        'brows': [4, 0, 5, 9],
        'eye_left': [23, 24, 19, 20, 21, 22],
        'eye_right': [30, 29, 28, 27, 26, 25],
        'nose': [10, 12, 14, 16, 18],
        'lips': [49, 40, 45, 31, 34, 37]
    }
    
    @staticmethod
    def get_facial_panoptic_landmarks(panoptic_path: str, hd_idx: int, ppl_idx: int, sel_scams: list) -> list:
        """
        Static method to get panoptic facial points for a specific frame and person.
        
        Args:
            panoptic_path (str): Path to panoptic dataset root directory
            hd_idx (int): HD frame index (timestamp)
            ppl_idx (int): Person index to extract landmarks for
            sel_scams (list): List of selected camera dictionaries, each containing:
                - K: Camera intrinsic matrix
                - R: Camera rotation matrix
                - t: Camera translation vector
                - distCoef: Camera distortion coefficients
                
        Returns:
            list: List of reprojected panoptic points for each selected camera. 
                  Each element is a list of 2D points (x,y) for the facial landmarks.
        """
        # Load face data from JSON file
        face_data = PanopticLandmarks._load_face_data(panoptic_path, hd_idx)
        reproj_panoptic = []
        
        # Process each person in the frame
        for face in face_data['people']:
            if face['id'] == ppl_idx:
                # Project face points for each selected camera
                for cam in sel_scams:
                    pt = PanopticLandmarks._project_face_points(face, cam)
                    # Select specific facial landmarks
                    reproj_panoptic.append(PanopticLandmarks._select_landmarks(pt))
        
        return reproj_panoptic


    @staticmethod
    def _load_face_data(panoptic_path: str, hd_idx: int) -> dict:
        """
        Loads face data from JSON file.
        
        Args:
            panoptic_path (str): Path to panoptic dataset
            hd_idx (int): HD frame index
            
        Returns:
            dict: Dictionary containing face data with 'people' key holding person data
        """
        hd_face_json_path = os.path.join(panoptic_path, 'hdFace3d/')
        filename = os.path.join(hd_face_json_path, f'faceRecon3D_hd{hd_idx:08d}.json')
        
        with open(filename) as f:
            return json.load(f)


    @staticmethod
    def _project_face_points(face_data: dict, cam: dict) -> np.ndarray:
        """
        Projects 3D face points to 2D using camera parameters.
        
        Args:
            face_data (dict): Dictionary containing face data with 'face70' key holding landmarks
            cam (dict): Camera parameters dictionary with:
                - K: Intrinsic matrix
                - R: Rotation matrix
                - t: Translation vector
                - distCoef: Distortion coefficients
                
        Returns:
            np.ndarray: Array of projected 2D points (N x 2)
        """
        face3d = np.array(face_data['face70']['landmarks']).reshape((-1, 3)).transpose()
        pt = panutils.projectPoints(
            face3d,
            cam['K'], 
            cam['R'], 
            cam['t'], 
            cam['distCoef']
        )
        return np.array([pt[0, 17:], pt[1, 17:]]).T


    @staticmethod
    def _select_landmarks(pt_face: np.ndarray) -> list:
        """
        Selects specific facial landmarks from all available points.
        
        Args:
            pt_face (np.ndarray): Array containing all facial points (N x 2)
            
        Returns:
            list: List of selected landmark points as [x,y] pairs
        """
        # Get all indices from our facial landmark groups
        all_indices = np.concatenate(list(PanopticLandmarks.FACIAL_LANDMARK_INDICES.values()))
        return [[pt_face[idx][0], pt_face[idx][1]] for idx in all_indices]