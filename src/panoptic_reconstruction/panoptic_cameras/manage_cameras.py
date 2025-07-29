import cv2
import json
import random 
import numpy as np 


class ManageCameras:
    """
    A class to manage camera operations including:
    - Camera functionality checking
    - Loading and managing calibration parameters
    - Camera selection for reconstruction
    - Camera pair selection
    """
    
    def __init__(self, calibration_path: str, cam_num: int) -> None:
        """
        Initialize the ManageCameras class.

        Args:
            calibration_path: Path to the camera calibration JSON file
            cam_num: Identifier for the camera to manage
        """
        self._calibration_path = calibration_path
        self._cam_num = cam_num 
        # HSV color range for detecting green screen (camera not working)
        self._lower = np.array([59, 254, 134], np.uint8)
        self._upper = np.array([61, 255, 136], np.uint8)

        self._cameras = None
        self._cam = None


    def not_camera_working(self, img: np.ndarray) -> bool:
        """
        Check if camera is not working by detecting green screen.
        
        Args:
            img: Input image in RGB format
            
        Returns:
            bool: True if camera shows green screen (not working), False otherwise
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, self._lower, self._upper)
        cam_working = np.all(green_mask == 255)

        return cam_working


    def get_cameras(self) -> tuple:
        """
        Load camera calibration parameters and process for current camera.
        
        Returns:
            tuple:
                - cameras: Dictionary containing all cameras' calibration data with keys (panel,node)
                - cam: Dictionary containing current camera's calibration data with keys:
                    - K: Intrinsic matrix (3x3 numpy matrix)
                    - distCoef: Distortion coefficients (1x5 numpy array)
                    - R: Rotation matrix (3x3 numpy matrix)
                    - t: Translation vector (3x1 numpy array)
        """
        # Load camera calibration parameters from JSON
        with open(self._calibration_path) as cfile:
            calib = json.load(cfile)

        # Organize cameras by (panel, node) tuple keys
        self._cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

        # Convert calibration parameters to numpy arrays
        for k, cam in self._cameras.items():    
            cam['K'] = np.matrix(cam['K'])
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['R'] = np.matrix(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3,1))
        
        # Load current camera parameters
        self._cam = self._cameras[(0, self._cam_num)]

        return self._cameras, self._cam


    @staticmethod
    def get_actual_cams(order_inliers_idx: list, sel_scams: list, 
                       sel_scams_id: list, num_cams: int) -> tuple:
        """
        Select cameras for reconstruction based on inlier ordering.
        
        Args:
            order_inliers_idx: List of camera indices ordered by inlier count
            sel_scams: List of selected camera dictionaries
            sel_scams_id: List of selected camera IDs
            num_cams: Number of cameras to select
            
        Returns:
            tuple:
                - new_sel_scams: List of selected camera dictionaries
                - new_sel_scams_id: List of selected camera IDs
        """
        new_sel_scams_id = []
        # Select top num_cams cameras based on inlier ordering
        for j in range(len(sel_scams_id)-1, len(sel_scams_id)-1-num_cams, -1):
            new_sel_scams_id.append(sel_scams_id[order_inliers_idx[j]])

        # Get camera parameters for selected IDs
        new_sel_scams = [sel_scams[idx] for idx in new_sel_scams_id]

        return new_sel_scams, new_sel_scams_id


    @staticmethod
    def camera_number_for_reconstruction(inliers_cameras: list) -> int:
        """
        Determine optimal number of cameras to use for reconstruction.
        
        Args:
            inliers_cameras: List of inlier counts per camera
            
        Returns:
            int: Number of cameras to use
        """
        num_cams = np.count_nonzero(inliers_cameras)
        if num_cams == 0:
            num_cams = len(inliers_cameras)
        else:
            if num_cams >= 4:
                counter = sum(1 for val in inliers_cameras if val >= 3)
                if counter >= 4:
                    num_cams = counter
        
        return num_cams


    @staticmethod
    def select_cameras_id(cams_id: list, combination: list) -> list:
        """
        Select camera IDs based on combination indices.
        """
        return [cams_id[idx] for idx in combination]


    def select_cameras(self, cams_id: list) -> list:
        """
        Get camera parameters for given camera IDs.
        """
        return [self._cameras[0, idx] for idx in cams_id]


    @staticmethod
    def select_cameras_from_set(cams: list, cams_id: list) -> list:
        """
        Select cameras from a predefined set.
        """
        return [cams[idx] for idx in cams_id]


    @staticmethod
    def select_random_camera_pair(num_cameras: int) -> list:
        """
        Randomly select a pair of camera indices.
        """
        return random.sample(range(num_cameras), 2)