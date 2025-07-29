import os
import cv2
import scipy 
import numpy as np 

from .face_pose import FacePose
from .face_cropper import FaceCropper
from .face_occlusion import FaceOcclusion


class FaceDetection: 
    """
    This class integrates face pose estimation, occlusion detection and face for detecting and processing faces in an image.
    """
    
    def __init__(self, seq: dict, save_cropped_imgs_dir: str, img: np.ndarray, points: dict, cam: str) -> None:
        """
        Initializes the FaceDetection class with the input image and configurations.

        Args:
            path_img (str): Path to the input image.
            save_cropped_imgs_dir (str): Directory to save cropped face images.
            img (ndarray): Input image as a NumPy array.
            points (dict): A dictionary containing detected landmarks and body points for each face ID.
            cam (str): Identifier for the camera used.
            seq_info (dict): Metadata and sequence information for processing.
        """        
        self._points = points
        self._img = img
        self._ids = list(points.keys())
        self._cam = cam
        self._path_img = seq['path_img']
        self._seq = seq
                
        # Initialize face processing classes
        self.face_pose = FacePose(img, points)
        self.face_occlusion = FaceOcclusion(points)
        self.face_cropper = FaceCropper(seq, save_cropped_imgs_dir, img, points, cam)


    def process_face_detection(self) -> dict: 
        """
        Processes the input image to handle face detection, pose estimation, occlusion checking and cropping.

        Returns:
            dict: Updated sequence information (`_seq`) containing metadata for cropped faces.
        """
        # Detect face orientation and visibility status
        clear_faces, poses = self.face_pose.detect_face_pose()

        # Check and handle occlusions
        clear, clear_faces = self.face_occlusion.process_occlusions(clear_faces, poses)

        # If all faces are occluded, further processing is skipped
        if(clear):
            # Get cropping limits, crop faces, and save metadata
            self._seq = self.face_cropper.get_limits_and_crop_face(clear_faces, poses)

        return self._seq