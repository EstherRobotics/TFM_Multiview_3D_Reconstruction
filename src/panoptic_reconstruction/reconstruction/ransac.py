import random
import numpy as np 
from typing import List, Tuple, Dict, Any

from .projection_utils import ProjectionUtils
from .linear_reconstruction import LinearReconstruction
from .bundle_adjustment import BundleAdjustment
from panoptic_reconstruction.prediction import LandmarksSelector3D


class Ransac:
    """
    Implements RANSAC-based camera selection algorithms for 3D reconstruction.
    
    Provides methods for:
    - Robust camera selection using RANSAC
    - Camera pair evaluation
    - Optimal camera combination selection
    - Reconstruction error computation
    """
    
    def __init__(self, manage_cams: Any) -> None:
        """
        Initialize Ransac with camera manager.
        
        Args:
            manage_cams: Camera manager instance providing camera selection utilities
        """
        self._manage_cams = manage_cams

    def select_cameras_with_ransac(self, all_points_hom: np.ndarray, 
                                 reproj_panoptic: List[np.ndarray],
                                 sel_landmarks: List[int], 
                                 sel_scams: List[Dict[str, np.ndarray]],
                                 sel_scams_id: List[int], 
                                 cameras: Dict,
                                 iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main RANSAC implementation for camera selection using panoptic landmarks.
        
        Args:
            all_points_hom: Homogeneous coordinates of all points (Nx3 array)
            reproj_panoptic: List of panoptic landmark reprojections
            sel_landmarks: Selected landmark indices
            sel_scams: List of selected camera dictionaries
            sel_scams_id: List of selected camera IDs
            cameras: Dictionary of all camera parameters
            iterations: Number of RANSAC iterations
            
        Returns:
            tuple:
                - Array of camera indices sorted by inlier count (best first)
                - Array counting inlier occurrences for each camera
        """
        inliers_cameras = np.zeros(len(sel_scams), dtype=int)
        final_error = self._initialize_with_all_cameras(all_points_hom, reproj_panoptic, 
                                                     sel_landmarks, sel_scams)

        for _ in range(iterations):
            idx_selected = self._manage_cams.select_random_camera_pair(len(sel_scams))
            total_error = self._evaluate_cameras(all_points_hom, reproj_panoptic,
                                               sel_landmarks, sel_scams, idx_selected)
            
            if total_error < final_error:
                inliers_cameras = self._update_inliers(inliers_cameras, idx_selected)
                final_error = total_error

        return self._get_final_combination(inliers_cameras), inliers_cameras


    def select_cameras_with_ransac_from_predictions(self, all_points_hom: np.ndarray,
                                                  reproj_pred: List[np.ndarray],
                                                  sel_landmarks: List[int],
                                                  sel_scams: List[Dict[str, np.ndarray]], 
                                                  sel_scams_id: List[int],
                                                  cameras: Dict,
                                                  iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        RANSAC variant using predicted landmarks instead of panoptic points.
        
        Args:
            all_points_hom: Homogeneous coordinates of all points (Nx3 array)
            reproj_pred: List of predicted landmark reprojections
            sel_landmarks: Selected landmark indices
            sel_scams: List of selected camera dictionaries
            sel_scams_id: List of selected camera IDs
            cameras: Dictionary of all camera parameters  
            iterations: Number of RANSAC iterations
            
        Returns:
            tuple:
                - Array of camera indices sorted by inlier count (best first)
                - Array counting inlier occurrences for each camera
        """
        inliers_cameras = np.zeros(len(sel_scams), dtype=int)
        final_error = self._initialize_with_all_cameras(all_points_hom, reproj_pred,
                                                    sel_landmarks, sel_scams)

        for _ in range(iterations):
            idx_selected = self._manage_cams.select_random_camera_pair(len(sel_scams))
            total_error = self._evaluate_cameras_from_predictions(all_points_hom, reproj_pred,
                                                               sel_scams, idx_selected)
            
            if total_error < final_error:
                inliers_cameras = self._update_inliers(inliers_cameras, idx_selected)
                final_error = total_error

        return self._get_final_combination(inliers_cameras), inliers_cameras


    def evaluate_cameras_per_pairwise_error(self, all_points_hom: np.ndarray,
                                         reproj_panoptic: List[np.ndarray],
                                         sel_landmarks: List[int],
                                         sel_scams: List[Dict[str, np.ndarray]],
                                         sel_scams_id: List[int],
                                         cameras: Dict,
                                         iterations: int) -> np.ndarray:
        """
        Evaluates all camera pairs and returns best combination based on mean error.
        
        Args:
            all_points_hom: Homogeneous coordinates of all points (Nx3 array)
            reproj_panoptic: List of panoptic landmark reprojections
            sel_landmarks: Selected landmark indices
            sel_scams: List of selected camera dictionaries
            sel_scams_id: List of selected camera IDs
            cameras: Dictionary of all camera parameters
            iterations: Number of evaluation iterations
            
        Returns:
            Array of camera indices sorted by mean error (best first)
        """
        mean_errors = []
        
        for i in range(len(sel_scams)):
            pair_errors = []
            for j in range(len(sel_scams)):
                if i != j:
                    error = self._evaluate_cameras(all_points_hom, reproj_panoptic,
                                                 sel_landmarks, sel_scams, [i,j])
                    pair_errors.append(error)
            mean_errors.append(np.mean(pair_errors))

        return np.argsort(mean_errors)


    def _initialize_with_all_cameras(self, all_points_hom: np.ndarray,
                                   reference_points: List[np.ndarray],
                                   sel_landmarks: List[int],
                                   sel_scams: List[Dict[str, np.ndarray]]) -> float:
        """
        Initializes RANSAC by evaluating reconstruction using all cameras.
        
        Args:
            all_points_hom: Homogeneous coordinates of all points
            reference_points: Reference reprojections (panoptic or predicted)
            sel_landmarks: Selected landmark indices
            sel_scams: List of selected camera dictionaries
            
        Returns:
            Initial reconstruction error using all cameras
        """
        idx_selected = list(range(len(sel_scams)))
        return self._evaluate_cameras(all_points_hom, reference_points,
                                    sel_landmarks, sel_scams, idx_selected)


    def _evaluate_cameras(self, all_points_hom: np.ndarray,
                        reference_points: List[np.ndarray],
                        sel_landmarks: List[int],
                        sel_scams: List[Dict[str, np.ndarray]],
                        idx_selected: List[int]) -> float:
        """
        Evaluates camera combination by performing reconstruction and computing error.
        
        Args:
            all_points_hom: Homogeneous coordinates of all points
            reference_points: Reference reprojections (panoptic landmarks)
            sel_landmarks: Selected landmark indices
            sel_scams: List of selected camera dictionaries
            idx_selected: Indices of cameras to evaluate
            
        Returns:
            Total reprojection error for the camera combination
        """
        resel_scams = self._manage_cams.select_cameras_from_set(sel_scams, idx_selected)
        projection_matrices = ProjectionUtils.get_sel_projection_matrices(resel_scams)
        points_hom = self._get_selected_points(all_points_hom, idx_selected)

        pts_rec = LinearReconstruction.reconstruct_all_views(projection_matrices, points_hom, len(idx_selected))
        pts_ref = BundleAdjustment.reconstruct_and_BA(pts_rec, resel_scams)
        all_reproj = ProjectionUtils.get_all_reproj(sel_scams, pts_ref)

        sel_27_reproj = LandmarksSelector3D.select_27_landmarks(all_reproj, sel_landmarks)
        total_error, _ = ProjectionUtils.get_reproj_error(reference_points, sel_27_reproj)
        return total_error


    def _evaluate_cameras_from_predictions(self, all_points_hom: np.ndarray,
                                         reproj_pred: List[np.ndarray],
                                         sel_scams: List[Dict[str, np.ndarray]],
                                         idx_selected: List[int]) -> float:
        """
        Evaluates camera combination using prediction-based reprojection error.
        
        Args:
            all_points_hom: Homogeneous coordinates of all points
            reproj_pred: Predicted landmark reprojections
            sel_scams: List of selected camera dictionaries
            idx_selected: Indices of cameras to evaluate
            
        Returns:
            Total reprojection error for the camera combination
        """
        resel_scams = self._manage_cams.select_cameras_from_set(sel_scams, idx_selected)
        projection_matrices = ProjectionUtils.get_sel_projection_matrices(resel_scams)
        points_hom = self._get_selected_points(all_points_hom, idx_selected)

        pts_rec = LinearReconstruction.reconstruct_all_views(projection_matrices, points_hom, len(idx_selected))
        pts_ref = BundleAdjustment.reconstruct_and_BA(pts_rec, resel_scams)
        all_reproj = ProjectionUtils.get_all_reproj(sel_scams, pts_ref)

        total_error, _ = ProjectionUtils.get_reproj_error(reproj_pred, all_reproj)
        return total_error


    def _get_selected_points(self, all_points_hom: np.ndarray,
                            idx_selected: List[int]) -> np.ndarray:
        """
        Gets homogeneous points for selected cameras.
        
        Args:
            all_points_hom: All points in homogeneous coordinates
            idx_selected: Indices of selected cameras
            
        Returns:
            Points for selected cameras only
        """
        return np.array([all_points_hom[idx].T for idx in idx_selected])


    def _update_inliers(self, inliers_cameras: np.ndarray,
                       idx_selected: List[int]) -> List[int]:
        """
        Updates inlier counts for selected cameras.
        
        Args:
            inliers_cameras: Current inlier counts
            idx_selected: Indices of cameras to increment
            
        Returns:
            Updated inlier counts
        """
        return [inliers_cameras[k] + 1 if k in idx_selected else inliers_cameras[k] 
                for k in range(len(inliers_cameras))]


    def _get_final_combination(self, inliers_cameras: np.ndarray) -> np.ndarray:
        """
        Determines final camera combination by sorting cameras by inlier count.
        
        Args:
            inliers_cameras: Array of inlier counts per camera
            
        Returns:
            Array of camera indices sorted by inlier count (descending)
        """
        return np.argsort(inliers_cameras)[::-1]