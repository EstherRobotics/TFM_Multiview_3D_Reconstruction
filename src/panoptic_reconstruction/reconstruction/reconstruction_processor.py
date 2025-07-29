
import numpy as np
from typing import Dict, Any, Tuple, Union, List

from .bundle_adjustment import BundleAdjustment
from .ransac import Ransac 
from .projection_utils import ProjectionUtils
from .linear_reconstruction import LinearReconstruction
from panoptic_reconstruction.utils import ResultSaver
from panoptic_reconstruction.panoptic_points import PanopticLandmarks
from panoptic_reconstruction.prediction import ScaledLandmarksProcessor, LandmarksSelector3D


class ReconstructProcessor:
    """
    Main processor class for 3D reconstruction pipeline.
    
    Handles the complete reconstruction workflow including:
    - Data preparation and organization
    - Camera selection using RANSAC
    - Linear 3D reconstruction
    - Bundle adjustment refinement
    - Result saving
    """
    
    def __init__(self, manage_paths: Any, manage_cams: Any, paths: Dict, cameras: Dict) -> None:
        """
        Initialize the ReconstructProcessor with path and camera managers.
        
        Args:
            manage_paths: Path management utility instance
            manage_cams: Camera management utility instance
            paths: Dictionary containing path configurations
            cameras: Dictionary containing camera configurations
        """
        self._manage_paths = manage_paths
        self._manage_cams = manage_cams
        self._cameras = cameras
        self._paths = paths
        self._ransac = Ransac(manage_cams)


    def process_reconstruction(self, seq: Dict) -> None:
        """
        Main entry point for processing reconstruction of a sequence.
        
        Args:
            seq: Dictionary containing sequence data with:
                - 'all_ppl_idx': List of person indices in the scene
                - Other sequence metadata
        """
        # Only process if people are detected in the scene
        if len(seq['all_ppl_idx']) > 0:
            print("\n\nPROCESSING PREDICTIONS FOR RECONSTRUCTION")
            print("All people idx to process:",seq['all_ppl_idx'])
            slandmarks = ScaledLandmarksProcessor(self._paths)

            # Process each person in the scene
            for ppl_idx in seq['all_ppl_idx']:
                print("Current person idx:", ppl_idx)

                # Get sorted cropped face images
                all_cropped_imgs = self._manage_paths.sort_cropped_imgs_paths(ppl_idx)

                # Need at least 2 views for reconstruction
                if len(all_cropped_imgs) >= 2:
                    # Get landmark predictions
                    all_slandmarks, all_ref_origen, all_hd_ori = slandmarks.get_all_slandmarks(all_cropped_imgs)
                    # Perform full reconstruction pipeline
                    self.reconstruct_scene(all_slandmarks, seq, ppl_idx, all_ref_origen, all_hd_ori)


    def _prepare_reconstruction_data(self, pred_landmarks: List[np.array], 
                                   sel_scams_id: List[int], sel_scams: List[Dict],
                                   hd_idx: int, ppl_idx: int, 
                                   panoptic_points_path: str) -> Tuple[np.array, np.array, List[int]]:
        """
        Prepares data needed for reconstruction pipeline.
        
        Args:
            pred_landmarks: List of predicted 2D landmarks
            sel_scams_id: List of selected camera IDs
            sel_scams: List of selected camera dictionaries
            hd_idx: HD frame index
            ppl_idx: Person index
            panoptic_points_path: Path to panoptic dataset
            
        Returns:
            Tuple containing:
                - all_points_hom: Homogeneous coordinates of predicted points
                - reproj_panoptic: Panoptic landmarks reprojections
                - sel_landmarks: Selected landmark indices
        """
        # Convert predictions to array format
        all_spredictions_arr = [np.array(spred).reshape(5023,2) for spred in pred_landmarks]
        # Convert to homogeneous coordinates
        all_points_hom = ProjectionUtils.get_all_points_hom(all_spredictions_arr)
        
        # Load panoptic ground truth points
        reproj_panoptic = PanopticLandmarks.get_facial_panoptic_landmarks(
            panoptic_points_path, hd_idx, ppl_idx, sel_scams
        )
        # Get landmark indices to use
        sel_landmarks = LandmarksSelector3D.extract_idx_landmarks_from_npy()
        
        return all_points_hom, reproj_panoptic, sel_landmarks


    def _perform_ransac_selection(self, all_points_hom: np.array, 
                                reproj_panoptic: np.array, 
                                sel_landmarks: List[int], 
                                sel_scams: List[Dict], 
                                sel_scams_id: List[int]) -> Tuple[List[int], List[int]]:
        """
        Performs RANSAC-based camera selection for robust reconstruction.
        
        Args:
            all_points_hom: Homogeneous coordinates of predicted points
            reproj_panoptic: Panoptic landmarks reprojections  
            sel_landmarks: Selected landmark indices
            sel_scams: List of selected camera dictionaries
            sel_scams_id: List of selected camera IDs
            
        Returns:
            Tuple containing:
                - final_combination: Selected camera indices
                - inliers_cameras: Inlier camera indices
        """
        print("Performing RANSAC to select optimal cameras")
        it = 50  # Number of RANSAC iterations
        final_combination, inliers_cameras = self._ransac.select_cameras_with_ransac(
            all_points_hom, reproj_panoptic, sel_landmarks, 
            sel_scams, sel_scams_id, self._cameras, it
        )
        print("RANSAC selection finished")
        return final_combination, inliers_cameras


    def _prepare_selected_data(self, sel_scams_id: List[int], 
                             final_combination: List[int], 
                             num_cams: int, 
                             all_points_hom: np.array) -> Tuple[List[int], List[Dict], np.array, np.array]:
        """
        Prepares the selected cameras and corresponding points for reconstruction.
        
        Args:
            sel_scams_id: Original camera IDs
            final_combination: Selected camera indices from RANSAC
            num_cams: Number of cameras to use
            all_points_hom: All points in homogeneous coordinates
            
        Returns:
            Tuple containing:
                - inlier_sel_scams_id: Selected camera IDs
                - inlier_sel_scams: Selected camera dictionaries
                - inlier_all_points_hom: Selected points in homogeneous coordinates
                - inlier_all_P: Selected projection matrices
        """
        # Select camera IDs
        all_sel_scams_id = self._manage_cams.select_cameras_id(sel_scams_id, final_combination)
        inlier_sel_scams_id = all_sel_scams_id[0:num_cams]
        
        # Select camera parameters
        all_sel_scams = self._manage_cams.select_cameras(all_sel_scams_id)
        inlier_sel_scams = all_sel_scams[0:num_cams]
        
        # Prepare corresponding points and projection matrices
        inlier_all_points_hom = ProjectionUtils.get_sel_points_hom(
            sel_scams_id, all_sel_scams_id, all_points_hom
        )
        inlier_all_P = ProjectionUtils.get_sel_projection_matrices(all_sel_scams)
        
        return inlier_sel_scams_id, inlier_sel_scams, inlier_all_points_hom, inlier_all_P


    def _save_reconstruction_results(self, sel_scams: List[Dict], 
                                   pts_ref: np.array, 
                                   all_hd_ori: List, 
                                   all_ref_origen: List,
                                   tams: List, 
                                   output_paths: Dict, 
                                   hd_idx: int, 
                                   ppl_idx: int, 
                                   num_cams: int, 
                                   sel_scams_id: List[int], 
                                   inliers_cameras: List[int], 
                                   inlier_sel_scams_id: List[int]) -> None:
        """
        Handles saving of reconstruction results.
        
        Args:
            sel_scams: Selected cameras
            pts_ref: Refined 3D points
            all_hd_ori: Original HD image references
            all_ref_origen: Reference origin points
            tams: Scaling factors
            output_paths: Dictionary of output paths
            hd_idx: HD frame index
            ppl_idx: Person index
            num_cams: Number of cameras used
            sel_scams_id: Original camera IDs
            inliers_cameras: Inlier camera indices
            inlier_sel_scams_id: Selected inlier camera IDs
        """
        # Compute reprojections
        all_reproj = ProjectionUtils.get_all_reproj(sel_scams, pts_ref)
        scaled_projected, projected = ProjectionUtils.process_reprojections(
            all_reproj, all_hd_ori, all_ref_origen, tams
        )

        # Initialize result saver and save
        result_saver = ResultSaver(
            output_paths, hd_idx, ppl_idx, num_cams, pts_ref,
            sel_scams_id, inliers_cameras, inlier_sel_scams_id, inlier_sel_scams_id
        )
        result_saver.save_results(scaled_projected, projected)


    def reconstruct_scene(self, pred_landmarks: List[np.array], 
                        seq: Dict, 
                        ppl_idx: int, 
                        all_ref_origen: List, 
                        all_hd_ori: List) -> None:
        """
        Main reconstruction pipeline for a scene with one individual.
        
        Args:
            pred_landmarks: List of predicted 2D landmarks
            seq: Dictionary containing sequence data
            ppl_idx: Person index to reconstruct
            all_ref_origen: List of reference origins
            all_hd_ori: List of original HD references
        """
        print(f"\n\n>> Processing {seq['hd_idx']}")
        
        # Extract sequence data
        seq_name = seq['name']
        sel_scams = seq['sel_scams'][ppl_idx]
        sel_scams_id = seq['sel_scams_id'][ppl_idx]
        tams = seq['all_tams'][ppl_idx]
        hd_idx = seq['hd_idx']
        
        # Setup output paths
        output_paths, panoptic_points_path = self._manage_paths.setup_output_paths(seq_name)
        
        # 1. Data preparation
        all_points_hom, reproj_panoptic, sel_landmarks = self._prepare_reconstruction_data(
            pred_landmarks, sel_scams_id, sel_scams, hd_idx, ppl_idx, panoptic_points_path
        )

        # 2. RANSAC camera selection
        final_combination, inliers_cameras = self._perform_ransac_selection(
            all_points_hom, reproj_panoptic, sel_landmarks, sel_scams, sel_scams_id
        )
        
        # 3. Prepare selected data
        num_cams = self._manage_cams.camera_number_for_reconstruction(inliers_cameras)
        inlier_sel_scams_id, inlier_sel_scams, inlier_all_points_hom, inlier_all_P = self._prepare_selected_data(
            sel_scams_id, final_combination, num_cams, all_points_hom
        )

        # 4. Reconstruction pipeline
        pts_rec = LinearReconstruction.reconstruct_all_views(inlier_all_P, inlier_all_points_hom, num_cams)
        pts_ref = BundleAdjustment.reconstruct_and_BA(pts_rec, inlier_sel_scams)

        # 5. Save results
        self._save_reconstruction_results(
            sel_scams, pts_ref, all_hd_ori, all_ref_origen, tams,
            output_paths, hd_idx, ppl_idx, num_cams, sel_scams_id,
            inliers_cameras, inlier_sel_scams_id
        )