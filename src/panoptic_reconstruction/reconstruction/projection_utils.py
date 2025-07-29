import cv2
import numpy as np

class ProjectionUtils:
    """
    Utility class for camera projection operations including:
    - Projection matrix computation
    - Coordinate transformations
    - Reprojection error calculations
    - Point cloud reprojections
    """

    @staticmethod
    def get_all_reproj(sel_scams: list, ptshom: np.ndarray) -> np.ndarray:
        """
        Projects 3D points to 2D for all selected cameras.
        
        Args:
            sel_scams: List of camera dictionaries containing:
                - 'R': Rotation matrix (3x3)
                - 't': Translation vector (3x1)
                - 'K': Intrinsic matrix (3x3)
            ptshom: 3D points to project (Nx3 or Nx4 array)
            
        Returns:
            np.ndarray: Array of reprojected 2D points for each camera (shape: [num_cams, 5023, 2])
            
        Note:
            - Converts points to homogeneous coordinates if needed
            - Normalizes by dividing by the last coordinate
        """
        all_reproj = []

        # Convert to homogeneous coordinates if needed
        if ptshom.shape == (5023, 3):
            ptshom = np.hstack((ptshom, np.ones((ptshom.shape[0], 1))))

        for i in range(len(sel_scams)):
            cam = sel_scams[i]
            # Create [R|t] matrix
            RT = np.hstack((cam['R'], cam['t'].reshape(3, 1)))
            # Compute projection matrix P = K[R|t]
            P = np.dot(cam['K'], RT)

            all_reproj.append([])    
            reproj = []

            # Project each point
            for pt in ptshom:
                reproj.append(np.array(P @ pt))

            # Convert to cartesian coordinates
            reproj = np.array(reproj).reshape(ptshom.shape[0], 3)
            reproj = reproj[:] / reproj[:, -1][:, np.newaxis]
            # Store only x,y coordinates
            all_reproj[i].append(reproj[:, 0:2]) 
            
        return np.array(all_reproj).reshape(len(sel_scams), 5023, 2)


    @staticmethod
    def get_all_points_hom(all_spredictions_arr: list) -> np.ndarray:
        """
        Convert all scaled predictions to homogeneous coordinates.
        
        Args:
            all_spredictions_arr: List of arrays with 2D predictions (shape: [N, 2])
            
        Returns:
            np.ndarray: Array of points in homogeneous coordinates (shape: [N, 3, num_points])
        """
        all_points = []
        for pred in all_spredictions_arr:
            # Convert to homogeneous coordinates (add z=1)
            pts = np.array(pred).T
            pts = np.vstack((pts, np.ones((pts.shape[1]))))
            all_points.append(pts)
            
        return np.array(all_points)
    

    @staticmethod
    def get_sel_projection_matrices(sel_scams: list) -> np.ndarray:
        """
        Compute projection matrices for selected cameras.
        
        Args:
            sel_scams: List of camera dictionaries containing:
                - 'R': Rotation matrix (3x3)
                - 't': Translation vector (3x1)
                - 'K': Intrinsic matrix (3x3)
                
        Returns:
            np.ndarray: Array of projection matrices (shape: [num_cams, 3, 4])
        """
        all_P_aux = []
        for cam in sel_scams:
            RT = np.hstack((cam['R'], cam['t'].reshape(3, 1)))
            P = np.dot(cam['K'], RT)
            all_P_aux.append(np.array(P))
            
        return np.array(all_P_aux)


    @staticmethod
    def get_sel_points_hom(sel_scams_id_aux: list, new_sel_scams_id_aux: list, 
                          points_hom_aux: np.ndarray) -> np.ndarray:
        """
        Select and reorder homogeneous points based on camera selection.
        
        Args:
            sel_scams_id_aux: Original camera IDs (list)
            new_sel_scams_id_aux: New camera IDs selection (list)
            points_hom_aux: Original homogeneous points (array)
            
        Returns:
            np.ndarray: Reordered homogeneous points array matching new camera selection
        """
        sorted_idx = []
        # Find matching indices between new and original camera lists
        for new_id in new_sel_scams_id_aux:
            for orig_idx, orig_id in enumerate(sel_scams_id_aux):
                if new_id == orig_id:
                    sorted_idx.append(orig_idx)
                    break

        new_all_points_hom = []
        for idx in sorted_idx:
            new_all_points_hom.append(points_hom_aux[idx].T)
            
        return np.array(new_all_points_hom)


    @staticmethod
    def process_reprojections(all_reproj: list, all_hd_ori: list, 
                            all_ref_origen: list, tams: list) -> tuple:
        """
        Process reprojections for scaled and non-scaled versions.
        
        Args:
            all_reproj: List of reprojected points
            all_hd_ori: List of original HD image references
            all_ref_origen: List of reference origin points
            tams: List of scaling factors
            
        Returns:
            tuple: (scaled_projected, projected) where:
                - scaled_projected: List of scaled reprojections
                - projected: List of original reprojections
        """
        scaled_projected = []
        projected = []
        all_reproj = np.array(all_reproj)

        for r, reproj in enumerate(all_reproj):  
            hd_ori = all_hd_ori[r]
            ref_origen = all_ref_origen[r]
            scaled_projected.append([])
            projected.append([])
            for pt in reproj:      
                # Apply scaling and offset
                scaled_projected[r].append([
                    (pt[0]-ref_origen['x'][hd_ori])/tams[r], 
                    (pt[1]-ref_origen['y'][hd_ori])/tams[r]
                ])
                projected[r].append([pt[0], pt[1]])
        
        return scaled_projected, projected


    @staticmethod
    def get_reproj_error(reproj_good: list, all_reproj: list) -> tuple:
        """
        Calculate reprojection error between prediction and reprojected points.
        
        Args:
            reproj_good: List of ground truth 2D points
            all_reproj: List of reprojected 2D points
            
        Returns:
            tuple: (total_error, errors_per_image) where:
                - total_error: RMS error across all points
                - errors_per_image: List of RMS errors per image
        """
        total_errors_per_image = 0
        total_points = 0
        errors_per_image = []

        for i in range(len(all_reproj)):
            # Calculate L2 norm between ground truth and reprojections
            norma_l2 = cv2.norm(np.array(reproj_good[i]), np.array(all_reproj[i]), cv2.NORM_L2)

            # Calculate per-image RMS error
            errors_per_image_aux = np.sqrt(norma_l2**2 / len(reproj_good[i]))       
            errors_per_image.append(errors_per_image_aux)
            total_errors_per_image += norma_l2**2
            total_points += len(reproj_good[i])

        # Calculate overall RMS error
        total_error = np.sqrt(total_errors_per_image / total_points)

        return total_error, errors_per_image


    @staticmethod
    def get_error_gt_sba(all_gt: list, sel_reproj_aux: list, 
                        idx_landmarks: list) -> tuple:
        """
        Calculate detailed reprojection error between ground truth and SBA results.
        
        Args:
            all_gt: List of ground truth points
            sel_reproj_aux: List of reprojected points
            idx_landmarks: List of landmark indices
            
        Returns:
            tuple: (total_error, errors_per_image, errors_per_points) where:
                - total_error: Overall RMS error
                - errors_per_image: RMS errors per image
                - errors_per_points: RMS errors per landmark point
        """
        total_errors_per_image = 0
        total_points = 0
        errors_per_image = []
        errors_per_points = np.zeros(27)
        total_idx_landmarks_points = np.zeros(27)

        for i in range(len(all_gt)):
            # Calculate coordinate differences
            dif = np.array(all_gt[i]) - np.array(sel_reproj_aux[i])
            # Calculate per-point L2 errors
            norma_l2_points = np.array([np.sqrt(d[0]**2 + d[1]**2) for d in dif])
            # Calculate per-image total error
            norma_l2 = np.sum(norma_l2_points)

            # Per-image RMS error
            errors_per_image_aux = np.sqrt(norma_l2**2 / len(dif))
            errors_per_image.append(errors_per_image_aux)

            # Accumulate for overall error
            total_errors_per_image += norma_l2**2
            total_points += len(dif)

            # Accumulate errors per landmark
            for j, idx in enumerate(idx_landmarks[i]):
                errors_per_points[idx] += norma_l2_points[j]**2
                total_idx_landmarks_points[idx] += 1
            
        # Calculate RMS errors
        errors_per_points = [np.sqrt(a / b) if b > 0 else 0 
                          for a, b in zip(errors_per_points, total_idx_landmarks_points)]
        total_error = np.sqrt(total_errors_per_image / total_points)
        
        return total_error, errors_per_image, errors_per_points