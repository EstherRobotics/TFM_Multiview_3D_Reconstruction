import os
import json
import numpy as np 


class ResultSaver:
    """
    Handles saving of 3D reconstruction results and reprojected points to various output formats.
    """
    
    def __init__(self, output_paths: dict, hd_idx: int, ppl_idx: int, num_cams: int, 
                 pts_ref: np.ndarray, sel_scams_id: list, inliers_cameras: list, 
                 all_sel_scams_id: list, new_sel_scams_id: list) -> None:
        """
        Initialize the ResultSaver with output paths and reconstruction data.
        
        Args:
            output_paths (dict): Dictionary containing output paths with keys:
                - 'rec': Path for 3D reconstruction results
                - 'rep': Path for scaled reprojection results
                - 'nsrep': Base path for non-scaled reprojection results
            hd_idx (int): HD frame index/timestamp
            ppl_idx (int): Person index to identify the subject
            num_cams (int): Number of cameras used in reconstruction
            pts_ref (np.ndarray): 3D reference points (N x 3 array)
            sel_scams_id (list): List of selected camera IDs
            inliers_cameras (list): List of inlier camera indices
            all_sel_scams_id (list): List of all selected camera IDs (before filtering)
            new_sel_scams_id (list): List of new selected camera IDs (after filtering)
        """
        self._output_rec_path = output_paths['rec']
        self._output_rep_path = output_paths['rep']
        self._output_nsrep_path = os.path.join(output_paths['nsrep'], 'ns')

        self._hd_idx = hd_idx
        self._ppl_idx = ppl_idx
        self._num_cams = num_cams
        self._pts_ref = pts_ref

        # Prepare metadata dictionary for JSON output
        self._data = {
            'cams': sel_scams_id,
            'inliers_cameras': np.array(inliers_cameras).tolist(),
            'rec_all': np.array(all_sel_scams_id).tolist(),
            'rec': np.array(new_sel_scams_id).tolist()
        }
        

    def _save_3d_reconstruction(self) -> None: 
        """
        Saves 3D reconstruction points to a text file.
        """
        output_path = os.path.join(self._output_rec_path, f"annotations_{self._hd_idx}_{self._ppl_idx}_{self._num_cams}.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, self._pts_ref, fmt='%.5f')
        print("\n3D reconstruction annotations saved in ", output_path)


    def _save_reprojections(self, path: str, proj: np.ndarray) -> None:
        """
        Saves reprojected points (either scaled or non-scaled) to JSON file.
        
        Args:
            path (str): Base output directory path
            proj (np.ndarray): Projected points to save (N x 2 array)
        """
        self._data['reproj'] = np.array(proj).tolist()
        output_path = f"{path}reprojections_{self._hd_idx}_{self._ppl_idx}_{self._num_cams}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as file:
            json.dump(self._data, file) 


    def save_results(self, scaled_projected: np.ndarray, projected: np.ndarray) -> None:
        """
        Main method to save all reconstruction results.
        
        Saves three types of results:
            1. 3D reconstruction points (text format)
            2. Scaled reprojections (JSON format)
            3. Non-scaled reprojections (JSON format)
            
        Args:
            scaled_projected (np.ndarray): Scaled projected 2D points
            projected (np.ndarray): Original (non-scaled) projected 2D points
        """
        # Save 3D reconstruction points
        self._save_3d_reconstruction()
        
        # Save scaled reprojection points
        self._save_reprojections(self._output_rep_path, scaled_projected)
        print("Scaled reprojections saved in ", self._output_rep_path)    

        # Save non-scaled reprojection points
        self._save_reprojections(self._output_nsrep_path, projected)
        print("Reprojections saved in ", self._output_nsrep_path)