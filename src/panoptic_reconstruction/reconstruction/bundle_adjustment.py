import os
import subprocess
import numpy as np 
from typing import List, Tuple

from .projection_utils import ProjectionUtils


class BundleAdjustment:
    """
    Handles bundle adjustment operations for 3D reconstruction refinement.
    """
        
    @staticmethod
    def execute_BA() -> np.ndarray:
        """
        Executes bundle adjustment using an external executable.
        
        Returns:
            np.ndarray: Refined 3D points after bundle adjustment (N x 3 array)
        """
        # Get absolute paths to required files
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        mycams = os.path.join(root, "data/cameras_info/mycams.txt")
        mypoints = os.path.join(root, "data/cameras_info/mypoints.txt")
        fy = os.path.join(root, "data/cameras_info/fy.txt")
        exe = os.path.join(root, "cvsba-1.0.0/build/utils/cvsba_TFM")

        # Build the command to execute
        command = f"{exe} {mycams} {mypoints} {fy}"

        # Execute the bundle adjustment command
        subprocess.run(command, shell=True)
        
        # Load and return the refined points
        pts_ref = np.loadtxt(os.path.join(root, "data/cameras_info/newpoints.txt"))
        return pts_ref
		

    @staticmethod
    def reconstruct_and_BA(pts_rec: np.ndarray, sel_scams: list) -> np.ndarray:
        """
        Performs complete reconstruction pipeline with bundle adjustment refinement.
        
        Args:
            pts_rec (np.ndarray): Initial 3D points from linear reconstruction (N x 3 array)
            sel_scams (list): List of selected camera dictionaries, each containing:
                - K: Camera intrinsic matrix
                - R: Camera rotation matrix
                - t: Camera translation vector
                - distCoef: Camera distortion coefficients
                
        Returns:
            np.ndarray: Refined 3D points after bundle adjustment (N x 3 array)
        """
        # Compute reprojections using linear reconstruction
        all_reproj = ProjectionUtils.get_all_reproj(sel_scams, pts_rec)

        # Prepare and save data for bundle adjustment
        BAFormatter.save_points_for_BA(pts_rec, all_reproj)
        BAFormatter.save_cameras_info_for_BA(sel_scams)

        # Execute bundle adjustment and return refined points
        pts_ref = BundleAdjustment.execute_BA()
        return pts_ref





class BAFormatter:
    """
    A utility class for formatting 3D points and camera information for Bundle Adjustment.
    
    Formats data according to the structure expected by SBA (Sparse Bundle Adjustment):
    
    For points (mypoints.txt format):
    ---------------------------------
    x, y, z, n, id0, u0, v0, id1, u1, v1, id2, u2, v2
    Where:
    - x,y,z: 3D coordinates
    - n: Number of cameras observing the point
    - idX: Camera ID (0-based index)
    - uX,vX: 2D projection coordinates in camera idX
    
    For cameras (mycams.txt format):
    -------------------------------
    fx cx cy AR s r2 r4 t1 t2 r6 q0 qi qj qk tx ty tz
    Where:
    - fx, fy: Focal lengths
    - cx, cy: Principal points
    - AR: Aspect ratio (typically 1)
    - s: Skew (typically 0)
    - r2, r4, r6: Radial distortion coefficients
    - t1, t2: Tangential distortion coefficients
    - q0-qi-qj-qk: Rotation quaternion
    - tx-ty-tz: Translation vector
    
    The fy values are stored separately in fy.txt
    """

    @staticmethod
    def format_points_for_BA(pts_rec: np.ndarray, all_reproj: List[np.ndarray]) -> Tuple[List[List[float]], np.ndarray]:
        """
        Organizes 2D and 3D points in the correct format for Bundle Adjustment.
        
        Args:
            pts_rec: Reconstructed 3D points (Nx3 array)
            all_reproj: List of 2D reprojections for each camera (each Mx2 array)
            
        Returns:
            tuple: 
                - xyzPoints: List of 3D coordinates
                - points: Formatted data for SBA (Nx(4+3*num_cams) array)
        """
        print("\nProcessing 3D points and 2D landmarks")

        xyzPoints = []
        points = []

        for i in range(len(all_reproj[0])):
            # Start with 3D coordinates and number of cameras
            data = [pts_rec[i][0], pts_rec[i][1], pts_rec[i][2], len(all_reproj)]
            xyzPoints.append([pts_rec[i][0], pts_rec[i][1], pts_rec[i][2]])

            # Add camera ID and 2D projection for each camera
            for j in range(len(all_reproj)):
                data.extend([j, all_reproj[j][i][0], all_reproj[j][i][1]])

            # Stack the data
            if len(points) == 0:
                points = data
            else:
                points = np.vstack([points, data])
                
        return xyzPoints, points


    @staticmethod
    def save_points_for_BA(pts_rec: np.ndarray, all_reproj: List[np.ndarray]) -> None:
        """
        Saves 3D points and their 2D projections in the format required by SBA.
        
        Creates two files:
        - mypoints.txt: Contains all 3D points with their 2D projections
        - newpoints.txt: Contains just the 3D coordinates
        
        Args:
            pts_rec: Reconstructed 3D points
            all_reproj: List of 2D reprojections for each camera
        """
        # Format points for saving
        xyzPoints, points = BAFormatter.format_points_for_BA(pts_rec, all_reproj)

        # Round and format numbers
        round_points = []
        round_xyzPoints = []
        for i in range(len(points)):
            aux = [format(round(float(num), 5), ".5f").rstrip("0").rstrip(".") for num in points[i]]
            aux_xyzPoints = [format(round(float(num), 5), ".5f").rstrip("0").rstrip(".") for num in xyzPoints[i]]

            round_points.append(" ".join(aux))
            round_xyzPoints.append(" ".join(aux_xyzPoints))

        # Save to files
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        with open(os.path.join(root, "data/cameras_info/mypoints.txt"), "w") as file:
            file.write("\n".join(round_points))

        with open(os.path.join(root, "data/cameras_info/newpoints.txt"), "w") as file:
            file.write("\n".join(round_xyzPoints))

        print("-- Points saved for Bundle Adjustment --")


    @staticmethod
    def rot2quat(R: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Converts rotation matrix to quaternion representation.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            tuple: (q0, qi, qj, qk) quaternion components
        """
        q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2    
        qi = np.copysign(0.5*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]), R[2,1]-R[1,2])
        qj = np.copysign(0.5*np.sqrt(1-R[0,0]+R[1,1]-R[2,2]), R[0,2]-R[2,0])
        qk = np.copysign(0.5*np.sqrt(1-R[0,0]-R[1,1]+R[2,2]), R[1,0]-R[0,1])
        
        return q0, qi, qj, qk


    @staticmethod
    def format_cameras_info_for_BA(sel_scams: List[dict]) -> Tuple[np.ndarray, List[float]]:
        """
        Formats camera parameters for Bundle Adjustment.
        
        Args:
            sel_scams: List of selected cameras with their parameters
            
        Returns:
            tuple:
                - cameras_info: Array of formatted camera parameters
                - all_fy: List of fy focal lengths
        """
        cameras_info = []
        all_fy = []
        AR = 1  # Aspect ratio (typically 1)
        s = 0   # Skew (typically 0)
        t1 = 0  # Tangential distortion (typically 0)
        t2 = 0  # Tangential distortion (typically 0)

        for cam in sel_scams: 
            # Camera intrinsics
            fx = cam['K'][0,0]
            fy = cam['K'][1,1]  # Stored separately
            cx = cam['K'][0,2]
            cy = cam['K'][1,2]
            r2 = cam['distCoef'][0]
            r4 = cam['distCoef'][1]
            r6 = cam['distCoef'][2]

            # Camera pose (quaternion + translation)
            q0, qi, qj, qk = BAFormatter.rot2quat(cam['R'])
            tx = float(cam['t'][0])
            ty = float(cam['t'][1])
            tz = float(cam['t'][2])

            data = np.array([fx, cx, cy, AR, s, r2, r4, t1, t2, r6, q0, qi, qj, qk, tx, ty, tz])
            all_fy.append(fy)

            if len(cameras_info) == 0:
                cameras_info = data
            else:
                cameras_info = np.vstack([cameras_info, data])

        return cameras_info, all_fy


    @staticmethod
    def save_cameras_info_for_BA(sel_scams: List[dict]) -> None:
        """
        Saves camera parameters in the format required by SBA.
        
        Creates two files:
        - mycams.txt: Contains all camera parameters except fy
        - fy.txt: Contains the fy focal lengths
        
        Args:
            sel_scams: List of selected cameras with their parameters
        """
        # Get formatted camera info
        cameras_info, all_fy = BAFormatter.format_cameras_info_for_BA(sel_scams)

        # Round and format numbers
        round_cams = []
        round_fy = []
        for i in range(len(cameras_info)):
            aux = [format(round(float(c), 5), ".5f").rstrip("0").rstrip(".") for c in cameras_info[i]]
            round_fy.append(format(round(float(all_fy[i]), 5), ".5f").rstrip("0").rstrip("."))
            round_cams.append(" ".join(aux))

        # Save to files
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        with open(os.path.join(root, "data/cameras_info/mycams.txt"), "w") as file:
            file.write("\n".join(round_cams))

        with open(os.path.join(root, "data/cameras_info/fy.txt"), "w") as file:
            file.write("\n".join(round_fy))

        print("-- Camera info saved for Bundle Adjustment --")