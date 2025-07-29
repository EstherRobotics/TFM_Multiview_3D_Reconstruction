import os
import numpy as np
import numpy.linalg as npla


class LinearReconstruction:
    """
    Performs linear triangulation for 3D point reconstruction from multiple 2D views.
    """

    @staticmethod
    def reconstruct(points1: np.ndarray, points2: np.ndarray, 
                   P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Reconstructs 3D points from 2D correspondences in two views using linear triangulation.
        
        Args:
            points1 (np.ndarray): Homogeneous 2D points in first view (3 x N array)
            points2 (np.ndarray): Homogeneous 2D points in second view (3 x N array)
            P1 (np.ndarray): Projection matrix for first camera (3 x 4)
            P2 (np.ndarray): Projection matrix for second camera (3 x 4)
            
        Returns:
            np.ndarray: Reconstructed 3D points in homogeneous coordinates (4 x N array)
        """
        # Validate input shapes
        assert (points1.shape == points2.shape), "reconstruct: different number of points!"
        assert (points1.shape[0] == 3), "reconstruct: not homogeneous points!"

        # Convert homogeneous to cartesian coordinates
        points1 = points1/points1[2,:]
        points2 = points2/points2[2,:]

        # Initialize output array for 3D points
        points3D = np.zeros((4, points1.shape[1]))

        # Reconstruct each point individually
        for i in range(points1.shape[1]):
            # Build coefficient matrix for the i-th point
            A = np.zeros((4, 4))
            A[0,:] = P1[0,:] - P1[2,:]*points1[0,i]  # x1*P1[2] - P1[0]
            A[1,:] = P1[1,:] - P1[2,:]*points1[1,i]  # y1*P1[2] - P1[1]              
            A[2,:] = P2[0,:] - P2[2,:]*points2[0,i]  # x2*P2[2] - P2[0]              
            A[3,:] = P2[1,:] - P2[2,:]*points2[1,i]  # y2*P2[2] - P2[1]          

            # Solve using SVD (smallest singular vector gives solution)
            _, _, Vt = npla.svd(A)
            points3D[:, i] = Vt[3,:]
                    
        return points3D


    @staticmethod
    def reconstruct_all_views(all_P: list, all_pts_hom: list, 
                            rec_number: int) -> np.ndarray:
        """
        Reconstructs 3D points from multiple views using linear least squares.
        
        Args:
            all_P (list): List of projection matrices (each 3 x 4 numpy array)
            all_pts_hom (list): List of point correspondences across views. 
                              Each element is a list of homogeneous 2D points (3-vectors)
            rec_number (int): Number of views to use for reconstruction
                             
        Returns:
            np.ndarray: Reconstructed 3D points in cartesian coordinates (N x 3 array)
        """
        pts_3d = []
        total_pts = len(all_pts_hom[0])

        # Reconstruct each 3D point from its 2D observations
        for i in range(total_pts):
            # Build coefficient matrix for current point
            A = np.zeros((2*rec_number, 4))
            
            # Add constraints from each view
            n = 0
            for j in range(rec_number):
                A[n] = all_pts_hom[j][i][0] * all_P[j][2] - all_P[j][0]  # x*P[2] - P[0]
                A[n+1] = all_pts_hom[j][i][1] * all_P[j][2] - all_P[j][1]  # y*P[2] - P[1]
                n += 2

            # Solve using SVD (smallest singular vector gives solution)
            _, _, V = np.linalg.svd(A)
            pt_aux = V[-1, :3] / V[-1, 3]  # Convert from homogeneous to cartesian
            pts_3d.append(pt_aux)

        return np.array(pts_3d)