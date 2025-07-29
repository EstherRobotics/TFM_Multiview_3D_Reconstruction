import numpy as np

class LandmarksSelector3D:
    """
    A utility class for selecting and processing facial landmarks.
    
    Provides static methods for:
    - Extracting landmark indices from pre-defined files
    - Selecting specific subsets of landmarks
    - Filtering visible landmarks based on ground truth indices
    """
    
    @staticmethod
    def extract_idx_landmarks_from_npy() -> np.ndarray:
        """
        Extracts and combines facial landmark indices from multiple NPY files.
        
        Returns:
            np.ndarray: Array containing selected landmark indices concatenated
        """
        # Define paths to landmark files
        base_path = '/home/esther/Escritorio/DAD_3DHeads/model_training/model/static/face_keypoints/keypoints_191/'
        paths = {
            'brows': base_path + 'brows.npy',
            'contour': base_path + 'contour.npy',
            'eyes': base_path + 'eyes.npy',
            'forehead': base_path + 'forehead.npy',
            'lips': base_path + 'lips.npy',
            'nose': base_path + 'nose.npy'
        }

        # Load all landmark indices
        landmarks = {}
        for key, path in paths.items():
            landmarks[key] = np.load(path, allow_pickle=True).item()

        # Combine all landmarks
        all_landmarks = []
        for index in landmarks.values():
            for val in index.values():
                all_landmarks.extend(val)

        # Define specific landmark groups
        brow_landmarks = [3684, 1983, 3851, 570]  # Left and right brow tops
        eye_landmarks = [
            2355, 2267, 2437,  # Left eye bottom
            2381, 2493, 3619,  # Left eye top
            827, 814, 1175,    # Right eye bottom
            1023, 1342, 3827   # Right eye top
        ]
        nose_landmarks = [3555, 3501, 2751, 3515, 1623]  # Bridge, wings, philtrum
        lip_landmarks = [3509, 3503, 3533, 2828, 3546, 1711]  # Upper and lower lips

        # Combine selected landmarks (total 27 points)
        sel_landmarks = np.concatenate([
            brow_landmarks,
            eye_landmarks,
            nose_landmarks,
            lip_landmarks
        ])

        return sel_landmarks


    @staticmethod
    def select_27_landmarks(reproj_arr: list, sel_landmarks: np.ndarray) -> list:
        """
        Selects specific 27 landmarks from reprojection arrays.
        
        Args:
            reproj_arr: List of reprojection arrays (each containing all landmarks)
            sel_landmarks: Array of indices for the 27 landmarks to select
            
        Returns:
            list: List of arrays containing only the selected 27 landmarks
        """
        sel_reproj_aux = []
        
        for i, reproj in enumerate(reproj_arr):
            sel_reproj_aux.append([])
            for idx in sel_landmarks:
                sel_reproj_aux[i].append(reproj[idx])
                
        return sel_reproj_aux


    @staticmethod
    def select_visible_27_landmarks(all_reproj_aux: list, idx_landmarks: list) -> list:
        """
        Filters visible landmarks based on ground truth indices.
        
        Args:
            all_reproj_aux: List of all reprojected landmarks
            idx_landmarks: List of indices indicating which landmarks are visible
            
        Returns:
            list: List containing only the visible landmarks from the original 27
        """
        sel_reproj_aux = []
        for i in range(len(idx_landmarks)):
            sel_reproj_aux.append([])
            for idx in idx_landmarks[i]:
                sel_reproj_aux[i].append(list(all_reproj_aux[i][idx])) 
        
        return sel_reproj_aux
