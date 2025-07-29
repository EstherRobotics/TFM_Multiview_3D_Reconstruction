import numpy as np
import pandas as pd
import os


class OriginReferenceHandler:
    """Handles saving and retrieving origin reference data for image sequences."""

    @staticmethod
    def save_json_origin(save_origin_dir: str, seq: dict, last_camera: bool = False) -> dict:
        """
        Save origin reference data (coordinates and file names) to CSV if the camera has changed 
        or it's the last camera being processed.

        Args:
            save_origin_dir (str): Directory where the CSV file will be saved.
            seq (dict): Dictionary containing sequence data with:
                - cam_num (int): Current camera number.
                - prev_cam_num (int): Previous camera number.
                - output_files (List[str]): List of cropped image file paths.
                - x (List[float]): List of x coordinates for cropped regions.
                - y (List[float]): List of y coordinates for cropped regions.
            last_camera (bool, optional): If True, forces saving even if the camera hasn’t changed.

        Returns:
            dict: Updated sequence dictionary with cleared tracking lists after saving.
        """
        # Save data if the camera changed or it's the last one
        if seq['cam_num'] != seq['prev_cam_num'] or last_camera:
            # Ensure lengths of tracked data match
            if not (len(seq['output_files']) == len(seq['x']) == len(seq['y'])):
                print(f"ERROR: Length mismatch - Files: {len(seq['output_files'])} X: {len(seq['x'])} Y: {len(seq['y'])}")
                raise ValueError("Mismatch in lengths of output_files, x, and y lists")

            if seq['output_files']:
                #print("-- Previous origin data saved --")
                data = {
                    'file': seq['output_files'],
                    'x': np.array(seq['x']),
                    'y': np.array(seq['y'])
                }

                df = pd.DataFrame(data)
                cam_str = f"{seq['prev_cam_num']:02d}"  # Zero-padded camera number

                os.makedirs(save_origin_dir, exist_ok=True)
                df.to_csv(os.path.join(save_origin_dir, f'origin_00_{cam_str}.csv'), index=False)

            # Reset the tracking lists after saving
            seq['prev_cam_num'] = seq['cam_num']
            seq['output_files'], seq['x'], seq['y'] = [], [], []

        return seq


    @staticmethod
    def get_ref_origin(origin_path: str, cropped_img_name: str) -> tuple:
        """
        Retrieve origin reference data for a given cropped image from a CSV file.

        Args:
            origin_path (str): Path to the CSV file containing origin reference data.
            cropped_img_name (str): Name of the cropped image to search for.

        Returns:
            tuple: (found, ref_origin_df, index)
                - found (bool): True if the image is found in the reference data.
                - ref_origin_df (pd.DataFrame): DataFrame containing all reference data.
                - index (int): Row index where the image was found.
        """
        ref_origin = pd.read_csv(origin_path)
        index = 0
        found = False

        for i, file_path in enumerate(ref_origin['file']):
            if file_path.split('/')[-1] == cropped_img_name:
                found = True
                index = i
                break

        return found, ref_origin, index
