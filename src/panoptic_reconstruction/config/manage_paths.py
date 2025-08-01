import os
import cv2
import sys
import yaml
import shutil
import numpy as np
from typing import Dict, List, Tuple


class ManagePaths:
    """
    Manages and organizes files and directory paths for processing.
    """

    def __init__(self) -> None:
        """
        Sets up the configuration file path and appends the parent directory to the system path.
        """
        self._root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._config_path = os.path.join(self._root_dir, 'config/config.yaml')
        self._config = self._load_config()
        self._paths: Dict[str, str] = {}
        self._paths['data_path'] = os.path.join(self._root_dir, 'data')
        self._paths['save_path'] = os.path.join(self._paths['data_path'], self._config["paths"]["save_cropped_imgs"])


    def _load_config(self) -> dict:
        """
        Loads the configuration from the YAML file.

        Returns:
            dict: Parsed configuration data.
        """
        with open(self._config_path, "r") as f:
            self._config = yaml.safe_load(f)
        return self._config


    def get_sequence_names(self) -> List[str]:
        """
        Returns a list with the sequence names.

        Returns:
            List[str]: Sequence names.
        """
        return self._config["seq_names"]


    def get_df_path(self, file_method: str) -> str:
        """
        Returns the annotation file path based on the specified method.

        Args:
            file_method (str): One of "train", "val", or "test".

        Returns:
            str: Path to the corresponding annotation file.
        """
        if file_method == 'train':
            ann_file = os.path.join(self._paths['data_path'], 'files/panoptic_ann_train.txt')
        elif file_method == 'val':
            ann_file = os.path.join(self._paths['data_path'], 'files/panoptic_ann_valid.txt')
        elif file_method == 'test':
            ann_file = os.path.join(self._paths['data_path'], 'files/panoptic_ann_test.txt')
        else:
            raise ValueError(f"Invalid file_method: {file_method}")
        return ann_file


    def get_setup_paths(self) -> Dict[str, str]:
        """
        Returns the basic paths (base, data, cropped_imgs_save) required by the project.
        """
        return self._paths


    def get_general_paths(self, seq_name: str) -> Dict[str, str]:
        """
        Generates the necessary file paths for the given sequence.

        Args:
            seq_name (str): The name of the sequence.

        Returns:
            dict: A dictionary containing the generated paths.
        """
        data_seq_path = os.path.join(self._paths['data_path'], 'sequences', seq_name)
        save_seq_path = os.path.join(self._paths['save_path'], seq_name)

        self._paths['data_seq_path'] = data_seq_path
        self._paths['save_seq_path'] = save_seq_path
        self._paths['hdimgs_path'] = os.path.join(data_seq_path, 'hdImgs')
        self._paths['save_seq_images_folder'] = os.path.join(save_seq_path, 'images')
        self._paths['save_origin_dir'] = os.path.join(save_seq_path, 'origin')
        self._paths['save_cropped_imgs_dir'] = os.path.join(save_seq_path, 'cropped_imgs')

        return self._paths


    def add_specific_paths(self, seq_name: str) -> Dict[str, str]:
        """
        Generates paths for 3D pose, face, hand data, and images related to the specified sequence.

        Args:
            seq_name (str): The name of the sequence.

        Returns:
            dict: Updated dictionary with additional paths.
        """
        self._paths['hd_skel_json_path'] = os.path.join(self._paths['data_seq_path'], 'hdPose3d_stage1_coco19')
        self._paths['hd_face_json_path'] = os.path.join(self._paths['data_seq_path'], 'hdFace3d')
        self._paths['hd_hand_json_path'] = os.path.join(self._paths['data_seq_path'], 'hdHand3d')

        self._paths['hd_img_path'] = os.path.join(self._paths['data_seq_path'], 'hdImgs')
        self._paths['calib_path'] = os.path.join(self._paths['data_seq_path'], f'calibration_{seq_name}.json')

        return self._paths


    def get_path_img_seq_info(self, seq: dict, path_img: str) -> dict:
        """
        Generates the complete path for a given image and extracts hd_idx and cam_num.

        Args:
            seq (dict): Dictionary with processing information.
            path_img (str): Image file name.

        Returns:
            dict: Updated dictionary with image path and extracted info.
        """
        seq['path_img'] = os.path.join(self._paths['hdimgs_path'], path_img)
        seq['hd_idx'] = int(path_img.split("_")[-1].split('.')[0])
        seq['cam_num'] = int(path_img.split("_")[-2])

        return seq


    def load_hd_image(self, seq: dict) -> np.ndarray:
        """
        Verifies and loads the HD image.

        Args:
            seq (dict): Dictionary with sequence info containing 'path_img'.

        Returns:
            np.ndarray: Loaded image, or None if not found or invalid.
        """
        img = None
        if os.path.isfile(seq['path_img']):
            img = cv2.imread(seq['path_img'])
            if img is None or img.size == 0:
                print(f"Error: Image could not be opened or is empty: {seq['path_img']}")
                img = None
        else:
            print(f"Error: Image file does not exist: {seq['path_img']}")
        return img


    def sort_cropped_imgs_paths(self, ppl_idx: int) -> List[str]:
        """
        Sorts cropped images by their numerical suffix.

        Args:
            ppl_idx (int): Index representing the person or entity.

        Returns:
            List[str]: A sorted list of cropped image filenames.
        """
        folder = os.path.join(self._paths['save_cropped_imgs_dir'], str(ppl_idx))
        all_cropped_imgs_unsorted = os.listdir(folder)
        all_cropped_imgs = sorted(all_cropped_imgs_unsorted, key=lambda x: int(x.split('_')[1]))
        print("Cropped images dir:",self._paths['save_cropped_imgs_dir'])
        print("Cropped images files:", all_cropped_imgs, "\n")

        return all_cropped_imgs


    def recreate_directories(self) -> None:
        """
        Deletes and recreates necessary output directories.
        """
        paths_dir = [
            self._paths['save_seq_images_folder'],
            self._paths['save_origin_dir'],
            self._paths['save_cropped_imgs_dir']
        ]
        for path_dir in paths_dir:
            if os.path.exists(path_dir):
                shutil.rmtree(path_dir)
            os.makedirs(path_dir)


    def setup_output_paths(self, seq_name: str) -> Tuple[Dict[str, str], str]:
        """
        Sets up the output paths for saving reconstruction results.

        Args:
            seq_name (str): The name of the sequence.

        Returns:
            Tuple[Dict[str, str], str]: Output directories and path to panoptic points.
        """
        panoptic_points_path = f"{self._paths['data_path']}/sequences/{seq_name}/"
        output_paths = {
            'rec': f"{self._paths['data_path']}/{self._config['paths']['annotations']}/3D_annotations/",
            'rep': f"{self._paths['data_path']}/{self._config['paths']['annotations']}/reprojections/",
            'nsrep': f"{self._paths['data_path']}/{self._config['paths']['annotations']}/nsreprojections/"
        }

        for path in output_paths.values():
            os.makedirs(path, exist_ok=True)

        return output_paths, panoptic_points_path
