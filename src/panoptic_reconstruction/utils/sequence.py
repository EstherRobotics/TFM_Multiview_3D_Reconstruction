class SequenceInfo:
    """
    Utility class for initializing and resetting sequence information.
    """

    @staticmethod
    def get_seq_info_init(seq_name: str) -> dict:
        """
        Initializes a sequence information dictionary for a given sequence name.

        Args:
            seq_name (str): Name of the sequence.

        Returns:
            dict: Initialized sequence information dictionary.
        """
        return {
            "name": seq_name,
            "prev_cam_num": 0,
            "cam_num": -1,
            "path_img": None,
            "hd_idx": None,
            "all_ppl_idx": [],
            "x": [],
            "y": [],
            "output_files": [],
            "sel_scams": {},
            "sel_scams_id": {},
            "all_tams": {}
        }


    @staticmethod
    def reset_seq_info(seq_info: dict) -> dict:
        """
        Resets specific fields in the sequence information dictionary.

        Args:
            seq_info (dict): Sequence information to reset.

        Returns:
            dict: Reset sequence information.
        """
        seq_info['all_ppl_idx'] = []
        seq_info['sel_scams'] = {}
        seq_info['sel_scams_id'] = {}
        seq_info['all_tams'] = {}

        return seq_info
