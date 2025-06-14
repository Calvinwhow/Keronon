import os
import json
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple
from processing_utils.optimizer_postprocessing import process_vta
from processing_utils.optimizer_preprocessing import handle_nii_map

def get_reconstructions(base_path: str) -> List[Dict]:
    """
    Gets reconstruction files of all leaddbs patients for batch processing
    """
    json_data = []
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            json_file_path = os.path.join(root, dir_name, 'clinical', f'{dir_name}_desc-reconstruction.json')
            if os.path.isfile(json_file_path):
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    json_data.append({'dir_name': dir_name, 'data': data})
    return json_data

def nii_to_mni(nifti_path: str) -> List[Tuple[float, float, float, float]]:
    """Reads a NIfTI file and converts it to a list of MNI coordinates with values."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    shape = data.shape
    indices = np.indices(shape[:3])
    indices = indices.reshape(3, -1) 
    indices = np.vstack((indices, np.ones((1, indices.shape[1]))))
    mni_coords = np.dot(affine, indices)
    values = data.flatten()
    mni_coords = [
        (mni_coords[0, i], mni_coords[1, i], mni_coords[2, i], values[i])
        for i in range(mni_coords.shape[1])
        if not np.isnan(values[i])
    ]

    return mni_coords

def process_data(base_path: str, mni_coords: List[Tuple[float, float, float, float]]):
    """Processes the data and calls handle_nii_map."""
    reconstructions = get_reconstructions(base_path)

    for reco in reconstructions:
        coords_list = ['coords_right', 'coords_left']
        side = ['right', 'left']
        for i, coords in enumerate(coords_list):
            try:
                electrode_coords = np.array(reco['data'][coords])
                v = handle_nii_map(
                    L=np.array(mni_coords),
                    sphere_coords=electrode_coords,
                )
                vta_file_path = os.path.join(base_path, 'derivatives', 'leaddbs', reco['dir_name'], 'clinical', 'gait', f'{reco["dir_name"]}_vta_{side[i]}.nii')
                # process_vta(v, elec_coords, vta_file_path)
            except Exception as e:
                print(f"Error processing {reco['dir_name']} for {coords}: {e}")
                continue
            
if __name__ == "__main__":
    # Example usage
    base_path = '/Users/savirmadan/Partners HealthCare Dropbox/Savir Madan/datasets/CbctDbs0282'
    nifti_path = '/Users/savirmadan/Downloads/BIDMCBERMDST_nosubj9_Gait_UPDRS_Rand_exch_vox_vstat_pcc_inverse_r_map.nii'
    mni_coords = nii_to_mni(nifti_path)
    process_data(base_path, mni_coords)
