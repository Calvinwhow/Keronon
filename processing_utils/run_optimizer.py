import os
import json
import numpy as np
import nibabel as nib
from typing import List, Tuple, Dict
from processing_utils.optimizer_postprocessing import process_vta
from processing_utils.optimizer_preprocessing import handle_nii_map
from matlab_utils.mat_reader import MatReader
from calvin_utils.ccm_utils.bounding_box import NiftiBoundingBox
from glob import glob

class OptimizerProcessor:
    def __init__(self, electrode_data, nifti_path: str, output_path: str):
        self.electrode_data = electrode_data
        self.nifti_path = nifti_path
        self.output_path = output_path

    def nii_to_mni(self, path) -> List[Tuple[float, float, float, float]]:
        """Reads a NIfTI file and converts it to a list of MNI coordinates with associated r values."""
        img = nib.load(path)
        data = img.get_fdata()
        affine = img.affine
        shape = data.shape
        indices = np.indices(shape[:3]).reshape(3, -1)
        indices = np.vstack((indices, np.ones((1, indices.shape[1]))))
        mni_coords = np.dot(affine, indices)
        values = data.flatten()
        mni_coords = [
            (mni_coords[0, i], mni_coords[1, i], mni_coords[2, i], values[i])
            for i in range(mni_coords.shape[1])
            if not np.isnan(values[i])
        ]
        return mni_coords
    
    def _get_lead_dbs_electrode(self): ## NOTE: Looks like you are trying to get multiple electrodes here. Just assign them to a new dictionary, 
        '''edits to be made by savir'''
        electrode = MatReader(self.reco_path).get_reconstructions() #NOTE:Just extracted this code to internal method and leaving for you. 
        electrode_coords = np.array(reco['data'][coords]) ## NOTE:This is a touch unclean. Should extract the entire list in get_coordinates()
        return [key for key in electrode['data'].keys() if key.startswith('coords_')] # NOTE:try to return a dict where each key is an integere (electrode) and the value is the coordinate list.
    
    def _get_json_electrode(self):
        '''Opens a standard JSON and reads the electrode data in'''
        with open('data.json', 'r') as file:
            return json.load(file)
    
    def get_electrode_dict(self) -> Dict:
        '''Function to receive electrode data of various sources'''
        if isinstance(self.electrode_data, str) and self.electrode_data.lower().endswith('.mat'):
            electrode_dict = self._get_lead_dbs_electrode()
        elif isinstance(self.electrode_data, str) and self.electrode_data.lower().endswith('.json'):
            electrode_dict = self._get_json_electrode()
        else:
            raise ValueError(f"File type not yet supported for file: {self.electrode_data}")
        return electrode_dict

    def optimize_electrode(self, target_coords:List, coords:list):
        '''Runs optimizer on list of contact coordinates using a list of target coords'''
        return handle_nii_map(L=np.array(self.mni_coords), sphere_coords=coords, lambda_val=0.0001, weight=1) #NOTE: I don't really think this handle_nii_map function is robust. It needs to be cleaner--its the critical link between this interface and the optimizer. 
        
    def save_vta(self, optima_ampers, electrode_coords, electrode_idx):
        out_dir = os.path.join(self.output_path, f'electrode_{electrode_idx}')
        os.makedirs(out_dir, exist_ok=True)
        process_vta(optima_ampers, electrode_coords, electrode_idx, out_dir) ##NOTE: This function could be cleaned up. 
        return out_dir
    
    def merge_vtas(self, path):
        bbox = NiftiBoundingBox(glob(path))
        bbox.gen_mask(path)
        
    def run(self):
        """Processes the data and calls handle_nii_map for each combination of lambda and weight."""
        target_coords = self.nii_to_mni(self.nifti_path)
        electrode_dict = self.get_electrode_dict()
        for electrode_idx, electrode_coords in enumerate(electrode_dict):
            optima_ampers = self.optimize_electrode(target_coords, electrode_coords)
            output_direct = self.save_vta(optima_ampers, electrode_coords, electrode_idx)
            self.merge_vtas(output_direct)
            
if __name__ == "__main__":
    reco_path = '/path/to/reconstruction.mat'
    nifti_path = '/path/to/target_map.nii'
    output_path = '/path/to/output'
    processor = OptimizerProcessor(reco_path, nifti_path, output_path)
    processor.run()
