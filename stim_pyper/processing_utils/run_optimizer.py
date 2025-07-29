import os
import json
import numpy as np
import nibabel as nib
from typing import List, Tuple, Dict
from stim_pyper.processing_utils.optimizer_postprocessing import process_vta
from stim_pyper.processing_utils.optimizer_preprocessing import handle_nii_map
from stim_pyper.matlab_utils.mat_reader import MatReader
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
    
    def _get_lead_dbs_electrode(self):
        electrode = MatReader(self.electrode_data).get_reconstructions()[0]
        electrode_coords = {int(key.split('_')[-1]): np.array(electrode['data'][key]) for key in electrode['data'].keys() if key.startswith('coords_')}
        return electrode_coords
    
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

    def optimize_electrode(self, target_coords, coords:list):
        '''Runs optimizer on list of contact coordinates using a list of target coords'''
        try:
            return handle_nii_map(L=np.array(target_coords), sphere_coords=coords, lambda_val=0.0001, weight=1)
        except Exception as e:
            print(f"Error in handle_nii_map: {e}")
            return None
        
    def save_vta(self, optima_ampers, electrode_coords, electrode_idx):
        out_dir = os.path.join(self.output_path, f'electrode_{electrode_idx}')
        os.makedirs(out_dir, exist_ok=True)
        process_vta(optima_ampers, electrode_coords, electrode_idx, out_dir)
        nii_files = [os.path.join(out_dir, file) for file in os.listdir(out_dir) if file.endswith('.nii')]
        return nii_files
    
    def merge_vtas(self, path, out_dir):
        bbox = NiftiBoundingBox(path)
        bbox.gen_mask(out_dir)
        
    def run(self):
        """Processes the data and calls handle_nii_map for each combination of lambda and weight."""
        target_coords = self.nii_to_mni(self.nifti_path)
        electrode_dict = self.get_electrode_dict()
        for electrode_idx in electrode_dict:
            electrode_coords = electrode_dict[electrode_idx]
            optima_ampers = self.optimize_electrode(target_coords, electrode_coords)
            output_direct = self.save_vta(optima_ampers, electrode_coords, electrode_idx)
            self.merge_vtas(output_direct, os.path.join(self.output_path, f'electrode_{electrode_idx}'))
            
if __name__ == "__main__":
    electrode_data = '/path/to/reco.mat'
    nifti_path = '/path/to/nifti.nii'
    output_path = '/path/to/output'
    processor = OptimizerProcessor(electrode_data, nifti_path, output_path)
    processor.run()
