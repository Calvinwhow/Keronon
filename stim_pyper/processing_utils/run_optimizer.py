import os
import json
import numpy as np
import nibabel as nib
from collections import defaultdict
from typing import List, Tuple, Dict
from stim_pyper.processing_utils.optimizer_postprocessing import process_vta
from stim_pyper.processing_utils.optimizer_preprocessing import handle_nii_map
from stim_pyper.matlab_utils.mat_reader import MatReader, MatReaderV2
from stim_pyper.json_utils.json_reader import JsonReader
from calvin_utils.ccm_utils.bounding_box import NiftiBoundingBox
from stim_pyper.vta_model.evaluate_directional_vta import EvaluateDirectionalVta

class OptimizerProcessor:
    def __init__(self, electrode_data_path, nifti_path: str, output_path: str):
        self.electrode_data = electrode_data_path
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
        '''Opens a standard lead dbs reconstruction.mat and reads the electrode data in'''
        electrode_info = MatReaderV2(self.electrode_data).run()
        return electrode_info
    
    def _get_json_electrode(self):
        '''Opens a standard JSON and reads the electrode data in'''
        electrode_info = JsonReader(self.electrode_data).run()
        return electrode_info
    
    def get_electrode_info(self) -> List[Dict]:
        '''Function to receive electrode data of various sources'''
        if isinstance(self.electrode_data, str) and self.electrode_data.lower().endswith('.mat'):
            electrode_info = self._get_lead_dbs_electrode()
        elif isinstance(self.electrode_data, str) and self.electrode_data.lower().endswith('.json'):
            electrode_info = self._get_json_electrode()
        else:
            raise ValueError(f"File type not yet supported for file: {self.electrode_data}")
        return electrode_info

    def optimize_electrode(self, target_coords, electrode_coords, dir_models_list):
        '''Runs optimizer on list of contact coordinates using a list of target coords'''
        try:
            landscape = np.array(target_coords)
            return handle_nii_map(L=landscape, sphere_coords=electrode_coords, lambda_val=0.001, directional_models_list=dir_models_list)
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
        try:
            bbox = NiftiBoundingBox(path)
            bbox.gen_mask(out_dir)
        except ValueError as e:
            print("Electrode was turned off by optimizer. No VTAs to combine/show.")
            
    def get_directional_electrodes(self, electrode_dict):
        '''Extracts info from electrode dict of style: {contact_int: {segment_int: array[x,y,z]}}'''
        n_contacts = len(electrode_dict)
        dir_models_list = [None] * n_contacts           # Pre-fill a list to place EvaluateDirectionalVta objects into
        elec_coords_list = [None] * n_contacts 
        
        # Get seg_map which is a dict which stores data like:
        # seg_map[segment] = [(contact_num, [coord_list]), (contact_num, [coord_list])] 
        seg_map = defaultdict(list)
        for c_idx, seg_dict in electrode_dict.items():
            seg_id, coord_list = next(iter(seg_dict.items()))
            seg_map[seg_id].append((c_idx,coord_list))    # appends the tuple
            elec_coords_list[c_idx] = coord_list          # adds electrode coords to the elec_coords_list
        for _, segment_group in seg_map.items():
            if len(segment_group) <= 1:                   # Leave full segments as None
                continue      
            segment_coords = [coord.tolist() for _, coord in segment_group]      # get the list of coordinates at this segment
            for i, (contact_num, _) in enumerate(segment_group):                  # get          
                vta_model = EvaluateDirectionalVta(contact_coordinates=segment_coords, primary_idx=i)
                dir_models_list[contact_num] = vta_model                          # assign to list
        return dir_models_list, elec_coords_list
        
    def run(self):
        """Processes the data and calls handle_nii_map for each combination of lambda."""
        target_coords = self.nii_to_mni(self.nifti_path)
        electrode_info = self.get_electrode_info()
        for electrode_idx, electrode_dict in enumerate(electrode_info):
            dir_models_list, elec_coords_list = self.get_directional_electrodes(electrode_dict)
            optima_ampers = self.optimize_electrode(target_coords, np.array(elec_coords_list), dir_models_list)
            output_direct = self.save_vta(optima_ampers, elec_coords_list, electrode_idx)
            self.merge_vtas(output_direct, os.path.join(self.output_path, f'electrode_{electrode_idx}'))
        return electrode_info
            
if __name__ == "__main__":
    recon_path = '/path/to/reco.mat'
    target_path = '/path/to/nifti.nii'
    out_dir = '/path/to/output'
    processor = OptimizerProcessor(electrode_data_path=recon_path, nifti_path=target_path, output_path=out_dir)
    processor.run()
