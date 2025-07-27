import os
import json
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple
from processing_utils.optimizer_postprocessing import process_vta
from processing_utils.optimizer_preprocessing import handle_nii_map
from processing_utils.testing import VtaAnalysis
from matlab_utils.mat_reader import MatReader
import glob

class OptimizerProcessor:
    def __init__(self, reco_path: str, nifti_path: str, output_path: str):
        self.reco_path = reco_path
        self.nifti_path = nifti_path
        self.output_path = output_path
        self._mni_coords = None

    @property
    def mni_coords(self) -> List[Tuple[float, float, float, float]]:
        if self._mni_coords is None:
            self._mni_coords = self.nii_to_mni()
        return self._mni_coords

    def nii_to_mni(self) -> List[Tuple[float, float, float, float]]:
        """Reads a NIfTI file and converts it to a list of MNI coordinates with associated r values."""
        img = nib.load(self.nifti_path)
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
    

    def run(self):
        """Processes the data and calls handle_nii_map for each combination of lambda and weight."""
        reconstructions = MatReader(self.reco_path).get_reconstructions()

        for reco in reconstructions:
            coords_list = [key for key in reco['data'].keys() if key.startswith('coords_')]
            for i, coords in enumerate(coords_list):
                try:
                    electrode_coords = np.array(reco['data'][coords])
                    output_dir = os.path.join(self.output_path, f'elec_{i}')
                    os.makedirs(self.output_path, exist_ok=True)
                    v = handle_nii_map(
                        L=np.array(self.mni_coords),
                        sphere_coords=electrode_coords,
                        lambda_val=0.001,
                        weight=10,
                    )
                    process_vta(v, electrode_coords, i, output_dir)
                    analysis = VtaAnalysis(folder_path=output_dir, target_path=self.nifti_path, dice_path = self.output_path)
                    correlation = analysis.run_analysis('combined_image.nii')
                except Exception as e:
                    print(f"Error processing {reco['dir_name']} for {coords}: {e}")
                    continue

if __name__ == "__main__":
    reco_path = '/path/to/reconstruction.mat'
    nifti_path = '/path/to/target_map.nii'
    output_path = '/path/to/output'
    processor = OptimizerProcessor(reco_path, nifti_path, output_path)
    processor.run()
